# Utility functions extracted from the notebook
# English comments/docstrings added.

import re
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from peft import PeftModel
import torch


class TokenizerWrapper:
    """
    Small adaptor so text-only tokenizers behave like processors used elsewhere:
    exposes `.tokenizer`, forwards attributes/methods, and supports save_pretrained.
    """

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, *args, **kwargs):
        return self.tokenizer(*args, **kwargs)

    def apply_chat_template(self, *args, **kwargs):
        return self.tokenizer.apply_chat_template(*args, **kwargs)

    def save_pretrained(self, *args, **kwargs):
        return self.tokenizer.save_pretrained(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)


def _is_multimodal_processor(processor) -> bool:
    """Heuristic: Qwen2-VL AutoProcessor exposes image processing bits."""
    return hasattr(processor, "image_processor") or hasattr(processor, "vision_processor")


def build_user_only_messages(processor, instruction: str) -> List[Dict[str, Any]]:
    """Construct a user-only chat message suitable for text or VL processors."""
    if _is_multimodal_processor(processor):
        return [{"role": "user", "content": [{"type": "text", "text": instruction}]}]
    return [{"role": "user", "content": instruction}]


def example_to_messages(ex: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Normalize a sample into ShareGPT-style messages."""
    # Already in messages form
    msgs = ex.get("messages")
    if isinstance(msgs, list) and msgs and isinstance(msgs[0], dict) and "role" in msgs[0]:
        return msgs

    # Common ShareGPT export format: conversations with `from`/`value`
    convs = ex.get("conversations") or ex.get("conversation")
    if isinstance(convs, list) and convs:
        out: List[Dict[str, Any]] = []
        for turn in convs:
            role = turn.get("from") or turn.get("role")
            content = turn.get("value") or turn.get("content")
            if not role or not content:
                continue
            role_norm = "user" if role in ("human", "user") else "assistant"
            out.append({"role": role_norm, "content": content})
        if out:
            return out

    return alpaca_to_sharegpt(ex)


class TextSFTDataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]]):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        ex = self.data[idx]
        return {"messages": example_to_messages(ex)}
class TextSFTCollator:
    def __init__(self, processor, max_length: int = 4096):
        self.processor = processor
        self.tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        self.max_length = max_length
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
    def __call__(self, batch: List[Dict[str, Any]]):
        conversations = [item["messages"] for item in batch]
        texts = [
            self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            for msgs in conversations
        ]
        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=self.max_length is not None,
            max_length=self.max_length,
        )
        labels = enc["input_ids"].clone()
        pad_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc


def generate_candidates(
    model,
    processor,
    instruction: str,
    num_candidates: int,
    generation_kwargs: Dict,
    device: Optional[torch.device] = None,
) -> List[str]:
    """Generate multiple candidate textual responses for a instruction.

    Args:
        model: The model object exposing `generate`.
        processor: The processor/tokenizer wrapper used to prepare inputs.
        instruction: Instruction text.
        num_candidates: Number of distinct candidate generations to produce.
        generation_kwargs: Extra kwargs forwarded to `model.generate`.
        device: Optional device to move tensors to. If None, no device move is performed.

    Returns:
        A list of decoded text strings (one per candidate).
    """
    candidates: List[str] = []
    tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    apply_template = processor.apply_chat_template if hasattr(processor, "apply_chat_template") else tokenizer.apply_chat_template
    msgs = build_user_only_messages(processor, instruction)
    prompt = apply_template(msgs, tokenize=False, add_generation_prompt=True)
    for _ in range(num_candidates):
        inputs = processor(text=prompt, return_tensors="pt")
        if device is not None:
            inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(**inputs, tokenizer=tokenizer, **generation_kwargs)
        gen = out[:, inputs["input_ids"].shape[1]:]
        candidates.append(tokenizer.decode(gen[0], skip_special_tokens=True))

    return candidates


def judge_response(
    model,
    tokenizer,
    instruction: str,
    response: str,
    reward_prompt_template: str,
    reward_score_regex: re.Pattern,
    n_votes: int = 3,
    device: Optional[torch.device] = None,
) -> Tuple[float, str]:
    """Ask a model to judge a response according to a provided template and parse the score.

    Args:
        model: The model used as the judge (exposes `generate`).
        tokenizer: Tokenizer that converts prompts to tensors and decodes outputs.
        instruction: The user's instruction / prompt text.
        response: Candidate response text to be judged.
        reward_prompt_template: A formatting template that must include placeholders
                                `{instruction}` and `{response}`.
        reward_score_regex: A compiled regex used to extract the numeric score from
                            the judge's output.
        n_votes: Number of stochastic samples to draw from the judge. The final
                 score is the mean of extracted scores from all returned sequences
                 (NaN is returned if none parse).
        device: Optional device to move tensors to.

    Returns:
        Tuple (mean_score, join_of_raw_judgements). mean_score is `float` or
        `math.nan` when no score could be parsed. The second element is the
        aggregated raw judge outputs joined with double newlines.
    """
    prompt = reward_prompt_template.format(instruction=instruction, response=response)
    inputs = tokenizer(text=prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) and device is not None else v) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            num_return_sequences=n_votes,
        )

    dec_start = inputs["input_ids"].shape[1]
    texts = [tokenizer.decode(seq[dec_start:], skip_special_tokens=True).strip() for seq in outputs]

    scores: List[float] = []
    for t in texts:
        m = reward_score_regex.search(t)
        if m:
            try:
                scores.append(float(m.group(1)))
            except Exception:
                pass

    mean_score = float(np.mean(scores)) if scores else math.nan
    return mean_score, "\n\n".join(texts)

# --- File & DPO utility helpers ---

def to_float(x: Any) -> Optional[float]:
    """Safely convert a value to a finite float, otherwise return None."""
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except Exception:
        return None


def iter_records(path: Path):
    """Yield JSON records from a `.jsonl` or `.json` file.

    If the file is a JSON list, each element is yielded. If it is a JSONL file,
    each non-empty line is parsed and yielded.

    Args:
        path: Path to the input file.
    """
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    elif path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for rec in data:
                yield rec
        else:
            yield data
    else:
        raise ValueError(f"Unsupported file type: {path.suffix}")



def build_dpo_entry(rec: Dict, include_scores: bool = False) -> Optional[Dict]:
    """Convert a raw record containing candidates into a cleaned DPO pair.

    The function expects the input `rec` to have keys `prompt` (or `instruction` or
    `query`), `images`, and `candidates` (list of dicts with `response` and
    `score`). It returns a dict with `prompt`, `images`, `chosen`, `rejected`, and
    optional scores when `include_scores` is True.

    Returns None when the record is invalid, has fewer than two valid scored
    candidates, or when the best and worst scores are tied within `TIE_EPS`.
    """
    prompt = rec.get("prompt") or rec.get("instruction") or rec.get("query")
    images = rec.get("images", [])
    cands = rec.get("candidates", [])
    TIE_EPS = 1e-6 # Change this value to adjust tie sensitivity

    if not prompt or not isinstance(cands, list) or len(cands) < 2:
        return None

    # filter out candidates with non-finite scores or empty responses
    valid: List[Dict[str, Any]] = []
    for c in cands:
        score = to_float(c.get("score"))
        if score is None:
            continue
        resp = c.get("response")
        if not isinstance(resp, str) or not resp.strip():
            continue
        valid.append({"response": resp.strip(), "score": score})

    if len(valid) < 2:
        return None

    # pick best and worst
    valid.sort(key=lambda z: z["score"], reverse=True)
    best = valid[0]
    worst = valid[-1]

    # drop only when responses collapse to the same text; allow equal scores if texts differ (useful for small/demo runs)
    if best["response"] == worst["response"]:
        return None

    out = {
        "prompt": prompt,
        "images": images,
        "chosen": best["response"],
        "rejected": worst["response"],
    }
    if include_scores:
        out["chosen_score"] = best["score"]
        out["rejected_score"] = worst["score"]
    return out

def alpaca_to_sharegpt(ex):
    if ex.get("input"):
        user_text = ex["instruction"] + "\n\n" + ex["input"]
    else:
        user_text = ex["instruction"]

    messages = [
        {"role": "user", "content": user_text},
        {"role": "assistant", "content": ex["output"]},
    ]
    return messages

def load_policy_with_lora(base_model, adapter_path: Path):
    base_model.gradient_checkpointing_enable()
    return PeftModel.from_pretrained(base_model, adapter_path.as_posix())

def evaluate_generation_metrics(
    model,
    processor,
    test_samples: List[Dict],
    max_new_tokens: int = 128,
    max_samples: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> Dict[str, Any]:
    """Generate predictions for `test_samples` and compute ROUGE & BLEU metrics.

    Each `sample` is expected to contain the conversational structure used
    in the notebook (i.e. `sample['messages'][0]['content']` is the user
    input and `sample['messages'][1]['content']` is the reference).

    Returns a dictionary with averaged metrics (0-100 scale) and raw lists.
    """
    # local imports to avoid hard dependency at module import time
    from rouge_score import rouge_scorer as _rouge_scorer
    from sacrebleu import corpus_bleu as _corpus_bleu

    scorer = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougeL_scores: List[float] = []

    preds: List[str] = []
    refs: List[str] = []

    model.eval()
    limit = min(len(test_samples), max_samples) if max_samples is not None else len(test_samples)

    for i in range(limit):
        messages = test_samples[i]
        if not messages or not isinstance(messages, list) or len(messages) < 2:
            continue

        reference = messages[1]["content"]
        apply_template = processor.apply_chat_template if hasattr(processor, "apply_chat_template") else processor.tokenizer.apply_chat_template
        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        text = apply_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=text, padding=True, return_tensors="pt")
        inputs = {k: (v.to(device) if device is not None and isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )

        # drop the prompt part from the decoded sequence
        dec_start = inputs["input_ids"].shape[1]
        prediction = tokenizer.decode(generated_tokens[0][dec_start:], skip_special_tokens=True).strip()

        preds.append(prediction)
        refs.append(reference)

        r = scorer.score(reference, prediction)
        rouge1_scores.append(r["rouge1"].fmeasure * 100)
        rouge2_scores.append(r["rouge2"].fmeasure * 100)
        rougeL_scores.append(r["rougeL"].fmeasure * 100)

    # compute corpus BLEU (sacrebleu expects list-of-references as outer list)
    bleu = float('nan')
    if preds and refs:
        bleu_obj = _corpus_bleu(preds, [refs])
        bleu = float(bleu_obj.score)

    results = {
        "rouge1_mean": float(sum(rouge1_scores) / len(rouge1_scores)) if rouge1_scores else float('nan'),
        "rouge2_mean": float(sum(rouge2_scores) / len(rouge2_scores)) if rouge2_scores else float('nan'),
        "rougeL_mean": float(sum(rougeL_scores) / len(rougeL_scores)) if rougeL_scores else float('nan'),
        "bleu4": bleu,
        "rouge1_list": rouge1_scores,
        "rouge2_list": rouge2_scores,
        "rougeL_list": rougeL_scores,
        "predictions": preds,
        "references": refs,
    }
    return results


def to_chat_prompt(processor, text: str) -> str:
    msgs = build_user_only_messages(processor, text)
    apply_template = processor.apply_chat_template if hasattr(processor, "apply_chat_template") else processor.tokenizer.apply_chat_template
    return apply_template(msgs, tokenize=False, add_generation_prompt=True)

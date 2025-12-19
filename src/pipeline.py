"""
End-to-end pipeline rewritten from the notebook:
- Supervised fine-tuning (LoRA)
- Self-rewarding data generation
- DPO training
"""

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import List, Tuple, Any, Dict, Optional
from contextlib import nullcontext
import os

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model

from src.config import PathConfig, LoraHyperParams, SFTConfig, DPOTrainingConfig, GRPOTrainingConfig, GSPOTrainingConfig
from utils import (
    TextSFTDataset,
    TextSFTCollator,
    alpaca_to_sharegpt,
    example_to_messages,
    build_dpo_entry,
    evaluate_generation_metrics,
    generate_candidates,
    iter_records,
    judge_response,
    load_policy_with_lora,
    to_chat_prompt,
    to_float,
    TokenizerWrapper,
)


PROMPT_GEN_TEMPLATE = (
    "You are an expert meteorologist creating challenging user instructions that require analyzing weather maps."
    "Below are example instructions (After User:) and responses(After Assistant:). Write a new instruction in a similar style that asks for a structured weather analysis."
    "Return only the new instruction. Do not include 'User:'. The new instruction should be concise and not exceed 30 words. "
    "{few_shot_block}"
    "New instruction:"
)

REWARD_PROMPT_TEMPLATE = (
    "Review the user's question and the corresponding response using the additive 5-point scoring system described below. "
    "Points are accumulated based on the satisfaction of each criterion: - Add 1 point if the response is relevant and provides some information related to the user's inquiry, even if it is incomplete or contains some irrelevant content. "
    "- Add another point if the response addresses a substantial portion of the user's question, but does not completely resolve the query or provide a direct answer. "
    "- Award a third point if the response answers the basic elements of the user's question in a useful way, regardless of whether it seems to have been written by an AI Assistant or if it has elements typically found in blogs or search results. "
    "- Grant a fourth point if the response is clearly written from an AI Assistant's perspective, addressing the user's question directly and comprehensively, and is well-organized and helpful, even if there is slight room for improvement in clarity, conciseness or focus. "
    "- Bestow a fifth point for a response that is impeccably tailored to the user's question by an AI Assistant, without extraneous information, reflecting expert knowledge, and demonstrating a high-quality, engaging, and insightful answer. "
    "User: {instruction}<response>{response}</response>"
    "After examining the user's instruction and the response:"
    "- First, conclude with exactly one line using the format: Score: <0-5>."
    "- Second, briefly justify your total score, up to 50 words."
    "You should not exceed the word limit of 50. "
    "Remember to assess from the AI Assistant perspective, utilizing web search knowledge as necessary. To evaluate the response in alignment with this additive scoring model, we will systematically attribute points based on the outlined criteria."
)

REWARD_SCORE_REGEX = re.compile(r"Score[:：]\s*([0-5](?:\.\d+)?)(?:\s*/\s*[0-5](?:\.\d+)?)?")

# Defaults
DEFAULT_NUM_CANDIDATES = 5
DEFAULT_N_VOTES = 3
DEMO_TRAIN_LIMIT = 8   # 固定 demo 使用的训练样本数（基于 mini 数据集）
DEMO_TEST_LIMIT = 2     # 可根据需要调整/减小
DEMO_TRAIN_FRACTION = 1.0  # 不再按比例裁剪，直接用上面的固定数
DEMO_TEST_FRACTION = 1.0
DEMO_EPOCHS = 1
DEMO_NUM_CANDIDATES = 3
DEMO_N_VOTES = 2
DEFAULT_GEN_MAX_NEW_TOKENS = 512
DEMO_GEN_MAX_NEW_TOKENS = 256
SFT_DATASET_CHOICES = ("alpaca", "sharegpt", "openorca", "self-instruct")
DPO_DATASET_CHOICES = ("self-reward", "pku")
GRPO_DATASET_CHOICES = ("self-reward", "ultrafeedback")


def _configure_sft_paths(paths: PathConfig, dataset: str, mode: str):
    dataset = dataset.lower()
    if dataset == "alpaca":
        paths.sft_training_path = Path("data/alpaca_en_mini.json") if mode == "demo" else Path("data/alpaca_en_train.json")
        paths.test_path = Path("data/alpaca_en_test.json")
    elif dataset == "sharegpt":
        paths.sft_training_path = paths.sharegpt_sft_path
        paths.test_path = paths.sharegpt_sft_path
    elif dataset == "openorca":
        paths.sft_training_path = paths.openorca_sft_path
        paths.test_path = paths.openorca_sft_path
    elif dataset == "self-instruct":
        paths.sft_training_path = paths.self_instruct_sft_path
        paths.test_path = paths.self_instruct_sft_path
    else:
        raise ValueError(f"Unsupported sft_dataset={dataset}")


def _configure_dpo_path(paths: PathConfig, dataset: str):
    dataset = dataset.lower()
    if dataset == "self-reward":
        return
    if dataset == "pku":
        paths.dpo_training_path = paths.pku_dpo_path
        return
    raise ValueError(f"Unsupported dpo_dataset={dataset}")


def _configure_grpo_path(paths: PathConfig, dataset: str):
    dataset = dataset.lower()
    if dataset == "self-reward":
        return
    if dataset == "ultrafeedback":
        paths.grpo_training_path = paths.ultrafeedback_grpo_path
        return
    raise ValueError(f"Unsupported grpo_dataset={dataset}")


def _extract_user_prompt(messages: List[Dict[str, Any]]) -> Optional[str]:
    """Return the first user text block from a conversation."""
    for msg in messages:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            texts = [c.get("text") for c in content if isinstance(c, dict) and c.get("type") == "text" and c.get("text")]
            if texts:
                return "\n".join(texts)
        elif isinstance(content, str):
            return content
    return None


def _normalize_model_id_or_path(model_path: Path) -> str:
    """Return a HF-friendly identifier. If local dir exists, return its POSIX path; otherwise normalize slashes."""
    if model_path.exists():
        return model_path.resolve().as_posix()
    return str(model_path).replace("\\", "/")


def _init_distributed(backend: str = "nccl"):
    """Initialize torch.distributed if env var WORLD_SIZE > 1."""
    if dist.is_available() and not dist.is_initialized():
        world_size = int(os.environ.get("WORLD_SIZE", "1"))
        if world_size > 1:
            dist.init_process_group(backend=backend)


def _is_dist():
    return dist.is_available() and dist.is_initialized()


def _get_rank():
    return dist.get_rank() if _is_dist() else 0


def _get_world_size():
    return dist.get_world_size() if _is_dist() else 1


def _precision_to_dtype(precision: str):
    precision = (precision or "none").lower()
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    return None


def _make_autocast(device: torch.device, amp_dtype):
    if device is not None and device.type == "cuda" and amp_dtype is not None:
        return torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    return nullcontext()


def _make_grad_scaler(device: torch.device, amp_dtype):
    enabled = device is not None and device.type == "cuda" and amp_dtype == torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:
        try:
            return torch.amp.GradScaler(enabled=enabled)
        except Exception:
            return torch.cuda.amp.GradScaler(enabled=enabled)


def _log_metric(stage: str, step: int, loss: float, epoch: int = 0, reports_dir: Path = Path("reports")):
    """Append a training loss metric to a jsonl file (rank0 only)."""
    if _get_rank() != 0:
        return
    reports_dir.mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, "epoch": epoch, "step": step, "loss": float(loss)}
    with (reports_dir / f"{stage}_loss.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def load_base_model_and_processor(model_path: Path, token: str = None):
    model_id = _normalize_model_id_or_path(model_path)
    torch_dtype = "auto"
    # Heuristic: try VL first when the id suggests it, otherwise fall back to text-only causal LM.
    prefer_vl = "vl" in model_id.lower()
    if prefer_vl:
        try:
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id, dtype=torch_dtype, device_map=None, token=token
            )
            processor = AutoProcessor.from_pretrained(model_id, token=token)
            return base_model, processor
        except Exception as exc:  # noqa: F841
            print(f"[load] Falling back to text-only AutoModelForCausalLM for {model_id}")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch_dtype, device_map=None, token=token
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return base_model, TokenizerWrapper(tokenizer)


def load_sft_data(train_path: Path, test_path: Path) -> Tuple[List[dict], List[dict]]:
    train_samples = list(iter_records(train_path))
    test_samples = list(iter_records(test_path))
    test_messages = [example_to_messages(ex) for ex in test_samples]
    return train_samples, test_messages


def detect_precision_flags() -> Tuple[bool, bool]:
    bf16_flag = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
    fp16_flag = torch.cuda.is_available() and not bf16_flag
    return bf16_flag, fp16_flag


def build_lora_model(base_model, lora_cfg: LoraHyperParams):
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    lora_config = LoraConfig(
        r=lora_cfg.r,
        lora_alpha=lora_cfg.lora_alpha,
        target_modules=lora_cfg.target_modules,
        lora_dropout=lora_cfg.lora_dropout,
        bias=lora_cfg.bias,
        task_type=lora_cfg.task_type,
    )
    return get_peft_model(base_model, lora_config)


def move_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    if device is None:
        return batch
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}


def get_primary_device(model) -> torch.device:
    try:
        return next(iter(model.parameters())).device
    except StopIteration:
        return torch.device("cpu")


def wrap_ddp(model, device_ids: Optional[List[int]] = None):
    if not _is_dist():
        return model
    # avoid import cycles for type checking
    return torch.nn.parallel.DistributedDataParallel(model, device_ids=device_ids, output_device=device_ids[0] if device_ids else None, find_unused_parameters=False)


def run_sft_training(
    paths: PathConfig,
    lora_cfg: LoraHyperParams,
    sft_cfg: SFTConfig,
    max_eval_samples: int = 5,
    hf_token: str = None,
    precision: str = "none",
    max_train_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
    max_train_fraction: float = 1.0,
    max_test_fraction: float = 1.0,
):
    _init_distributed()
    rank = _get_rank()
    world_size = _get_world_size()

    train_samples, test_messages = load_sft_data(paths.sft_training_path, paths.test_path)
    if max_train_fraction is not None and max_train_fraction < 1.0:
        limit = max(1, int(len(train_samples) * max_train_fraction))
        train_samples = train_samples[:limit]
    if max_train_samples is not None:
        train_samples = train_samples[:max_train_samples]
    if max_test_fraction is not None and max_test_fraction < 1.0:
        limit_test = max(1, int(len(test_messages) * max_test_fraction))
        test_messages = test_messages[:limit_test]
    if max_test_samples is not None:
        test_messages = test_messages[:max_test_samples]
    base_model, processor = load_base_model_and_processor(paths.model_path, token=hf_token)
    policy_model = build_lora_model(base_model, lora_cfg)

    # device / DDP
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        policy_model = policy_model.to(local_rank)
        device = torch.device(local_rank)
        device_ids = [local_rank]
    else:
        device = get_primary_device(policy_model)
        device_ids = None

    policy_model = wrap_ddp(policy_model, device_ids=device_ids)
    amp_dtype = _precision_to_dtype(precision) if device.type == "cuda" else None
    autocast_ctx = _make_autocast(device, amp_dtype)
    scaler = _make_grad_scaler(device, amp_dtype)
    sft_dataset = TextSFTDataset(train_samples)
    sft_collator = TextSFTCollator(processor, max_length=sft_cfg.max_length)
    sampler = DistributedSampler(sft_dataset, num_replicas=world_size, rank=rank, shuffle=True) if _is_dist() else None
    dataloader = DataLoader(
        sft_dataset,
        batch_size=sft_cfg.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=sft_collator,
    )

    batches_per_epoch = max(1, len(dataloader))
    updates_per_epoch = max(1, math.ceil(batches_per_epoch / sft_cfg.gradient_accumulation_steps))
    total_steps = updates_per_epoch * sft_cfg.num_train_epochs
    warmup_steps = max(1, int(0.03 * total_steps))

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=sft_cfg.learning_rate, weight_decay=sft_cfg.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    policy_model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(sft_cfg.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader, 1):
            batch = move_to_device(batch, device)
            with autocast_ctx:
                outputs = policy_model(**batch)
                loss = outputs.loss / sft_cfg.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % sft_cfg.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if rank == 0 and global_step % sft_cfg.logging_steps == 0:
                    print(f"[sft] epoch {epoch+1} step {global_step} loss={loss.item() * sft_cfg.gradient_accumulation_steps:.4f}")
                    _log_metric("sft", global_step, loss.item() * sft_cfg.gradient_accumulation_steps, epoch=epoch + 1)

    if rank == 0:
        paths.sft_adapter_path.parent.mkdir(parents=True, exist_ok=True)
        # unwrap DDP for saving
        to_save = policy_model.module if hasattr(policy_model, "module") else policy_model
        to_save.save_pretrained(paths.sft_adapter_path.as_posix())
        processor.save_pretrained(paths.sft_adapter_path.as_posix())

    # rank0 only for eval/log to avoid duplication
    if rank == 0:
        eval_results = evaluate_generation_metrics(
            model=policy_model.module if hasattr(policy_model, "module") else policy_model,
            processor=processor,
            test_samples=test_messages,
            max_new_tokens=128,
            max_samples=max_eval_samples,
            device=device,
        )
        print("SFT evaluation:", eval_results)
    return policy_model, processor, train_samples, test_messages


def generate_self_rewarding_data(
    policy_model,
    processor,
    train_samples: List[dict],
    paths: PathConfig,
    num_candidates: int = 4,
    n_votes: int = 3,
    generation_kwargs: Optional[Dict] = None,
    sample_log_limit: int = 20,
    sample_log_path: Optional[Path] = None,
):
    rank = _get_rank()
    world_size = _get_world_size()
    shard = train_samples[rank::world_size] if world_size > 1 else train_samples
    device = get_primary_device(policy_model)
    dpo_records: List[Dict[str, Any]] = []
    random.seed(42 + rank)
    sample_buffer: List[Dict[str, Any]] = [] if rank == 0 else []
    sample_seen = 0
    sample_log_path = sample_log_path or paths.raw_response_path.with_name("self_reward_samples.jsonl")
    # keep a full log (one per instruction) even if later dropped, for inspection
    full_raw: List[Dict[str, Any]] = [] if rank == 0 else []

    iterator = tqdm(shard, desc="Building self-rewarding DPO data") if rank == 0 else shard
    for ex in iterator:
        messages = example_to_messages(ex)
        instruction = _extract_user_prompt(messages)
        if not instruction:
            continue
        candidate_responses = generate_candidates(
            model=policy_model,
            processor=processor,
            instruction=instruction,
            num_candidates=num_candidates,
            generation_kwargs=generation_kwargs or dict(max_new_tokens=DEFAULT_GEN_MAX_NEW_TOKENS, temperature=0.7, top_p=0.9, do_sample=True),
            device=device,
        )

        scored_candidates = []
        for candidate in candidate_responses:
            score, judgement = judge_response(
                model=policy_model,
                tokenizer=processor,
                instruction=instruction,
                response=candidate,
                reward_prompt_template=REWARD_PROMPT_TEMPLATE,
                reward_score_regex=REWARD_SCORE_REGEX,
                n_votes=n_votes,
                device=device,
            )
            scored_candidates.append({"response": candidate, "score": score, "judge_output": judgement})

        valid_candidates = [c for c in scored_candidates if not math.isnan(c["score"])]

        # log all instructions for inspection, even if dropped
        if rank == 0:
            full_raw.append(
                {
                    "instruction": instruction,
                    "candidates": scored_candidates,
                    "has_valid_pair": len(valid_candidates) >= 2,
                }
            )

        if len(valid_candidates) < 2:
            continue

        sorted_candidates = sorted(valid_candidates, key=lambda c: c["score"], reverse=True)
        best = sorted_candidates[0]
        worst = sorted_candidates[-1]

        dpo_records.append(
            {
                "instruction": instruction,
                "candidates": scored_candidates,
                "chosen": best["response"],
                "chosen_score": best["score"],
                "rejected": worst["response"],
                "rejected_score": worst["score"],
            }
        )

        if rank == 0 and sample_log_limit > 0:
            sample_seen += 1
            sample_entry = {
                "instruction": instruction,
                "candidates": scored_candidates,
                "chosen": best["response"],
                "chosen_score": best["score"],
                "rejected": worst["response"],
                "rejected_score": worst["score"],
            }
            if len(sample_buffer) < sample_log_limit:
                sample_buffer.append(sample_entry)
            else:
                j = random.randint(0, sample_seen - 1)
                if j < sample_log_limit:
                    sample_buffer[j] = sample_entry

    if _is_dist() and world_size > 1:
        gathered: List[List[Dict[str, Any]]] = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, dpo_records)
        if rank == 0:
            dpo_records = [rec for shard_records in gathered for rec in shard_records]

    if rank == 0:
        paths.raw_response_path.parent.mkdir(parents=True, exist_ok=True)
        with paths.raw_response_path.open("w", encoding="utf-8") as f:
            for item in dpo_records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(dpo_records)} raw DPO pairs to {paths.raw_response_path}")

        if full_raw:
            full_path = paths.raw_response_path.with_name("raw_responses_full.jsonl")
            with full_path.open("w", encoding="utf-8") as f:
                for item in full_raw:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[raw-full] wrote {len(full_raw)} instructions to {full_path}")

        if sample_buffer:
            sample_log_path.parent.mkdir(parents=True, exist_ok=True)
            with sample_log_path.open("w", encoding="utf-8") as f:
                for item in sample_buffer:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            print(f"[sample] kept {len(sample_buffer)}/{sample_seen} examples -> {sample_log_path}")

    return dpo_records


def clean_self_rewarding_data(paths: PathConfig, include_scores: bool = False, tie_eps: float = 1e-6):
    if _get_rank() != 0:
        return 0
    paths.dpo_training_path.parent.mkdir(parents=True, exist_ok=True)
    total = kept = dropped_nan = dropped_tie = dropped_other = 0

    with paths.dpo_training_path.open("w", encoding="utf-8") as g:
        for rec in iter_records(paths.raw_response_path):
            total += 1
            cands = rec.get("candidates", [])
            if isinstance(cands, list) and any(to_float(c.get("score")) is None for c in cands):
                dropped_nan += 1

            dpo = build_dpo_entry(rec, include_scores=include_scores)
            if dpo is None:
                vals = [to_float(c.get("score")) for c in cands]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2 and (max(vals) - min(vals) <= tie_eps):
                    dropped_tie += 1
                else:
                    dropped_other += 1
                continue

            g.write(json.dumps(dpo, ensure_ascii=False) + "\n")
            kept += 1

    print(f"[clean] total={total}, kept={kept}, had_nan={dropped_nan}, dropped_tie={dropped_tie}, dropped_other={dropped_other}")
    print(f"[out] {paths.dpo_training_path}")
    return kept


def build_grpo_dataset(paths: PathConfig):
    """Flatten raw self-rewarding candidates into (prompt, response, reward) triples."""
    if _get_rank() != 0:
        return 0
    paths.grpo_training_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    written = 0
    with paths.grpo_training_path.open("w", encoding="utf-8") as g:
        for rec in iter_records(paths.raw_response_path):
            prompt = rec.get("instruction") or rec.get("prompt") or rec.get("query")
            if not prompt:
                continue
            for cand in rec.get("candidates", []):
                total += 1
                score = to_float(cand.get("score"))
                if score is None:
                    continue
                resp = cand.get("response")
                if not isinstance(resp, str) or not resp.strip():
                    continue
                out = {"prompt": prompt, "response": resp.strip(), "reward": score}
                g.write(json.dumps(out, ensure_ascii=False) + "\n")
                written += 1
    if _get_rank() == 0:
        print(f"[grpo] flattened {written}/{total} candidates to {paths.grpo_training_path}")
    return written


def build_gspo_dataset(paths: PathConfig, temperature: float = 1.0):
    """Aggregate candidates per prompt, convert scores to soft weights via softmax(score/temperature)."""
    if _get_rank() != 0:
        return 0
    paths.gspo_training_path.parent.mkdir(parents=True, exist_ok=True)
    by_prompt: Dict[str, List[Tuple[str, float]]] = {}
    for rec in iter_records(paths.raw_response_path):
        prompt = rec.get("instruction") or rec.get("prompt") or rec.get("query")
        if not prompt:
            continue
        for cand in rec.get("candidates", []):
            score = to_float(cand.get("score"))
            resp = cand.get("response")
            if score is None or not isinstance(resp, str) or not resp.strip():
                continue
            by_prompt.setdefault(prompt, []).append((resp.strip(), score))

    written = 0
    with paths.gspo_training_path.open("w", encoding="utf-8") as g:
        for prompt, pairs in by_prompt.items():
            if not pairs:
                continue
            scores = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
            weights = torch.softmax(scores / max(temperature, 1e-6), dim=0).tolist()
            record = {
                "prompt": prompt,
                "responses": [p[0] for p in pairs],
                "weights": weights,
            }
            g.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
    if _get_rank() == 0:
        print(f"[gspo] wrote {written} prompts to {paths.gspo_training_path}")
    return written


class DPODataset(Dataset):
    def __init__(self, records: List[Dict[str, str]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return rec["prompt"], rec["chosen"], rec["rejected"]


class GRPODataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return rec["prompt"], rec["response"], rec["reward"]


class GSPODataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return rec["prompt"], rec["responses"], rec["weights"]


class GRPODataset(Dataset):
    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        return rec["prompt"], rec["response"], rec["reward"]


def compute_logps(
    model,
    tokenizer,
    prompts: List[str],
    responses: List[str],
    device: torch.device,
    max_length: int = 1024,
    no_grad: bool = False,
    autocast_ctx=None,
) -> torch.Tensor:
    """Compute log-probabilities of responses given prompts."""
    logps: List[torch.Tensor] = []
    ctx = torch.no_grad() if no_grad else torch.enable_grad()
    outer = autocast_ctx if autocast_ctx is not None else nullcontext()

    with ctx, outer:
        for prompt, response in zip(prompts, responses):
            prompt_ids = tokenizer(prompt, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
            response_ids = tokenizer(response, add_special_tokens=False, truncation=True, max_length=max_length).input_ids
            input_ids = torch.tensor([prompt_ids + response_ids], device=device)
            attn = torch.ones_like(input_ids, device=device)

            outputs = model(input_ids=input_ids, attention_mask=attn)
            logits = outputs.logits[:, :-1, :]
            labels = input_ids[:, 1:]

            log_probs = torch.log_softmax(logits, dim=-1)
            token_logps = torch.gather(log_probs, 2, labels.unsqueeze(-1)).squeeze(-1)

            prompt_len = len(prompt_ids)
            start = max(prompt_len - 1, 0)
            resp_mask = torch.zeros_like(labels, dtype=torch.bool, device=device)
            resp_mask[:, start:] = True

            seq_logp = (token_logps * resp_mask).sum(dim=-1)
            logps.append(seq_logp)

    return torch.cat(logps, dim=0) if logps else torch.tensor([], device=device)


def load_dpo_dataset(paths: PathConfig, processor) -> DPODataset:
    manual_path = paths.dpo_training_path.with_name("dpo_training_manual.jsonl")
    load_path = paths.dpo_training_path if paths.dpo_training_path.exists() else manual_path
    raw_pairs = list(iter_records(load_path))
    rows: List[Dict[str, str]] = []
    for rec in raw_pairs:
        prompt = rec.get("prompt") or rec.get("instruction") or rec.get("query")
        chosen, rejected = rec.get("chosen"), rec.get("rejected")
        if not prompt or not chosen or not rejected:
            continue
        rows.append({"prompt": to_chat_prompt(processor, prompt), "chosen": chosen, "rejected": rejected})
    if not rows:
        raise ValueError(f"No valid DPO rows loaded from {load_path}")
    return DPODataset(rows)


def load_grpo_dataset(paths: PathConfig) -> GRPODataset:
    raw = list(iter_records(paths.grpo_training_path))
    rows: List[Dict[str, Any]] = []
    for rec in raw:
        p, r, rew = rec.get("prompt"), rec.get("response"), rec.get("reward")
        if not p or not isinstance(r, str) or not isinstance(rew, (int, float)):
            continue
        rows.append({"prompt": p, "response": r, "reward": float(rew)})
    if not rows:
        raise ValueError(f"No valid GRPO rows in {paths.grpo_training_path}")
    return GRPODataset(rows)


def load_gspo_dataset(paths: PathConfig) -> GSPODataset:
    raw = list(iter_records(paths.gspo_training_path))
    rows: List[Dict[str, Any]] = []
    for rec in raw:
        p, responses, weights = rec.get("prompt"), rec.get("responses"), rec.get("weights")
        if not p or not isinstance(responses, list) or not isinstance(weights, list):
            continue
        if len(responses) != len(weights) or len(responses) == 0:
            continue
        rows.append({"prompt": p, "responses": responses, "weights": weights})
    if not rows:
        raise ValueError(f"No valid GSPO rows in {paths.gspo_training_path}")
    return GSPODataset(rows)


def run_dpo_training(paths: PathConfig, dpo_cfg: DPOTrainingConfig, hf_token: str = None, ref_device: str = "auto"):
    _init_distributed()
    rank = _get_rank()
    world_size = _get_world_size()

    base_policy, processor = load_base_model_and_processor(paths.model_path, token=hf_token)
    policy_model = load_policy_with_lora(base_policy, paths.sft_adapter_path)

    base_ref, _ = load_base_model_and_processor(paths.model_path, token=hf_token)
    ref_model = load_policy_with_lora(base_ref, paths.sft_adapter_path)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    dpo_dataset = load_dpo_dataset(paths, processor)
    sampler = DistributedSampler(dpo_dataset, num_replicas=world_size, rank=rank, shuffle=True) if _is_dist() else None
    dataloader = DataLoader(
        dpo_dataset,
        batch_size=dpo_cfg.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda batch: list(zip(*batch)),
    )

    ref_device = (ref_device or "auto").lower()

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        policy_model = policy_model.to(local_rank)
        device_policy = torch.device(local_rank)
        device_ids = [local_rank]

        if ref_device == "cpu":
            ref_model = ref_model.to("cpu")
            device_ref = torch.device("cpu")
        else:
            try:
                ref_model = ref_model.to(local_rank)
                device_ref = torch.device(local_rank)
            except torch.cuda.OutOfMemoryError:
                if ref_device == "auto":
                    ref_model = ref_model.to("cpu")
                    device_ref = torch.device("cpu")
                    if rank == 0:
                        print("[dpo] Ref model fell back to CPU due to OOM; expect slower throughput.")
                else:
                    raise
    else:
        device_policy = get_primary_device(policy_model)
        ref_model = ref_model.to("cpu")
        device_ref = torch.device("cpu")
        device_ids = None

    policy_model = wrap_ddp(policy_model, device_ids=device_ids)
    amp_dtype = _precision_to_dtype(getattr(dpo_cfg, "precision", "none"))
    amp_dtype = amp_dtype if device_policy.type == "cuda" else None
    autocast_policy = _make_autocast(device_policy, amp_dtype)
    autocast_ref = _make_autocast(device_ref, amp_dtype)
    scaler = _make_grad_scaler(device_policy, amp_dtype)

    total_batches = len(dataloader)
    updates_per_epoch = max(1, math.ceil(total_batches / dpo_cfg.gradient_accumulation_steps))
    total_steps = updates_per_epoch * dpo_cfg.num_train_epochs
    warmup_steps = max(1, int(0.03 * total_steps))

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=dpo_cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    policy_model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(dpo_cfg.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader, 1):
            prompts, chosen_resps, rejected_resps = batch

            logp_chosen = compute_logps(policy_model, processor.tokenizer, list(prompts), list(chosen_resps), device_policy, max_length=dpo_cfg.max_length, no_grad=False, autocast_ctx=autocast_policy)
            logp_rejected = compute_logps(policy_model, processor.tokenizer, list(prompts), list(rejected_resps), device_policy, max_length=dpo_cfg.max_length, no_grad=False, autocast_ctx=autocast_policy)

            with torch.no_grad():
                logp_chosen_ref = compute_logps(ref_model, processor.tokenizer, list(prompts), list(chosen_resps), device_ref, max_length=dpo_cfg.max_length, no_grad=True, autocast_ctx=autocast_ref)
                logp_rejected_ref = compute_logps(ref_model, processor.tokenizer, list(prompts), list(rejected_resps), device_ref, max_length=dpo_cfg.max_length, no_grad=True, autocast_ctx=autocast_ref)

            log_ratio = (logp_chosen - logp_rejected) - (logp_chosen_ref - logp_rejected_ref)
            loss = -torch.nn.functional.logsigmoid(dpo_cfg.beta * log_ratio)
            loss = loss.mean() / dpo_cfg.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % dpo_cfg.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if rank == 0 and global_step % 10 == 0:
                    print(f"[dpo] epoch {epoch+1} step {global_step} loss={loss.item() * dpo_cfg.gradient_accumulation_steps:.4f}")
                    _log_metric("dpo", global_step, loss.item() * dpo_cfg.gradient_accumulation_steps, epoch=epoch + 1)

    if rank == 0:
        paths.dpo_model_path.parent.mkdir(parents=True, exist_ok=True)
        to_save = policy_model.module if hasattr(policy_model, "module") else policy_model
        to_save.save_pretrained(paths.dpo_model_path.as_posix())
        processor.save_pretrained(paths.dpo_model_path.as_posix())
        print(f"Saved DPO adapter to {paths.dpo_model_path}")


def run_grpo_training(paths: PathConfig, grpo_cfg: GRPOTrainingConfig, hf_token: str = None):
    _init_distributed()
    rank = _get_rank()
    world_size = _get_world_size()

    base_policy, processor = load_base_model_and_processor(paths.model_path, token=hf_token)
    policy_model = load_policy_with_lora(base_policy, paths.sft_adapter_path)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        policy_model = policy_model.to(local_rank)
        device_policy = torch.device(local_rank)
        device_ids = [local_rank]
    else:
        device_policy = get_primary_device(policy_model)
        device_ids = None

    policy_model = wrap_ddp(policy_model, device_ids=device_ids)

    grpo_dataset = load_grpo_dataset(paths)
    sampler = DistributedSampler(grpo_dataset, num_replicas=world_size, rank=rank, shuffle=True) if _is_dist() else None
    dataloader = DataLoader(
        grpo_dataset,
        batch_size=grpo_cfg.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda batch: list(zip(*batch)),
    )

    amp_dtype = _precision_to_dtype(getattr(grpo_cfg, "precision", "none"))
    amp_dtype = amp_dtype if device_policy.type == "cuda" else None
    autocast_policy = _make_autocast(device_policy, amp_dtype)
    scaler = _make_grad_scaler(device_policy, amp_dtype)

    total_batches = len(dataloader)
    updates_per_epoch = max(1, math.ceil(total_batches / grpo_cfg.gradient_accumulation_steps))
    total_steps = updates_per_epoch * grpo_cfg.num_train_epochs
    warmup_steps = max(1, int(0.03 * total_steps))

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=grpo_cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    policy_model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(grpo_cfg.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader, 1):
            prompts, responses, rewards = batch
            rewards = torch.tensor(rewards, device=device_policy, dtype=torch.float32)
            if grpo_cfg.reward_clip is not None:
                rewards = torch.clamp(rewards, -grpo_cfg.reward_clip, grpo_cfg.reward_clip)

            logp = compute_logps(
                policy_model,
                processor.tokenizer,
                list(prompts),
                list(responses),
                device_policy,
                max_length=1024,
                no_grad=False,
                autocast_ctx=autocast_policy,
            )

            # baseline: global mean reward (distributed-aware)
            reward_sum = rewards.sum()
            count = torch.tensor([rewards.numel()], device=device_policy, dtype=torch.float32)
            if _is_dist():
                dist.all_reduce(reward_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(count, op=dist.ReduceOp.SUM)
            baseline = reward_sum / count.clamp(min=1.0)
            advantages = rewards - baseline

            loss = -(advantages * logp).mean() / grpo_cfg.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grpo_cfg.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if rank == 0 and global_step % 10 == 0:
                    print(f"[grpo] epoch {epoch+1} step {global_step} loss={loss.item() * grpo_cfg.gradient_accumulation_steps:.4f}")
                    _log_metric("grpo", global_step, loss.item() * grpo_cfg.gradient_accumulation_steps, epoch=epoch + 1)

    if rank == 0:
        out_dir = paths.dpo_model_path.parent / "grpo_adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        to_save = policy_model.module if hasattr(policy_model, "module") else policy_model
        to_save.save_pretrained(out_dir.as_posix())
        processor.save_pretrained(out_dir.as_posix())
        print(f"Saved GRPO adapter to {out_dir}")


def run_gspo_training(paths: PathConfig, gspo_cfg: GSPOTrainingConfig, hf_token: str = None):
    _init_distributed()
    rank = _get_rank()
    world_size = _get_world_size()

    base_policy, processor = load_base_model_and_processor(paths.model_path, token=hf_token)
    policy_model = load_policy_with_lora(base_policy, paths.sft_adapter_path)

    base_ref, _ = load_base_model_and_processor(paths.model_path, token=hf_token)
    ref_model = load_policy_with_lora(base_ref, paths.sft_adapter_path)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        policy_model = policy_model.to(local_rank)
        ref_model = ref_model.to(local_rank)
        device_policy = torch.device(local_rank)
        device_ref = torch.device(local_rank)
        device_ids = [local_rank]
    else:
        device_policy = get_primary_device(policy_model)
        device_ref = get_primary_device(ref_model)
        device_ids = None

    policy_model = wrap_ddp(policy_model, device_ids=device_ids)

    gspo_dataset = load_gspo_dataset(paths)
    sampler = DistributedSampler(gspo_dataset, num_replicas=world_size, rank=rank, shuffle=True) if _is_dist() else None
    dataloader = DataLoader(
        gspo_dataset,
        batch_size=gspo_cfg.per_device_train_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        collate_fn=lambda batch: list(zip(*batch)),
    )

    amp_dtype = _precision_to_dtype(getattr(gspo_cfg, "precision", "none"))
    amp_dtype = amp_dtype if device_policy.type == "cuda" else None
    autocast_policy = _make_autocast(device_policy, amp_dtype)
    autocast_ref = _make_autocast(device_ref, amp_dtype)
    scaler = _make_grad_scaler(device_policy, amp_dtype)

    total_batches = len(dataloader)
    updates_per_epoch = max(1, math.ceil(total_batches / gspo_cfg.gradient_accumulation_steps))
    total_steps = updates_per_epoch * gspo_cfg.num_train_epochs
    warmup_steps = max(1, int(0.03 * total_steps))

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=gspo_cfg.learning_rate)
    scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    policy_model.train()
    optimizer.zero_grad()
    global_step = 0

    for epoch in range(gspo_cfg.num_train_epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader, 1):
            prompts, resp_lists, weight_lists = batch

            losses = []
            for prompt, responses, weights in zip(prompts, resp_lists, weight_lists):
                if not responses:
                    continue
                w = torch.tensor(weights, device=device_policy, dtype=torch.float32)
                if w.numel() == 0:
                    continue
                # normalize to sum=1 for stability
                w = w / w.sum().clamp(min=1e-6)

                logps = []
                with autocast_policy:
                    for resp in responses:
                        lp = compute_logps(
                            policy_model,
                            processor.tokenizer,
                            [prompt],
                            [resp],
                            device_policy,
                            max_length=1024,
                            no_grad=False,
                            autocast_ctx=None,  # already in autocast
                        )
                        logps.append(lp.squeeze(0))
                logps = torch.stack(logps)

                kl_term = 0.0
                if gspo_cfg.kl_coef > 0:
                    with torch.no_grad(), autocast_ref:
                        logps_ref = []
                        for resp in responses:
                            lp_ref = compute_logps(
                                ref_model,
                                processor.tokenizer,
                                [prompt],
                                [resp],
                                device_ref,
                                max_length=1024,
                                no_grad=True,
                                autocast_ctx=None,
                            )
                            logps_ref.append(lp_ref.squeeze(0))
                        logps_ref = torch.stack(logps_ref)
                    kl_term = gspo_cfg.kl_coef * torch.mean(logps - logps_ref)

                loss_i = -(w * logps).sum() + kl_term
                losses.append(loss_i)

            if not losses:
                continue

            loss = sum(losses) / len(losses)
            loss = loss / gspo_cfg.gradient_accumulation_steps

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % gspo_cfg.gradient_accumulation_steps == 0:
                if scaler.is_enabled():
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if rank == 0 and global_step % 10 == 0:
                    print(f"[gspo] epoch {epoch+1} step {global_step} loss={loss.item() * gspo_cfg.gradient_accumulation_steps:.4f}")
                    _log_metric("gspo", global_step, loss.item() * gspo_cfg.gradient_accumulation_steps, epoch=epoch + 1)

    if rank == 0:
        out_dir = paths.dpo_model_path.parent / "gspo_adapter"
        out_dir.mkdir(parents=True, exist_ok=True)
        to_save = policy_model.module if hasattr(policy_model, "module") else policy_model
        to_save.save_pretrained(out_dir.as_posix())
        processor.save_pretrained(out_dir.as_posix())
        print(f"Saved GSPO adapter to {out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(description="Self-rewarding -> DPO training pipeline")
    parser.add_argument(
        "--stage",
        choices=["sft", "prepare-dpo", "dpo", "grpo", "gspo", "all"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument("--sft-dataset", choices=SFT_DATASET_CHOICES, default="alpaca", help="SFT dataset: alpaca/sharegpt/openorca/self-instruct.")
    parser.add_argument("--dpo-dataset", choices=DPO_DATASET_CHOICES, default="self-reward", help="DPO dataset: self-reward (generated) or pku.")
    parser.add_argument("--grpo-dataset", choices=GRPO_DATASET_CHOICES, default="self-reward", help="GRPO dataset: self-reward (generated) or ultrafeedback.")
    parser.add_argument("--max-eval-samples", type=int, default=5, help="Cap evaluation samples for quick checks.")
    parser.add_argument("--num-candidates", type=int, default=DEFAULT_NUM_CANDIDATES, help="Candidates to generate per instruction during self-rewarding.")
    parser.add_argument("--n-votes", type=int, default=DEFAULT_N_VOTES, help="Judger samples per candidate.")
    parser.add_argument("--model-path", type=str, default=None, help="Local model path or HF repo id (e.g., Qwen/Qwen2.5-VL-7B).")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token for private/gated models.")
    parser.add_argument("--precision", type=str, default="none", choices=["none", "fp16", "bf16"], help="Mixed precision mode for training.")
    parser.add_argument("--gspo-temperature", type=float, default=1.0, help="Temperature for softmaxing scores in GSPO preprocessing.")
    parser.add_argument("--gspo-kl", type=float, default=0.0, help="KL coefficient to reference policy for GSPO.")
    parser.add_argument("--mode", type=str, choices=["normal", "demo"], default="normal", help="demo uses small subsets and 1 epoch; normal uses full data/config.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient_accumulation_steps for all training stages.")
    parser.add_argument("--ref-device", choices=["auto", "cuda", "cpu"], default="auto", help="Where to place the DPO reference model; auto tries GPU then falls back to CPU on OOM.")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = PathConfig(model_path=Path(args.model_path) if args.model_path else PathConfig.model_path)
    # 数据路径：demo 用 mini，小尺寸；normal 用全量 alpaca_en_train.json
    _configure_sft_paths(paths, args.sft_dataset, args.mode)
    _configure_dpo_path(paths, args.dpo_dataset)
    _configure_grpo_path(paths, args.grpo_dataset)
    lora_cfg = LoraHyperParams()
    sft_cfg = SFTConfig()
    dpo_cfg = DPOTrainingConfig()
    grpo_cfg = GRPOTrainingConfig()
    gspo_cfg = GSPOTrainingConfig()
    use_self_reward_dpo = args.dpo_dataset == "self-reward"
    use_self_reward_grpo = args.grpo_dataset == "self-reward"

    demo_mode = args.mode == "demo"
    if demo_mode:
        if args.num_candidates == DEFAULT_NUM_CANDIDATES:
            args.num_candidates = DEMO_NUM_CANDIDATES
        if args.n_votes == DEFAULT_N_VOTES:
            args.n_votes = DEMO_N_VOTES
        gen_max_new_tokens = DEMO_GEN_MAX_NEW_TOKENS
    else:
        gen_max_new_tokens = DEFAULT_GEN_MAX_NEW_TOKENS
    if args.grad_accum is not None and args.grad_accum > 0:
        sft_cfg.gradient_accumulation_steps = args.grad_accum
        dpo_cfg.gradient_accumulation_steps = args.grad_accum
        grpo_cfg.gradient_accumulation_steps = args.grad_accum
        gspo_cfg.gradient_accumulation_steps = args.grad_accum
    # demo presets: use small subsets and 1 epoch
    demo_train_limit = DEMO_TRAIN_LIMIT if demo_mode else None
    demo_test_limit = DEMO_TEST_LIMIT if demo_mode else None
    demo_train_fraction = DEMO_TRAIN_FRACTION if demo_mode else 1.0
    demo_test_fraction = DEMO_TEST_FRACTION if demo_mode else 1.0
    if demo_mode:
        sft_cfg.num_train_epochs = 1
        dpo_cfg.num_train_epochs = 1
        grpo_cfg.num_train_epochs = 1
        gspo_cfg.num_train_epochs = 1

    if args.stage in {"prepare-dpo", "all"} and use_self_reward_dpo:
        _init_distributed()

    policy_model = processor = None
    train_samples = test_messages = None

    if args.stage in {"sft", "all"}:
        policy_model, processor, train_samples, test_messages = run_sft_training(
            paths,
            lora_cfg,
            sft_cfg,
            max_eval_samples=args.max_eval_samples,
            hf_token=args.hf_token,
            precision=args.precision,
            max_train_samples=demo_train_limit,
            max_test_samples=demo_test_limit,
            max_train_fraction=demo_train_fraction,
            max_test_fraction=demo_test_fraction,
        )
    elif args.stage == "prepare-dpo" and use_self_reward_dpo:
        base_for_policy, processor = load_base_model_and_processor(paths.model_path, token=args.hf_token)
        policy_model = load_policy_with_lora(base_for_policy, paths.sft_adapter_path)
        train_samples, test_messages = load_sft_data(paths.sft_training_path, paths.test_path)
        if demo_mode:
            limit = max(1, int(len(train_samples) * demo_train_fraction))
            train_samples = train_samples[:limit] if demo_train_limit is None else train_samples[:demo_train_limit]
            limit_test = max(1, int(len(test_messages) * demo_test_fraction))
            test_messages = test_messages[:limit_test] if demo_test_limit is None else test_messages[:demo_test_limit]
        else:
            train_samples = train_samples[:50]

    if use_self_reward_dpo and args.stage in {"prepare-dpo", "all"}:
        if policy_model is None or processor is None:
            base_for_policy, processor = load_base_model_and_processor(paths.model_path, token=args.hf_token)
            policy_model = load_policy_with_lora(base_for_policy, paths.sft_adapter_path)
        if train_samples is None:
            train_samples, test_messages = load_sft_data(paths.sft_training_path, paths.test_path)
            if demo_mode:
                limit = max(1, int(len(train_samples) * demo_train_fraction))
                train_samples = train_samples[:limit] if demo_train_limit is None else train_samples[:demo_train_limit]
                limit_test = max(1, int(len(test_messages) * demo_test_fraction))
                test_messages = test_messages[:limit_test] if demo_test_limit is None else test_messages[:demo_test_limit]
            else:
                train_samples = train_samples[:50]
        elif not demo_mode:
            train_samples = train_samples[:50]
        generate_self_rewarding_data(
            policy_model=policy_model,
            processor=processor,
            train_samples=train_samples,
            paths=paths,
            num_candidates=args.num_candidates,
            n_votes=args.n_votes,
            generation_kwargs=dict(max_new_tokens=gen_max_new_tokens, temperature=0.7, top_p=0.9, do_sample=True),
        )
        if dist.is_available() and dist.is_initialized():
            dist.barrier()
        clean_self_rewarding_data(paths)
    elif args.stage == "prepare-dpo" and not use_self_reward_dpo and _get_rank() == 0:
        print(f"[prepare-dpo] Skipped self-reward generation because dpo_dataset={args.dpo_dataset}.")

    if args.stage in {"dpo", "all"}:
        dpo_cfg.precision = args.precision
        run_dpo_training(paths, dpo_cfg, hf_token=args.hf_token, ref_device=args.ref_device)

    if args.stage == "grpo":
        grpo_cfg.precision = args.precision
        if use_self_reward_grpo and _get_rank() == 0 and not paths.grpo_training_path.exists():
            build_grpo_dataset(paths)
        if use_self_reward_grpo and not paths.grpo_training_path.exists():
            # wait for rank0 creation
            dist.barrier() if dist.is_available() and dist.is_initialized() else None
        run_grpo_training(paths, grpo_cfg, hf_token=args.hf_token)

    if args.stage == "gspo":
        gspo_cfg.precision = args.precision
        gspo_cfg.temperature = args.gspo_temperature
        gspo_cfg.kl_coef = args.gspo_kl
        if _get_rank() == 0 and not paths.gspo_training_path.exists():
            build_gspo_dataset(paths, temperature=gspo_cfg.temperature)
        if not paths.gspo_training_path.exists():
            dist.barrier() if dist.is_available() and dist.is_initialized() else None
        run_gspo_training(paths, gspo_cfg, hf_token=args.hf_token)


if __name__ == "__main__":
    main()

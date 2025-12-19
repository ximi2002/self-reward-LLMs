"""
Simple CLI to chat with the SFT+LoRA checkpoint.

Usage:
  python -m scripts.chat_sft --model-path Qwen/Qwen2.5-VL-7B-Instruct --adapter-path checkpoints/sft_adaptor --precision bf16

Type 'exit' or 'quit' to leave the loop.
"""

import argparse
from pathlib import Path
from typing import Optional

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    Qwen2_5_VLForConditionalGeneration,
)
from peft import PeftModel
from utils import TokenizerWrapper, build_user_only_messages


def normalize_model_id(model_path: Path) -> str:
    return model_path.resolve().as_posix() if model_path.exists() else str(model_path).replace("\\", "/")


def load_model_and_processor(model_path: Path, adapter_path: Path, precision: str):
    model_id = normalize_model_id(model_path)
    prefer_vl = "vl" in model_id.lower()
    base = None
    processor = None

    if prefer_vl:
        try:
            base = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, device_map=None, torch_dtype="auto")
            processor = AutoProcessor.from_pretrained(model_id)
        except Exception:
            print(f"[chat] Falling back to text-only AutoModelForCausalLM for {model_id}")

    if base is None or processor is None:
        base = AutoModelForCausalLM.from_pretrained(model_id, device_map=None, torch_dtype="auto")
        tok = AutoTokenizer.from_pretrained(model_id)
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token_id = tok.eos_token_id
        processor = TokenizerWrapper(tok)

    model = PeftModel.from_pretrained(base, adapter_path.as_posix())

    if torch.cuda.is_available():
        device = torch.device("cuda")
        model = model.to(device)
    else:
        device = torch.device("cpu")

    amp_dtype: Optional[torch.dtype] = None
    if device.type == "cuda":
        if precision == "fp16":
            amp_dtype = torch.float16
        elif precision == "bf16":
            amp_dtype = torch.bfloat16

    return model, processor, device, amp_dtype


def main():
    parser = argparse.ArgumentParser(description="Interactive chat with SFT adapter.")
    parser.add_argument("--model-path", type=str, required=True, help="Base model path or HF repo id.")
    parser.add_argument("--adapter-path", type=str, default="checkpoints/sft_adaptor", help="LoRA adapter directory.")
    parser.add_argument("--precision", type=str, choices=["none", "fp16", "bf16"], default="none", help="AMP precision when CUDA is available.")
    parser.add_argument("--max-new-tokens", type=int, default=256, help="Generation length.")
    args = parser.parse_args()

    model_path = Path(args.model_path)
    adapter_path = Path(args.adapter_path)
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter not found: {adapter_path}")

    print("Loading model and adapter...")
    model, processor, device, amp_dtype = load_model_and_processor(model_path, adapter_path, args.precision)
    model.eval()
    print("Ready. Type 'exit' or 'quit' to stop.")

    while True:
        user_in = input("User: ").strip()
        if user_in.lower() in {"exit", "quit"}:
            break
        if not user_in:
            continue

        tokenizer = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        apply_template = processor.apply_chat_template if hasattr(processor, "apply_chat_template") else tokenizer.apply_chat_template
        messages = build_user_only_messages(processor, user_in)
        prompt = apply_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=prompt, return_tensors="pt")
        inputs = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in inputs.items()}

        with torch.no_grad():
            ctx = torch.amp.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype is not None else torch.no_grad()
            with ctx:
                outputs = model.generate(
                    **inputs,
                    tokenizer=tokenizer,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                )

        dec_start = inputs["input_ids"].shape[1]
        reply = tokenizer.decode(outputs[0][dec_start:], skip_special_tokens=True).strip()
        print(f"Assistant: {reply}\n")


if __name__ == "__main__":
    main()

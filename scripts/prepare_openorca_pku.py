"""
Download and prepare OpenOrca (SFT) and PKU-SafeRLHF (DPO) subsets.

Outputs:
- data/openorca_sft.json       (Alpaca-style list: instruction/input/output)
- data/pku_safedpo.jsonl       (JSONL with prompt/ chosen / rejected for DPO)

Notes:
- Uses huggingface `datasets`. Set HF token via env if needed.
- Defaults to small subsets for quick start; adjust --sft-num / --dpo-num.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict
from datasets import load_dataset
import random


def prepare_openorca(num_samples: int, out_path: Path, seed: int = 42):
    ds = load_dataset("Open-Orca/OpenOrca", split="train")
    if num_samples and num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    records: List[Dict] = []
    for ex in ds:
        instr = ex.get("question") or ""
        resp = ex.get("response") or ""
        records.append(
            {
                "instruction": instr.strip(),
                "input": "",
                "output": resp.strip(),
            }
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[openorca] wrote {len(records)} samples -> {out_path}")


def prepare_pku_safedpo(num_samples: int, out_path: Path, seed: int = 42):
    ds = load_dataset("PKU-Alignment/PKU-SafeRLHF", split="train")
    if num_samples and num_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(num_samples))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        n = 0
        for ex in ds:
            prompt = ex.get("prompt")
            chosen = ex.get("chosen")
            rejected = ex.get("rejected")
            # Fallback for label / safer_response_id with response_0/1
            r0 = ex.get("response_0")
            r1 = ex.get("response_1")
            lbl = ex.get("label")
            safer = ex.get("safer_response_id")
            better = ex.get("better_response_id")
            if (chosen is None or rejected is None) and isinstance(r0, str) and isinstance(r1, str):
                if better in (0, 1):
                    chosen, rejected = (r0, r1) if better == 0 else (r1, r0)
                elif lbl in (0, 1):
                    chosen, rejected = (r0, r1) if lbl == 0 else (r1, r0)
                elif safer in ("response_0", "response_1"):
                    chosen, rejected = (r0, r1) if safer == "response_0" else (r1, r0)
            if not prompt or not chosen or not rejected:
                continue
            f.write(json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected}, ensure_ascii=False) + "\n")
            n += 1
    print(f"[pku] wrote {n} pairs -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare OpenOrca (SFT) and PKU-SafeRLHF (DPO) subsets.")
    parser.add_argument("--sft-num", type=int, default=5000, help="Number of OpenOrca samples to keep (0 or None for all).")
    parser.add_argument("--dpo-num", type=int, default=20000, help="Number of PKU-SafeRLHF pairs to keep (0 or None for all).")
    parser.add_argument("--sft-out", type=Path, default=Path("data/openorca_sft.json"), help="Output path for OpenOrca SFT JSON.")
    parser.add_argument("--dpo-out", type=Path, default=Path("data/pku_safedpo.jsonl"), help="Output path for PKU DPO JSONL.")
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed before selecting subsets.")
    args = parser.parse_args()

    prepare_openorca(args.sft_num, args.sft_out, seed=args.seed)
    prepare_pku_safedpo(args.dpo_num, args.dpo_out, seed=args.seed)


if __name__ == "__main__":
    main()

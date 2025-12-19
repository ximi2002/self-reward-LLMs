"""
Convert ShareGPT-style conversations to SFT (ShareGPT JSON) and optional GRPO pseudo-reward data.

Inputs: a JSON or JSONL ShareGPT file; each entry should have a list of messages with roles ("user"/"assistant").
Outputs:
- data/sharegpt_sft.json        (list of {messages: [...]}) for SFT
- data/sharegpt_grpo.jsonl      (optional, flattened (prompt, response, reward) with pseudo reward=1.0)

This is a minimal converter; adjust filtering/roles as needed.
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any


def iter_records(path: Path):
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            for rec in data:
                yield rec
        else:
            yield data


def to_messages(conv: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    msgs = []
    for m in conv:
        role = m.get("from") or m.get("role")
        content = m.get("value") or m.get("content")
        if not role or content is None:
            continue
        msgs.append({"role": role, "content": content})
    return msgs


def main():
    parser = argparse.ArgumentParser(description="Convert ShareGPT to SFT/GRPO-ready files.")
    parser.add_argument("--input", type=Path, required=True, help="Path to ShareGPT JSON/JSONL.")
    parser.add_argument("--sft-out", type=Path, default=Path("data/sharegpt_sft.json"), help="Output path for SFT JSON.")
    parser.add_argument("--grpo-out", type=Path, default=Path("data/sharegpt_grpo.jsonl"), help="Optional GRPO JSONL (prompt/response/reward).")
    parser.add_argument("--make-grpo", action="store_true", help="If set, write pseudo-reward GRPO data with reward=1.0.")
    args = parser.parse_args()

    sft_records = []
    grpo_lines = []

    for rec in iter_records(args.input):
        conv = rec.get("conversations") or rec.get("messages") or rec.get("data")
        if not conv or not isinstance(conv, list):
            continue
        msgs = to_messages(conv)
        if len(msgs) < 2:
            continue
        sft_records.append({"messages": msgs})
        if args.make_grpo:
            # Use the first user message as prompt, last assistant as response, reward=1.0
            prompt = None
            response = None
            for m in msgs:
                if m["role"] == "user" and prompt is None:
                    prompt = m["content"]
                if m["role"] == "assistant":
                    response = m["content"]
            if prompt and response:
                grpo_lines.append(json.dumps({"prompt": prompt, "response": response, "reward": 1.0}, ensure_ascii=False))

    args.sft_out.parent.mkdir(parents=True, exist_ok=True)
    args.sft_out.write_text(json.dumps(sft_records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[sharegpt] SFT records: {len(sft_records)} -> {args.sft_out}")

    if args.make_grpo:
        Path(args.grpo_out).parent.mkdir(parents=True, exist_ok=True)
        with args.grpo_out.open("w", encoding="utf-8") as f:
            for line in grpo_lines:
                f.write(line + "\n")
        print(f"[sharegpt] GRPO pseudo-reward lines: {len(grpo_lines)} -> {args.grpo_out}")


if __name__ == "__main__":
    main()

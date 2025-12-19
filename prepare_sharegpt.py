import json
from pathlib import Path

input_path = Path("data/sharegpt/ShareGPT_V3_unfiltered_cleaned_split.json")
output_path = Path("data/sharegpt_sft.json")

def normalize_role(role):
    role = (role or "").lower()
    if role in ("human", "user", "instruction", "prompt"):
        return "user"
    if role in ("gpt", "assistant", "bot", "model"):
        return "assistant"
    if role == "system":
        return "system"
    return None

data = json.loads(input_path.read_text(encoding="utf-8"))
out = []
skipped = 0
for rec in data:
    conv = rec.get("conversations") or rec.get("messages") or rec.get("data")
    if not isinstance(conv, list):
        skipped += 1
        continue
    msgs = []
    for m in conv:
        if not isinstance(m, dict):
            continue
        role = normalize_role(m.get("from") or m.get("role"))
        content = m.get("value") if "value" in m else m.get("content")
        if role is None or content is None or content == "":
            continue
        msgs.append({"role": role, "content": content})
    if len(msgs) < 2:
        skipped += 1
        continue
    out.append({"messages": msgs})

output_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"kept={len(out)} skipped={skipped}")
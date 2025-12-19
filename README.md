# Self-Rewarding → DPO/GRPO/GSPO Pipeline

Refactored notebook into reusable scripts for instruction tuning with self-rewarding data. Supports single/multi-GPU (DDP) and AMP. New additions:
- Self-reward sampling log `data/self_reward_samples.jsonl` for quick eyeballing.
- CLI chat script `scripts/chat_sft.py` to load the SFT LoRA and chat from terminal.
- Text-only support: automatically falls back to `AutoModelForCausalLM` + `AutoTokenizer` for `Qwen2.5-Instruct` (non-VL) models.

## Layout
- `src/config.py`: path + hyperparameter dataclasses.
- `src/pipeline.py`: main entry (`--stage` for SFT, prepare-dpo, dpo/grpo/gspo, all).
- `utils.py`: datasets, generation/judging helpers, DPO cleaning, evaluation.
- `scripts/chat_sft.py`: load base model + SFT LoRA and chat interactively.
- `scripts/report.py`: read logs/data and plot distributions/curves.
- `data/`: expected data location.
- `checkpoints/`: saved adapters.
- `reports/`: loss logs and plots.

## Install
```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
# install a CUDA-matching torch wheel if using GPU
```

## Data
- SFT: Alpaca-style JSON with `instruction`/`input`/`output`.
- Self-judged raw: `data/raw_responses.jsonl` (from `--stage prepare-dpo`).
- Cleaned DPO: `data/dpo_training.jsonl`; falls back to `data/dpo_training_manual.jsonl` if the main file is absent.
- GRPO: `data/grpo_training.jsonl` (flattened `prompt/response/reward` from raw).
- GSPO: `data/gspo_training.jsonl` (softmax(scores) → `prompt/responses/weights`).
- Example data fetch:
  ```bash
  python scripts/prepare_openorca_pku.py --sft-num 5000 --dpo-num 20000
  ```
  Outputs `data/openorca_sft.json` (SFT) and `data/pku_safedpo.jsonl` (DPO).
- Dataset switches:
  - `--sft-dataset {alpaca, sharegpt, openorca, self-instruct}`. Defaults to alpaca (demo uses mini); expects files like `data/sharegpt_sft.json`, `data/openorca_sft.json`, `data/self_instruct_sft.json`.
  - `--dpo-dataset {self-reward, pku}`. `pku` reads `data/pku_safedpo.jsonl`; `self-reward` uses generated `data/raw_responses.jsonl` → `data/dpo_training.jsonl`.
  - `--grpo-dataset {self-reward, ultrafeedback}`. `ultrafeedback` expects `data/ultrafeedback_grpo.jsonl` with `prompt/response/reward`.
- Self-reward scoring: `REWARD_PROMPT_TEMPLATE` + regex `REWARD_SCORE_REGEX` (matches `Score: <0-5>`) parse scores into raw, then cleaned to DPO/GRPO/GSPO.

## Quickstart (single GPU)
1) SFT (LoRA)
```bash
python -m src.pipeline --stage sft --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
Text-only variant:
```bash
python -m src.pipeline --stage sft --model-path Qwen/Qwen2.5-7B-Instruct --precision bf16 --mode demo
```
Outputs `checkpoints/sft_adaptor`, log `reports/sft_loss.jsonl`.

2) Self-reward generate + clean
```bash
python -m src.pipeline --stage prepare-dpo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
Outputs `data/raw_responses.jsonl`, `data/dpo_training.jsonl`, and sampled log `data/self_reward_samples.jsonl` (up to 20; tweak in `generate_self_rewarding_data` if needed).

3) DPO training
```bash
python -m src.pipeline --stage dpo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
Outputs `checkpoints/dpo_no_self_instruction`, log `reports/dpo_loss.jsonl`.

4) Or GRPO
```bash
python -m src.pipeline --stage grpo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
Text-only: add `--model-path Qwen/Qwen2.5-7B-Instruct`.
Auto-creates `data/grpo_training.jsonl` if missing. Outputs `checkpoints/grpo_adapter`.

5) Or GSPO
```bash
python -m src.pipeline --stage gspo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --gspo-temperature 1.0 --gspo-kl 0.0 --mode demo
```
Text-only: add `--model-path Qwen/Qwen2.5-7B-Instruct`.
Auto-creates `data/gspo_training.jsonl` if missing. Outputs `checkpoints/gspo_adapter`.

6) Full chain (SFT → generate → DPO)
```bash
python -m src.pipeline --stage all --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
Text-only: add `--model-path Qwen/Qwen2.5-7B-Instruct`.

## Interact & inspect
- Chat with SFT model:
  ```bash
  python -m scripts.chat_sft --model-path Qwen/Qwen2.5-VL-7B-Instruct --adapter-path checkpoints/sft_adaptor --precision bf16
  ```
  Text-only: add `--model-path Qwen/Qwen2.5-7B-Instruct`.
  Type your question; `exit`/`quit` to leave.
- Self-reward samples: `data/self_reward_samples.jsonl` contains instructions, candidates, judge outputs, and chosen/rejected for quick sanity checks.

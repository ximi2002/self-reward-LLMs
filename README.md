# Self-Rewarding → DPO/GRPO/GSPO Pipeline

重构版指令微调与自奖励偏好训练流水线，支持单卡/多卡、AMP。新增：
- 自奖励阶段采样日志 `data/self_reward_samples.jsonl`（便于快速肉眼检查）。
- 命令行对话脚本 `scripts/chat_sft.py` 可加载 SFT LoRA 交互。
- 新：支持文本版 Qwen2.5-Instruct，非 VL 模型自动回落到 `AutoModelForCausalLM` + `AutoTokenizer`。

## 目录结构
- `src/config.py`：路径与超参 dataclass。
- `src/pipeline.py`：主入口（`--stage` 控制 SFT、prepare-dpo、dpo/grpo/gspo、all）。
- `utils.py`：数据集封装、生成与判分、DPO 清洗、评测。
- `scripts/chat_sft.py`：加载基础模型 + SFT LoRA，命令行对话。
- `scripts/report.py`：读取日志与数据，绘制分布/曲线。
- `data/`：默认数据路径（自备或脚本生成）。
- `checkpoints/`：SFT/DPO/GRPO/GSPO 适配器。
- `reports/`：loss 日志与可视化输出。

## 环境安装
```bash
python -m venv .venv
.\.venv\Scripts\activate      # Windows
# source .venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
# 如需 GPU，请安装匹配 CUDA 的 torch 轮子
```

## 数据准备
- SFT：Alpaca 风格 JSON，含 `instruction`/`input`/`output`。
- 自奖励原始：`data/raw_responses.jsonl`（`--stage prepare-dpo` 生成）。
- 清洗后 DPO：`data/dpo_training.jsonl`；若不存在则尝试 `data/dpo_training_manual.jsonl`。
- GRPO：`data/grpo_training.jsonl`（从 raw 扁平化出 `prompt/response/reward`）。
- GSPO：`data/gspo_training.jsonl`（从 raw softmax(score) 得 `prompt/responses/weights`）。
- 示例数据获取：
  ```bash
  python scripts/prepare_openorca_pku.py --sft-num 5000 --dpo-num 20000
  ```
  产出 `data/openorca_sft.json`（SFT）与 `data/pku_safedpo.jsonl`（DPO）。
- 自奖励打分：`REWARD_PROMPT_TEMPLATE` + 正则 `REWARD_SCORE_REGEX`（匹配 `Score: <0-5>`）解析判分，写入 raw，再清洗成 DPO/GRPO/GSPO。

## 快速上手（单卡示例）
1) SFT（LoRA）
```bash
python -m src.pipeline --stage sft --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
文本版（纯文本 Qwen2.5-Instruct）：
```bash
python -m src.pipeline --stage sft --model-path Qwen/Qwen2.5-7B-Instruct --precision bf16 --mode demo
```
输出 `checkpoints/sft_adaptor`，日志 `reports/sft_loss.jsonl`。

2) 自奖励生成 + 清洗
```bash
python -m src.pipeline --stage prepare-dpo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
输出 `data/raw_responses.jsonl`、`data/dpo_training.jsonl`，并抽样日志 `data/self_reward_samples.jsonl`（最多 20 条，可在 `generate_self_rewarding_data` 调整）。
- 多卡并行准备：`torchrun --nproc_per_node=4 python -m src.pipeline --stage prepare-dpo --model-path ... --precision bf16 --mode demo`，各 rank 分片生成，rank0 汇总写文件。

3) DPO 训练
```bash
python -m src.pipeline --stage dpo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
输出 `checkpoints/dpo_no_self_instruction`，日志 `reports/dpo_loss.jsonl`。
- 若参考模型与策略模型同卡 OOM，可加 `--ref-device cpu`；默认 `auto`，GPU OOM 时会自动回落到 CPU（速度略慢）。

4) 或 GRPO
```bash
python -m src.pipeline --stage grpo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
文本版：`--model-path Qwen/Qwen2.5-7B-Instruct`。
如缺 `data/grpo_training.jsonl` 会自动生成。输出 `checkpoints/grpo_adapter`。

5) 或 GSPO
```bash
python -m src.pipeline --stage gspo --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --gspo-temperature 1.0 --gspo-kl 0.0 --mode demo
```
文本版：`--model-path Qwen/Qwen2.5-7B-Instruct`。
如缺 `data/gspo_training.jsonl` 会自动生成。输出 `checkpoints/gspo_adapter`。

6) 一键全流程（SFT→生成→DPO）
```bash
python -m src.pipeline --stage all --model-path Qwen/Qwen2.5-VL-7B-Instruct --precision bf16 --mode demo
```
文本版：`--model-path Qwen/Qwen2.5-7B-Instruct`。

## 交互与检查
- 命令行对话 SFT 模型：
  ```bash
  python -m scripts.chat_sft --model-path Qwen/Qwen2.5-VL-7B-Instruct --adapter-path checkpoints/sft_adaptor --precision bf16
  ```
  文本版：`--model-path Qwen/Qwen2.5-7B-Instruct`。
  输入问题，`exit`/`quit` 退出。
- 自奖励抽样查看：`data/self_reward_samples.jsonl` 含 instruction、候选、判分输出、选/拒答案，便于快速 sanity check。

## Dataset switches
- --sft-dataset {alpaca, sharegpt, openorca, self-instruct} (files: data/alpaca_en_train.json, data/sharegpt_sft.json, data/openorca_sft.json, data/self_instruct_sft.json).
- --dpo-dataset {self-reward, pku} (self-reward uses generated data/dpo_training.jsonl; pku reads data/pku_safedpo.jsonl).
- --grpo-dataset {self-reward, ultrafeedback} (ultrafeedback expects data/ultrafeedback_grpo.jsonl with prompt/response/reward).

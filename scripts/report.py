"""
Generate quick plots and text summaries from self-rewarding outputs.

Reads:
- raw_responses.jsonl  -> score hist, candidates per prompt
- dpo_training.jsonl   -> pair count
- grpo_training.jsonl  -> reward hist
- gspo_training.jsonl  -> weight hist
- reports/*_loss.jsonl -> training curves (sft/dpo/grpo/gspo)

Outputs plots to the specified output directory and prints tables to stdout.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def summarize_raw(raw_path: Path) -> Tuple[Dict, List[float], List[int]]:
    prompts = 0
    candidates = 0
    valid_scores: List[float] = []
    per_prompt_counts: List[int] = []

    for rec in read_jsonl(raw_path):
        prompt = rec.get("instruction") or rec.get("prompt") or rec.get("query")
        cands = rec.get("candidates", [])
        prompts += 1
        per_prompt_counts.append(len(cands))
        for c in cands:
            score = c.get("score")
            if score is None:
                continue
            try:
                s = float(score)
            except Exception:
                continue
            if math.isfinite(s):
                valid_scores.append(s)
                candidates += 1

    stats = {
        "prompts": prompts,
        "candidates": candidates,
        "scores_mean": float(sum(valid_scores) / len(valid_scores)) if valid_scores else float("nan"),
        "scores_min": float(min(valid_scores)) if valid_scores else float("nan"),
        "scores_max": float(max(valid_scores)) if valid_scores else float("nan"),
        "avg_cands_per_prompt": float(sum(per_prompt_counts) / len(per_prompt_counts)) if per_prompt_counts else 0.0,
    }
    return stats, valid_scores, per_prompt_counts


def summarize_jsonl_lines(path: Path, value_key: str = None) -> Tuple[int, List[float]]:
    count = 0
    values: List[float] = []
    if not path.exists():
        return count, values
    for rec in read_jsonl(path):
        count += 1
        if value_key and value_key in rec:
            try:
                v = float(rec[value_key])
                if math.isfinite(v):
                    values.append(v)
            except Exception:
                pass
    return count, values


def summarize_gspo(path: Path) -> Tuple[int, List[float]]:
    count = 0
    weights: List[float] = []
    if not path.exists():
        return count, weights
    for rec in read_jsonl(path):
        count += 1
        ws = rec.get("weights", [])
        for w in ws:
            try:
                v = float(w)
                if math.isfinite(v):
                    weights.append(v)
            except Exception:
                pass
    return count, weights


def plot_hist(data: List[float], title: str, xlabel: str, out_path: Path, bins: int = 30):
    if not data:
        return
    plt.figure()
    plt.hist(data, bins=bins, edgecolor="black")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def summarize_losses(path: Path) -> Tuple[Dict, List[Tuple[int, float]]]:
    points: List[Tuple[int, float]] = []
    if not path.exists():
        return {}, points
    for rec in read_jsonl(path):
        step = rec.get("step")
        loss = rec.get("loss")
        if step is None or loss is None:
            continue
        try:
            s = int(step)
            l = float(loss)
        except Exception:
            continue
        points.append((s, l))
    points.sort(key=lambda x: x[0])
    if not points:
        return {}, points
    losses = [p[1] for p in points]
    stats = {
        "steps": len(points),
        "last_loss": losses[-1],
        "min_loss": min(losses),
        "max_loss": max(losses),
        "mean_loss": float(sum(losses) / len(losses)),
    }
    return stats, points


def plot_curve(points: List[Tuple[int, float]], title: str, xlabel: str, ylabel: str, out_path: Path):
    if not points:
        return
    xs, ys = zip(*points)
    plt.figure()
    plt.plot(xs, ys, marker=".", linewidth=1)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Report and plots for self-rewarding pipelines.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Directory containing jsonl outputs.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports"), help="Directory to save plots.")
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"), help="Directory containing *_loss.jsonl logs.")
    parser.add_argument("--rounds-dir", type=Path, default=None, help="If set, summarize each round_<k>/data under this path.")
    args = parser.parse_args()

    rounds_root = args.rounds_dir
    if rounds_root is None:
        default_rounds = Path("checkpoints")
        if any(default_rounds.glob("round_*/data")):
            rounds_root = default_rounds

    def process_dataset(data_dir: Path, out_dir: Path, label: str = ""):
        suffix = f" ({label})" if label else ""
        raw_path = data_dir / "raw_responses.jsonl"
        dpo_path = data_dir / "dpo_training.jsonl"
        grpo_path = data_dir / "grpo_training.jsonl"
        gspo_path = data_dir / "gspo_training.jsonl"

        print(f"== Summary{suffix} ==")

        if raw_path.exists():
            stats, scores, per_prompt = summarize_raw(raw_path)
            print(f"raw_responses: prompts={stats['prompts']} candidates={stats['candidates']} "
                  f"mean_score={stats['scores_mean']:.3f} min={stats['scores_min']:.3f} max={stats['scores_max']:.3f} "
                  f"avg_cands_per_prompt={stats['avg_cands_per_prompt']:.2f}")
            plot_hist(scores, f"Reward score distribution{suffix}", "Score", out_dir / ("scores_hist" + (f"_{label}" if label else "") + ".png"))
            plot_hist(per_prompt, f"Candidates per prompt{suffix}", "#candidates", out_dir / ("candidates_per_prompt" + (f"_{label}" if label else "") + ".png"), bins=20)
        else:
            print(f"raw_responses missing at {raw_path}")

        dpo_count, _ = summarize_jsonl_lines(dpo_path)
        print(f"dpo_training: pairs={dpo_count} (path={dpo_path})")

        grpo_count, rewards = summarize_jsonl_lines(grpo_path, value_key="reward")
        print(f"grpo_training: rows={grpo_count} (path={grpo_path})")
        plot_hist(rewards, f"GRPO reward distribution{suffix}", "Reward", out_dir / ("grpo_rewards" + (f"_{label}" if label else "") + ".png"))

        gspo_count, weights = summarize_gspo(gspo_path)
        print(f"gspo_training: prompts={gspo_count} (path={gspo_path})")
        plot_hist(weights, f"GSPO weight distribution{suffix}", "Weight", out_dir / ("gspo_weights" + (f"_{label}" if label else "") + ".png"))

    # main dataset
    process_dataset(args.data_dir, args.out_dir, label="")

    # optional round archives
    if rounds_root:
        for rd in sorted(rounds_root.glob("round_*/data")):
            label = rd.parent.name  # e.g., round_1
            process_dataset(rd, args.out_dir, label=label)

    # Training curves (default)
    for stage in ["sft", "dpo", "grpo", "gspo"]:
        loss_file = args.reports_dir / f"{stage}_loss.jsonl"
        stats, points = summarize_losses(loss_file)
        if stats:
            print(f"{stage} loss: steps={stats['steps']} last={stats['last_loss']:.4f} "
                  f"min={stats['min_loss']:.4f} mean={stats['mean_loss']:.4f} (log={loss_file})")
            plot_curve(points, f"{stage.upper()} loss", "step", "loss", args.out_dir / f"{stage}_loss.png")
        else:
            print(f"{stage} loss: no log found at {loss_file}")

    print(f"Plots (if data existed) saved to: {args.out_dir}")


if __name__ == "__main__":
    main()

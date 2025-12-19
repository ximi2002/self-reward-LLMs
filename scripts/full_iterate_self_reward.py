"""
One-click script to run SFT once, then multiple self-rewarding rounds (DPO/GRPO/GSPO),
archiving results per round.

Flow:
0) Run SFT -> checkpoints/sft_adaptor
1..N) For each round:
   - Use current adapter as policy (copied to checkpoints/sft_adaptor)
   - prepare-dpo (self-generate + self-judge)
   - train chosen stage (dpo/grpo/gspo)
   - archive adapter + data to checkpoints/round_<k>/
   - next round uses newly trained adapter

Warning: Each round overwrites default outputs (checkpoints/sft_adaptor, data/*.jsonl, stage adapter),
but archives per round to avoid loss.
"""

import argparse
import shutil
import subprocess
from pathlib import Path


STAGE_TO_ADAPTER = {
    "dpo": Path("checkpoints/dpo_no_self_instruction"),
    "grpo": Path("checkpoints/grpo_adapter"),
    "gspo": Path("checkpoints/gspo_adapter"),
}

DATA_FILES = [
    Path("data/raw_responses.jsonl"),
    Path("data/dpo_training.jsonl"),
    Path("data/grpo_training.jsonl"),
    Path("data/gspo_training.jsonl"),
]


def copy_tree(src: Path, dst: Path):
    # If source and destination are the same, skip copy to avoid deleting src
    if src.resolve() == dst.resolve():
        return
    if dst.exists():
        if dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    shutil.copytree(src, dst)


def run_cmd(cmd):
    print(f"[run] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run SFT once, then multi-round self-reward + DPO/GRPO/GSPO.")
    parser.add_argument("--rounds", type=int, default=2, help="Number of iterative rounds after SFT.")
    parser.add_argument("--stage", type=str, choices=["dpo", "grpo", "gspo"], default="dpo", help="Training stage per round.")
    parser.add_argument("--model-path", type=str, required=True, help="Base model path or HF repo id.")
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"], help="Mixed precision.")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "demo"], help="Use demo mode for quick tests.")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token for private/gated models.")
    parser.add_argument("--grad-accum", type=int, default=None, help="Override gradient_accumulation_steps for all stages.")
    args = parser.parse_args()

    # 0) SFT
    sft_cmd = [
        "python",
        "-m",
        "src.pipeline",
        "--stage",
        "sft",
        "--model-path",
        args.model_path,
        "--precision",
        args.precision,
        "--mode",
        args.mode,
    ]
    if args.hf_token:
        sft_cmd += ["--hf-token", args.hf_token]
    if args.grad_accum:
        sft_cmd += ["--grad-accum", str(args.grad_accum)]
    run_cmd(sft_cmd)

    current_adapter = Path("checkpoints/sft_adaptor")
    out_adapter_name = STAGE_TO_ADAPTER[args.stage]
    rounds = args.rounds

    for r in range(rounds):
        print(f"\n=== Round {r+1}/{rounds} ({args.stage}) ===")
        if not current_adapter.exists():
            raise FileNotFoundError(f"Adapter not found: {current_adapter}")
        copy_tree(current_adapter, Path("checkpoints/sft_adaptor"))

        # prepare-dpo
        cmd_prepare = [
            "python",
            "-m",
            "src.pipeline",
            "--stage",
            "prepare-dpo",
            "--model-path",
            args.model_path,
            "--precision",
            args.precision,
            "--mode",
            args.mode,
        ]
        if args.hf_token:
            cmd_prepare += ["--hf-token", args.hf_token]
        if args.grad_accum:
            cmd_prepare += ["--grad-accum", str(args.grad_accum)]
        run_cmd(cmd_prepare)

        # train stage
        cmd_stage = [
            "python",
            "-m",
            "src.pipeline",
            "--stage",
            args.stage,
            "--model-path",
            args.model_path,
            "--precision",
            args.precision,
            "--mode",
            args.mode,
        ]
        if args.hf_token:
            cmd_stage += ["--hf-token", args.hf_token]
        if args.grad_accum:
            cmd_stage += ["--grad-accum", str(args.grad_accum)]
        run_cmd(cmd_stage)

        # archive
        round_dir = Path("checkpoints") / f"round_{r+1}"
        model_out_dir = round_dir / "adapter"
        data_out_dir = round_dir / "data"
        round_dir.mkdir(parents=True, exist_ok=True)
        if out_adapter_name.exists():
            copy_tree(out_adapter_name, model_out_dir)
            print(f"[archive] adapter -> {model_out_dir}")
        else:
            print(f"[warn] adapter output not found: {out_adapter_name}")

        data_out_dir.mkdir(parents=True, exist_ok=True)
        for f in DATA_FILES:
            if f.exists():
                shutil.copy2(f, data_out_dir / f.name)
                print(f"[archive] data -> {data_out_dir / f.name}")

        current_adapter = model_out_dir


if __name__ == "__main__":
    main()

"""
Run self-rewarding + DPO/GRPO/GSPO for N rounds, archiving each round's data/checkpoints.

Flow per round:
1) Copy the current adapter into checkpoints/sft_adaptor (so pipeline uses it).
2) Run prepare-dpo (self-generate + self-judge) to produce raw/dpo data.
3) Run the chosen stage (dpo/grpo/gspo) to train a new adapter.
4) Archive outputs under checkpoints/round_<k>/ and use the new adapter as the next round's current adapter.

Warning: This will overwrite the default paths (checkpoints/sft_adaptor, data/raw_responses.jsonl, etc.) each round.
Archived copies are kept per round to avoid loss.
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
    # If source and destination are the same, do nothing
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
    parser = argparse.ArgumentParser(description="Iterate self-reward + DPO/GRPO/GSPO for N rounds.")
    parser.add_argument("--rounds", type=int, default=2, help="Number of iterative rounds.")
    parser.add_argument("--stage", type=str, choices=["dpo", "grpo", "gspo"], default="dpo", help="Training stage each round.")
    parser.add_argument("--model-path", type=str, required=True, help="Base model path or HF repo id.")
    parser.add_argument("--precision", type=str, default="bf16", choices=["none", "fp16", "bf16"], help="Mixed precision.")
    parser.add_argument("--mode", type=str, default="normal", choices=["normal", "demo"], help="Use demo mode for quick tests.")
    parser.add_argument("--hf-token", type=str, default=None, help="HF token for private/gated models.")
    parser.add_argument("--start-adapter", type=Path, default=Path("checkpoints/sft_adaptor"), help="Initial adapter to start from.")
    args = parser.parse_args()

    current_adapter = args.start_adapter
    rounds = args.rounds
    stage = args.stage
    out_adapter_name = STAGE_TO_ADAPTER[stage]

    for r in range(rounds):
        print(f"\n=== Round {r+1}/{rounds} ({stage}) ===")
        # 1) ensure pipeline uses current adapter
        if not current_adapter.exists():
            raise FileNotFoundError(f"Adapter not found: {current_adapter}")
        copy_tree(current_adapter, Path("checkpoints/sft_adaptor"))

        # 2) prepare-dpo
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
        run_cmd(cmd_prepare)

        # 3) train stage
        cmd_stage = [
            "python",
            "-m",
            "src.pipeline",
            "--stage",
            stage,
            "--model-path",
            args.model_path,
            "--precision",
            args.precision,
            "--mode",
            args.mode,
        ]
        if args.hf_token:
            cmd_stage += ["--hf-token", args.hf_token]
        run_cmd(cmd_stage)

        # 4) archive outputs
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

        # next round uses newly trained adapter
        current_adapter = model_out_dir


if __name__ == "__main__":
    main()

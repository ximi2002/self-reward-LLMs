"""
One-click checkpoint cleanup script.

By default removes:
- checkpoints/sft_adaptor
- checkpoints/dpo_no_self_instruction
- checkpoints/grpo_adapter
- checkpoints/gspo_adapter

Use --dry-run to preview without deleting.
"""

import argparse
from pathlib import Path
import shutil


DEFAULT_DIRS = [
    Path("checkpoints/sft_adaptor"),
    Path("checkpoints/dpo_no_self_instruction"),
    Path("checkpoints/grpo_adapter"),
    Path("checkpoints/gspo_adapter"),
]


def remove_path(p: Path, dry_run: bool):
    if not p.exists():
        return f"{p} (missing)"
    if dry_run:
        return f"{p} (would remove)"
    if p.is_dir():
        shutil.rmtree(p)
        return f"{p} (removed dir)"
    else:
        p.unlink()
        return f"{p} (removed file)"


def main():
    parser = argparse.ArgumentParser(description="Clean up checkpoint directories.")
    parser.add_argument("--dirs", nargs="*", default=None, help="Directories to remove (relative or absolute).")
    parser.add_argument("--dry-run", action="store_true", help="Preview without deleting.")
    args = parser.parse_args()

    targets = [Path(d) for d in args.dirs] if args.dirs else DEFAULT_DIRS
    for t in targets:
        msg = remove_path(t, args.dry_run)
        print(msg)


if __name__ == "__main__":
    main()

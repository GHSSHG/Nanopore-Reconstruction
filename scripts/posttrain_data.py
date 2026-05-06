#!/usr/bin/env python3
from __future__ import annotations

import importlib
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


COMMANDS: dict[str, tuple[str, str]] = {
    "build-candidates": (
        "codec.posttrain.tools.build_candidates",
        "Build readstats-only post-training candidate manifests.",
    ),
    "enrich-truth": (
        "codec.posttrain.tools.enrich_truth",
        "Fetch truth sequences from BAM for candidate reads.",
    ),
    "merge-truth": (
        "codec.posttrain.tools.merge_truth",
        "Merge candidate/truth datasets.",
    ),
    "materialize-shards": (
        "codec.posttrain.tools.make_shards",
        "Materialize POD5-backed post-training shard datasets.",
    ),
    "merge-shards": (
        "codec.posttrain.tools.merge_shards",
        "Merge materialized shard datasets.",
    ),
    "check-nll": (
        "codec.posttrain.tools.check_nll",
        "Check Dorado SUP CRF NLL on materialized shards.",
    ),
}


def _print_help() -> None:
    print("Usage: python scripts/posttrain_data.py <command> [args]\n")
    print("Commands:")
    width = max(len(name) for name in COMMANDS)
    for name, (_module, description) in COMMANDS.items():
        print(f"  {name:<{width}}  {description}")
    print("\nRun a command with --help for command-specific options.")


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _print_help()
        return
    command = args[0]
    if command not in COMMANDS:
        print(f"Unknown posttrain data command: {command}", file=sys.stderr)
        _print_help()
        raise SystemExit(2)

    module_name, _description = COMMANDS[command]
    module = importlib.import_module(module_name)
    sys.argv = [f"{Path(__file__).name} {command}", *args[1:]]
    module.main()


if __name__ == "__main__":
    main()


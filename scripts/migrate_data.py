"""Move generated data outside File_Router so the repo can be replaced wholesale.

Everything File_Router generates -- downloads, indexes, logs, experiment results
-- currently lives inside the repo, so re-uploading the folder destroys ~100MB
that costs about 2.5 hours to rebuild.  This moves those directories to a
sibling and points the config at them.

Run once per machine:

    python3 -m scripts.migrate_data            # show what would move
    python3 -m scripts.migrate_data --apply    # actually move it
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PROJECT_DIR = Path(__file__).resolve().parents[1]

# Directory -> the config key that must follow it.  experiment_campaigns has no
# config key; the shell scripts point at it via CAMPAIGN_ROOT.
MOVABLE = ["data", "store", "logs", "experiment_campaigns", "validation_runs"]

NEW_PATHS = {
    "data_dir": "../file_router_data/data",
    "raw_dir": "../file_router_data/data/raw",
    "unified_dir": "../file_router_data/data/unified",
    "resources_dir": "../file_router_data/data/resources",
    "splits_dir": "../file_router_data/data/splits",
    "store_dir": "../file_router_data/store",
    "log_dir": "../file_router_data/logs",
    "training_data_path": "../file_router_data/logs/router_training.jsonl",
    "router_model_dir": "../file_router_data/store/router_model",
}


def _size(path: Path) -> str:
    if not path.exists():
        return "—"
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    for unit in ("B", "KB", "MB", "GB"):
        if total < 1024:
            return f"{total:.0f}{unit}"
        total /= 1024
    return f"{total:.1f}TB"


def migrate(target: Path, apply: bool) -> int:
    target.mkdir(parents=True, exist_ok=True) if apply else None
    moved = []

    print(f"[migrate] File_Router : {PROJECT_DIR}")
    print(f"[migrate] data target : {target.resolve() if apply else target}")
    print()
    print(f"{'directory':<24} {'size':>8}  action")
    print("-" * 56)

    for name in MOVABLE:
        source = PROJECT_DIR / name
        destination = target / name
        if not source.exists():
            print(f"{name:<24} {'—':>8}  not present, skipped")
            continue
        if source.is_symlink():
            print(f"{name:<24} {'—':>8}  already a symlink, skipped")
            continue
        if destination.exists():
            print(f"{name:<24} {_size(source):>8}  "
                  f"TARGET EXISTS -- merge manually, skipped")
            continue

        print(f"{name:<24} {_size(source):>8}  "
              f"{'moving' if apply else 'would move'} -> {destination}")
        if apply:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            # A symlink keeps every relative path in the codebase working, so
            # nothing has to know the data now lives outside the repo.
            source.symlink_to(destination.resolve(), target_is_directory=True)
            moved.append(name)

    print()
    if not apply:
        print("[migrate] dry run; nothing changed. Re-run with --apply")
        return 0

    _rewrite_config(apply=True)
    print(f"[migrate] moved {len(moved)} directories and left symlinks behind")
    print("[migrate] config paths now point at ../file_router_data/")
    print()
    print("[migrate] you can now replace the whole File_Router folder without")
    print("[migrate] losing downloads, indexes, or experiment results.")
    return 0


def _rewrite_config(apply: bool) -> None:
    """Point config paths at the new location.

    Uses line-wise replacement rather than a YAML round-trip so the file keeps
    its comments -- they carry the reasoning behind several tuned values.
    """
    config_path = PROJECT_DIR / "config" / "file_router.yaml"
    if not config_path.exists():
        print("[migrate] WARNING: config not found; paths not rewritten")
        return

    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    out, changed = [], 0
    for line in lines:
        stripped = line.strip()
        replaced = False
        for key, value in NEW_PATHS.items():
            if stripped.startswith(f"{key}:"):
                indent = line[:len(line) - len(line.lstrip())]
                comment = ""
                if "#" in line:
                    comment = "  " + line[line.index("#"):].rstrip()
                out.append(f'{indent}{key}: "{value}"{comment}\n')
                changed += 1
                replaced = True
                break
        if not replaced:
            out.append(line)

    if apply and changed:
        config_path.write_text("".join(out), encoding="utf-8")
    print(f"[migrate] rewrote {changed} config paths")


def relink(target: Path) -> int:
    """Restore symlinks and config paths after the repo was replaced.

    Uploading the whole folder overwrites the symlinks and the config, but not
    the data itself (that lives outside).  This puts the pointers back, so the
    upload workflow is: replace folder, run --relink, carry on.
    """
    if not target.exists():
        print(f"[migrate] no data directory at {target}")
        print("[migrate] run --apply first to move data out of the repo")
        return 1

    print(f"[migrate] data found at {target.resolve()}")
    restored = 0
    for name in MOVABLE:
        source = PROJECT_DIR / name
        destination = target / name
        if not destination.exists():
            continue
        if source.is_symlink():
            if source.resolve() == destination.resolve():
                continue
            source.unlink()
        elif source.exists():
            # Uploading the repo brings its own data/ and store/ along, so what
            # sits here is usually a stale copy from the upload machine -- but it
            # could equally be the only copy of something never migrated.  Never
            # delete: rename to .stale so the symlink can be created and the old
            # contents remain recoverable.
            if any(source.iterdir()):
                stale = source.with_name(f"{name}.stale")
                if stale.exists():
                    print(f"[migrate] {name}: both a directory and {stale.name} "
                          "are in the way; clean up manually, then re-run")
                    continue
                source.rename(stale)
                print(f"[migrate] {name}: moved aside -> {stale.name} "
                      f"({_size(stale)}); delete it once you have verified "
                      "the migrated data is the one you want")
            else:
                source.rmdir()
        source.symlink_to(destination.resolve(), target_is_directory=True)
        print(f"[migrate] relinked {name}")
        restored += 1

    _rewrite_config(apply=True)
    print(f"[migrate] restored {restored} symlinks")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="../file_router_data",
                        help="where the generated data should live")
    parser.add_argument("--apply", action="store_true",
                        help="actually move (default is a dry run)")
    parser.add_argument("--relink", action="store_true",
                        help="after re-uploading the repo: restore symlinks "
                             "and config paths (data is left untouched)")
    args = parser.parse_args()

    target = Path(args.target)
    if not target.is_absolute():
        target = PROJECT_DIR / target
    if args.relink:
        return relink(target)
    return migrate(target, args.apply)


if __name__ == "__main__":
    raise SystemExit(main())

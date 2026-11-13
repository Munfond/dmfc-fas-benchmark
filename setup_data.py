"""Prepare public face anti-spoofing datasets for the benchmark."""

import argparse
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from benchmark import write_csv
from prepare_eval import VIDEO_SUFFIXES, prepare

REPLAY_SPLITS = {"train": "reference", "devel": "calibration", "test": "test"}
REPLAY_COUNTS = {"train": (60, 300), "devel": (60, 300), "test": (80, 400)}
UNIATTACK_LABELS = {
    "1_0_0": "print",
    "1_0_1": "replay",
    "1_0_2": "cutout",
    "1_1_0": "3d_mask_transparent",
    "1_1_1": "3d_mask_plaster",
    "1_1_2": "3d_mask_resin",
}
UNIATTACK_SPLITS = {"train": "reference", "val": "calibration", "test": "test"}
SCRIPTS = Path(__file__).resolve().parent


def replay_inventory(root):
    rows, counts, subjects = [], Counter(), {}
    for group, split in REPLAY_SPLITS.items():
        for category in ("real", "attack"):
            folder = root / group / category
            if not folder.is_dir():
                raise ValueError(f"Missing {folder}. Point --root to the folder containing train, devel and test.")
            for clip in sorted(folder.rglob("*")):
                if not clip.is_file() or clip.suffix.lower() not in VIDEO_SUFFIXES:
                    continue
                parts = clip.stem.split("_")
                subject = parts[0]
                attack = ""
                if category == "attack":
                    if len(parts) < 3 or parts[0] != "attack" or parts[1] not in {"print", "mobile", "highdef"}:
                        raise ValueError(f"Unexpected Replay-Attack filename: {clip.name}")
                    subject = parts[2]
                    attack = "print" if parts[1] == "print" else "replay"
                if not re.fullmatch(r"client\d+", subject):
                    raise ValueError(f"Missing client ID in {clip.name}")
                if subjects.setdefault(subject, group) != group:
                    raise ValueError(f"{subject} appears in multiple official splits")
                counts[group, category] += 1
                if group == "train" and attack:
                    continue
                rows.append(dict(
                    path=str(clip.resolve()), source_dataset="Replay-Attack", split=split,
                    label="spoof" if attack else "live", attack_type=attack,
                    subject_id=subject, video_id=clip.relative_to(root).as_posix(),
                ))
    actual = {group: (counts[group, "real"], counts[group, "attack"]) for group in REPLAY_SPLITS}
    if actual != REPLAY_COUNTS:
        raise ValueError(f"Incomplete Replay-Attack Grandtest. Found {actual}, expected {REPLAY_COUNTS}.")
    return rows


def uniattack_inventory(root, test_protocol, minimum):
    rows, paths, test_counts = [], set(), Counter()
    for group, split in UNIATTACK_SPLITS.items():
        protocol = test_protocol if group == "test" and test_protocol else root / f"Protocol-{group}.txt"
        for number, line in enumerate(protocol.read_text(encoding="utf-8-sig").splitlines(), 1):
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 2:
                raise ValueError(
                    f"{protocol}:{number}: expected PATH LABEL. A path-only test list needs "
                    "official subtype labels via --test-protocol."
                )
            relative, code = parts
            if code.startswith("2_"):
                continue
            live = code == "0_0_0"
            if not live and code not in UNIATTACK_LABELS:
                raise ValueError(f"{protocol}:{number}: unknown code {code!r}")
            if split == "reference" and not live:
                continue
            media = (root / relative).resolve()
            if not media.is_relative_to(root):
                raise ValueError(f"{protocol}:{number}: path is outside --root")
            if media in paths:
                raise ValueError(f"{protocol}:{number}: duplicate media path {relative}")
            paths.add(media)
            attack = "" if live else UNIATTACK_LABELS[code]
            if split == "test":
                test_counts[attack or "live"] += 1
            rows.append(dict(
                path=str(media), source_dataset="UniAttackData+", split=split,
                label="live" if live else "spoof", attack_type=attack,
                video_id=media.relative_to(root).as_posix(),
            ))

    required = {"live": 1, **dict.fromkeys(UNIATTACK_LABELS.values(), minimum)}
    print("UniAttackData+ test entries: " + ", ".join(f"{name}={test_counts[name]}" for name in required))
    missing = [f"{name} ({test_counts[name]}/{count})" for name, count in required.items()
               if test_counts[name] < count]
    if missing:
        raise ValueError("Insufficient physical test coverage: " + ", ".join(missing))
    live_splits = {row["split"] for row in rows if row["label"] == "live"}
    if not {"reference", "calibration"}.issubset(live_splits):
        raise ValueError("Need bona fide entries in both train and validation protocols")
    return rows


def main(args):
    output = args.output or Path("data") / args.source
    if args.source == "smoke":
        return subprocess.call([sys.executable, str(SCRIPTS / "prepare_smoke.py"), "--output", str(output)])
    if args.source == "wmca":
        command = [sys.executable, str(SCRIPTS / "prepare_wmca.py"),
                   "--metadata-dir", str(output / "metadata"),
                   "--manifest", str(output / "manifest.csv"),
                   "--frames-per-clip", str(args.frames_per_clip)]
        if args.metadata_only:
            command.append("--metadata-only")
        else:
            command.extend(("--rgb-root", str(args.root), "--color-order", args.color_order))
            if args.dataset_key:
                command.extend(("--dataset-key", args.dataset_key))
        return subprocess.call(command)

    if args.source == "uniattack-plus":
        rows = uniattack_inventory(args.root.resolve(), args.test_protocol, args.min_test_per_category)
        if args.metadata_only:
            return 0
    else:
        rows = replay_inventory(args.root.resolve())
    output.mkdir(parents=True, exist_ok=True)
    inventory = output / "inventory.csv"
    write_csv(inventory, rows, tuple(rows[0]))
    prepare(SimpleNamespace(inventory=inventory, manifest=output / "manifest.csv",
                            holdout_source=[], seed=42, reference_groups=0,
                            calibration_groups=0, frames_per_video=args.frames_per_clip))
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_subparsers(dest="source", required=True)
    for name in ("smoke", "wmca", "replay-attack", "uniattack-plus"):
        source = sources.add_parser(name)
        source.add_argument("--output", type=Path, default=None)
        if name != "smoke":
            source.add_argument("--root", type=Path, required=name != "wmca")
            source.add_argument("--frames-per-clip", type=int, default=8)
        if name == "wmca":
            source.add_argument("--metadata-only", action="store_true")
            source.add_argument("--dataset-key")
            source.add_argument("--color-order", choices=("rgb", "bgr"), default="rgb")
        if name == "uniattack-plus":
            source.add_argument("--metadata-only", action="store_true")
            source.add_argument("--test-protocol", type=Path)
            source.add_argument("--min-test-per-category", type=int, default=1)
    args = parser.parse_args()
    if args.source == "uniattack-plus" and args.min_test_per_category < 1:
        parser.error("--min-test-per-category must be positive")
    if args.source != "smoke" and args.frames_per_clip < 1:
        parser.error("--frames-per-clip must be positive")
    if args.source == "wmca" and not args.metadata_only and args.root is None:
        parser.error("wmca requires --root or --metadata-only")
    try:
        sys.exit(main(args))
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")

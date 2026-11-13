"""Build a grouped benchmark manifest from a labeled inventory of local images or videos."""

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from benchmark import FIELDS, sha256, write_csv


GROUP_FIELDS = ("subject_id", "session_id", "video_id", "mask_id", "group_id")
SPLITS = ("reference", "calibration", "test")
VIDEO_SUFFIXES = {".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v"}


def canonical_source(value):
    return value.strip().casefold()


def build_groups(rows):
    parents = list(range(len(rows)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    owners = {}
    for index, row in enumerate(rows):
        source = canonical_source(row["source_dataset"])
        for field in GROUP_FIELDS:
            value = row.get(field, "")
            if value:
                scope = "" if field == "group_id" else source
                key = (scope, field, value)
                parents[root(index)] = root(owners.setdefault(key, index))
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[root(index)].append(row)
    return list(groups.values())


def load_inventory(path):
    rows = []
    seen_hashes = {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"path", "source_dataset", "label", "attack_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Inventory needs columns {', '.join(sorted(required))}")
        for line, raw in enumerate(reader, 2):
            row = {key: (raw.get(key) or "").strip() for key in FIELDS + ("path", "group_id")}
            row["label"] = row["label"].lower()
            row["attack_type"] = row["attack_type"].lower()
            row["split"] = row["split"].lower()
            if (not row["path"] or not row["source_dataset"] or
                    row["label"] not in {"live", "spoof"} or
                    (row["label"] == "spoof" and not row["attack_type"]) or
                    row["split"] not in ("", *SPLITS)):
                raise ValueError(f"Invalid inventory fields on line {line}")
            media = (path.parent / row["path"]).resolve(strict=True)
            if not media.is_file():
                raise ValueError(f"Expected a file on line {line}: {media}")
            if media.suffix.lower() in VIDEO_SUFFIXES and not row["video_id"]:
                row["video_id"] = row["path"]
            if not any(row[field] for field in GROUP_FIELDS):
                raise ValueError(f"Add a grouping ID on line {line}; for frame folders use video_id")
            digest = sha256(media)
            if digest in seen_hashes:
                raise ValueError(f"Duplicate file bytes on lines {seen_hashes[digest]} and {line}")
            seen_hashes[digest] = line
            row["path"] = str(media)
            row["sha256"] = digest
            if row["label"] == "live":
                row["attack_type"] = ""
            rows.append(row)
    if not rows:
        raise ValueError("Inventory has no media")
    return rows


def assign_splits(rows, holdouts, seed, reference_groups, calibration_groups):
    groups = build_groups(rows)
    rng = random.Random(seed)
    rng.shuffle(groups)
    reference_count = calibration_count = 0
    pending = []
    for group in groups:
        splits = {row["split"] for row in group if row["split"]}
        if any(canonical_source(row["source_dataset"]) in holdouts for row in group):
            splits.add("test")
        if len(splits) > 1:
            raise ValueError(f"Connected group crosses explicit splits: {group[0]['path']}")
        if not splits:
            pending.append(group)
            continue
        split = splits.pop()
        if split == "reference" and not any(row["label"] == "live" for row in group):
            raise ValueError(f"Reference group contains no live media: {group[0]['path']}")
        for row in group:
            row["split"] = split
        reference_count += split == "reference"
        calibration_count += split == "calibration" and any(r["label"] == "live" for r in group)
    # Use live-only groups for reference first. A mixed group can provide live
    # references too, but its spoof media must remain outside all scored splits.
    pending.sort(key=lambda group: any(row["label"] == "spoof" for row in group))
    has_test = any(row["split"] == "test" for row in rows)
    for index, group in enumerate(pending):
        labels = {r["label"] for r in group}
        remaining = len(pending) - index
        reserve_for_test = 0 if has_test else 1
        if remaining <= reserve_for_test:
            split = "test"
        elif ("live" in labels and reference_count < reference_groups and
              (calibration_count >= calibration_groups or remaining > reserve_for_test + 1)):
            split = "reference"
            reference_count += 1
        elif "live" in labels and calibration_count < calibration_groups:
            split = "calibration"
            calibration_count += 1
        else:
            split = "test"
        has_test |= split == "test"
        for row in group:
            row["split"] = split

    return rows


def sample_video(row, directory, frames):
    import cv2

    capture = cv2.VideoCapture(row["path"])
    try:
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 1:
            raise ValueError(f"Could not determine frame count: {row['path']}")
        take = min(count, frames)
        for i in range(take):
            frame_index = (2 * i + 1) * count // (2 * take)
            output = directory / f"{row['sha256'][:24]}_{frame_index:06d}.png"
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, pixels = capture.read()
            if not ok or not cv2.imwrite(str(output), pixels):
                raise ValueError(f"Could not extract frame {frame_index}: {row['path']}")
            yield output
    finally:
        capture.release()


def prepare(args):
    rows = load_inventory(args.inventory)
    holdouts = {canonical_source(value) for value in args.holdout_source}
    absent = holdouts - {canonical_source(row["source_dataset"]) for row in rows}
    if absent:
        raise ValueError(f"Unknown holdout sources: {', '.join(sorted(absent))}")
    rows = assign_splits(rows, holdouts, args.seed, args.reference_groups, args.calibration_groups)
    excluded = sum(row["split"] == "reference" and row["label"] == "spoof" for row in rows)
    if excluded:
        print(f"Excluded {excluded} spoof files grouped with reference subjects or sessions")
    rows = [row for row in rows if row["split"] != "reference" or row["label"] == "live"]
    if not any(row["split"] == "test" for row in rows):
        raise ValueError("No test groups. Add data or designate a holdout source.")
    for split in SPLITS:
        if not any(row["split"] == split and row["label"] == "live" for row in rows):
            print(f"Warning: {split} has no live samples")
    if not any(row["split"] == "test" and row["label"] == "spoof" for row in rows):
        print("Warning: test has no spoof samples")

    manifest = args.manifest.resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    frame_dir = manifest.parent / f"{manifest.stem}_frames"
    output = []
    for row in rows:
        path = Path(row["path"])
        if path.suffix.lower() in VIDEO_SUFFIXES:
            frame_dir.mkdir(exist_ok=True)
            paths = sample_video(row, frame_dir, args.frames_per_video)
        else:
            paths = [path]
        output.extend({**{k: row[k] for k in FIELDS}, "image_path": str(p)} for p in paths)
    write_csv(manifest, output, FIELDS)
    counts = defaultdict(int)
    for row in output:
        key = (row["split"], row["source_dataset"], row["label"], row["attack_type"])
        counts[key] += 1
    summary = [dict(zip(("split", "source_dataset", "label", "attack_type", "frames"), (*key, n)))
               for key, n in sorted(counts.items())]
    manifest.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Wrote {len(output)} rows to {manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--holdout-source", action="append", default=[])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reference-groups", type=int, default=13)
    parser.add_argument("--calibration-groups", type=int, default=30)
    parser.add_argument("--frames-per-video", type=int, default=8)
    args = parser.parse_args()
    if args.frames_per_video < 1 or min(args.reference_groups, args.calibration_groups) < 0:
        parser.error("Invalid frame or group counts")
    try:
        prepare(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")

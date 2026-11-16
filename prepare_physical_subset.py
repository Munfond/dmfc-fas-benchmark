"""Export a balanced six-attack test subset within a 5 GB file-size budget."""

import argparse
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

from benchmark import FIELDS, read_manifest, sha256, write_csv
from setup_data import UNIATTACK_LABELS

CLASSES = ("live", *UNIATTACK_LABELS.values())
MASK_CLASSES = frozenset(name for name in CLASSES if name.startswith("3d_mask_"))
MAX_BYTES = 5_000_000_000


def read_candidates(manifest):
    clips, support = defaultdict(list), []
    splits_by_id = {}
    for row, image in read_manifest(manifest):
        split = row["split"]
        category = row["attack_type"] if row["label"] == "spoof" else row["label"]
        if split == "test" and category not in CLASSES:
            continue
        source = row["source_dataset"].casefold()
        if source == "unknown" or not row["subject_id"]:
            raise ValueError(f"Source and subject_id are required: {image}")
        if split == "test" and category in MASK_CLASSES and not row["mask_id"]:
            raise ValueError(f"Physical mask_id is required for {category}: {image}")
        for field in ("subject_id", "session_id", "mask_id"):
            if row[field]:
                identifier = source, field, row[field]
                if splits_by_id.setdefault(identifier, split) != split:
                    raise ValueError(f"{field} crosses splits: {identifier}")
        if not image.is_file():
            raise ValueError(f"Media file missing: {image}")
        if split != "test":
            support.append((row, image))
            continue
        clip = row["video_id"] or str(image)
        clips[source, bool(row["video_id"]), clip].append((row, image))
    by_category = defaultdict(list)
    for entries in clips.values():
        first = entries[0][0]
        fields = ("label", "attack_type", "subject_id", "session_id", "mask_id")
        if any(any(row[key] != first[key] for key in fields) for row, _ in entries):
            raise ValueError(f"Inconsistent clip metadata: {entries[0][1]}")
        category = first["attack_type"] if first["label"] == "spoof" else "live"
        by_category[category].append(entries)
    return by_category, support


def diversity_order(candidates, seed, target):
    """Prefer less represented sources, people, and physical masks per category."""
    def rank(clip):
        row, image = clip[0]
        stable_id = row["video_id"] or row["image_path"]
        return hashlib.sha256(f"{seed}|{row['source_dataset']}|{stable_id}".encode()).digest(), image

    remaining = sorted(candidates, key=rank)
    sources, people, masks = Counter(), Counter(), Counter()

    def keys(clip):
        row = clip[0][0]
        source = row["source_dataset"].casefold()
        return source, (source, row["subject_id"]), (source, row["mask_id"])

    for _ in range(min(target, len(remaining))):
        def priority(clip):
            source, subject, mask = keys(clip)
            return masks[mask], people[subject], sources[source]

        index = min(range(len(remaining)), key=lambda index: priority(remaining[index]))
        clip = remaining.pop(index)
        source, subject, mask = keys(clip)
        sources[source] += 1
        people[subject] += 1
        if mask[1]:
            masks[mask] += 1
        yield clip


def summarize(clips):
    counts = Counter(clip[0][0]["source_dataset"] for clip in clips)
    ids = defaultdict(set)
    for clip in clips:
        row = clip[0][0]
        for key in ("subject_id", "session_id", "mask_id"):
            if row[key]:
                ids[key].add((row["source_dataset"].casefold(), row[key]))
    return dict(presentations=len(clips), frames=sum(map(len, clips)),
                sources=dict(sorted(counts.items())), subjects=len(ids["subject_id"]),
                sessions=len(ids["session_id"]), mask_instances=len(ids["mask_id"]))


def plan(candidate_groups, args):
    groups = {}
    for category in CLASSES:
        candidates = candidate_groups[category]
        if len(candidates) < args.per_class:
            raise ValueError(f"{category}: need {args.per_class} presentations, found {len(candidates)}")
        selected = list(diversity_order(candidates, args.seed, args.per_class))
        stats = summarize(selected)
        if stats["subjects"] < args.min_subjects:
            raise ValueError(f"{category}: selected {stats['subjects']} subjects, need {args.min_subjects}")
        if category in MASK_CLASSES and stats["mask_instances"] < args.min_masks:
            raise ValueError(f"{category}: selected {stats['mask_instances']} physical masks, need {args.min_masks}")
        groups[category] = selected
    return groups


def export(groups, support, args):
    output = args.output.resolve()
    if output.exists():
        raise ValueError(f"Output already exists: {output}. Choose a new directory.")
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [{**row, "image_path": str(image)} for row, image in support]
    transfers, split_by_hash = [], {}
    for row, image in support:
        split = row["split"]
        prior = split_by_hash.setdefault(sha256(image), split)
        if prior != split:
            raise ValueError(f"Identical media appears across {prior} and {split}: {image}")
    for category, clips in groups.items():
        for index, clip in enumerate(clips, 1):
            for frame, (row, image) in enumerate(clip, 1):
                digest = sha256(image)
                presentation = (category, index)
                prior = split_by_hash.setdefault(digest, presentation)
                if prior != presentation:
                    raise ValueError(f"Identical media appears in different presentations or splits: {image}")
                relative = Path("test") / category / f"{index:05d}" / f"{frame:04d}{image.suffix.lower()}"
                transfers.append((image, relative, image.stat().st_size))
                rows.append({**row, "image_path": relative.as_posix()})

    media_bytes = sum(size for _, _, size in transfers)
    if media_bytes >= MAX_BYTES:
        raise ValueError(f"Selected test media uses {media_bytes} bytes. Lower --per-class or sampling frames per clip.")

    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory) / "subset"
        staging.mkdir()
        manifest = staging / "manifest.csv"
        report = staging / "distribution.json"
        write_csv(manifest, rows, FIELDS)
        report.write_text(json.dumps({
            "seed": args.seed,
            "input_manifest_sha256": sha256(args.manifest),
            "budget_bytes": MAX_BYTES,
            "test_media_bytes": media_bytes,
            "manifest_bytes": manifest.stat().st_size,
            "presentations_per_class": args.per_class,
            "reference_calibration_media": "external",
            "classes": {category: summarize(clips) for category, clips in groups.items()},
        }, indent=2) + "\n", encoding="utf-8")
        expected_bytes = media_bytes + manifest.stat().st_size + report.stat().st_size
        if expected_bytes > MAX_BYTES:
            raise ValueError(f"Output requires {expected_bytes} bytes. Lower --per-class or frames per clip.")
        for image, relative, size in transfers:
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(image, destination)
            if destination.stat().st_size != size:
                raise ValueError(f"Source size changed while copying: {image}")
        if output.exists():
            raise ValueError(f"Output was created during export: {output}")
        staging.rename(output)
    print(f"Exported {args.per_class} presentations per class, {expected_bytes} bytes, to {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/physical-test"))
    parser.add_argument("--per-class", type=int, default=100)
    parser.add_argument("--min-subjects", type=int, default=20)
    parser.add_argument("--min-masks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if not 1 <= min(args.min_subjects, args.min_masks) <= max(args.min_subjects, args.min_masks) <= args.per_class:
        parser.error("Minimum counts must be positive and no greater than --per-class.")
    try:
        candidates, support = read_candidates(args.manifest)
        export(plan(candidates, args), support, args)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

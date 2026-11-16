"""Download and prepare the official CelebA-Spoof image benchmark."""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from benchmark import FIELDS, write_csv

DRIVE_FOLDER = "https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z"
SHARDS = {f"CelebA_Spoof.zip.{index:03d}": 1_048_576_000 for index in range(1, 74)}
SHARDS["CelebA_Spoof.zip.074"] = 453_369_048
ATTACK_TYPES = ("", "print", "print", "print", "cutout", "cutout", "cutout",
                "replay", "replay", "replay", "3d_mask_unspecified")


def get_shards(directory):
    gdown = [sys.executable, "-m", "gdown", "--no-cookies", "--timeout", "60"]
    result = subprocess.run([*gdown, "--json", DRIVE_FOLDER], check=True, text=True, stdout=subprocess.PIPE)
    urls = {entry["path"]: entry["url"] for entry in json.loads(result.stdout)}
    if urls.keys() != SHARDS.keys():
        raise ValueError("The official Drive listing differs from the expected 74-part release.")
    directory.mkdir(parents=True, exist_ok=True)
    for name, size in SHARDS.items():
        target = directory / name
        if target.is_file():
            if target.stat().st_size != size:
                raise ValueError(f"{target} has the wrong size. Move it aside before retrying.")
            continue
        subprocess.run([*gdown, "--continue", "--retries", "3", "-O", str(target), urls[name]], check=True)
        if not target.is_file() or target.stat().st_size != size:
            raise ValueError(f"Incomplete download: {target}")


def extract_shards(directory, output):
    for name, size in SHARDS.items():
        part = directory / name
        if not part.is_file() or part.stat().st_size != size:
            raise ValueError(f"Missing or incomplete archive part: {part}")
    extractor = next((path for name in ("7zz", "7z") if (path := shutil.which(name))), None)
    if extractor is None:
        raise ValueError("Install 7-Zip and put 7z or 7zz on PATH.")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Use an empty extraction directory.")
    output.mkdir(parents=True, exist_ok=True)
    subprocess.run([extractor, "x", str((directory / next(iter(SHARDS))).resolve()),
                    f"-o{output.resolve()}"], check=True)


def load_annotations(root, split):
    metadata = root / "metas" / "intra_test" / f"{split}_label.json"
    labels = json.loads(metadata.read_text(encoding="utf-8-sig"))
    if not isinstance(labels, dict) or not labels:
        raise ValueError(f"Expected a nonempty path-to-label object in {metadata}")
    for name, attributes in labels.items():
        match = re.fullmatch(rf"Data/{split}/([^/]+)/(live|spoof)/[^/]+\.(?:jpg|jpeg|png)", name, re.IGNORECASE)
        if not match or not isinstance(attributes, list) or len(attributes) < 44:
            raise ValueError(f"Unexpected annotation in {metadata}: {name}")
        subject = match.group(1)
        subtype, binary = attributes[40], attributes[43]
        live = binary == 0
        if (type(subtype) is not int or not 0 <= subtype < len(ATTACK_TYPES)
                or binary not in (0, 1) or (subtype == 0) != live):
            raise ValueError(f"Inconsistent official labels for {name}: {subtype}, {binary}")
        yield dict(image_path=name, source_dataset="CelebA-Spoof", label="live" if live else "spoof",
                   attack_type=ATTACK_TYPES[subtype], subject_id=subject)


def prepare(root, manifest, metadata_only, reference_subjects, calibration_subjects):
    root = root.resolve()
    train_live, train_subjects = {}, set()
    for row in load_annotations(root, "train"):
        subject = row["subject_id"]
        train_subjects.add(subject)
        if row["label"] == "live" and (subject not in train_live
                or row["image_path"] < train_live[subject]["image_path"]):
            train_live[subject] = row

    rows, test_counts, test_subjects = [], Counter(), set()
    for row in load_annotations(root, "test"):
        test_subjects.add(row["subject_id"])
        test_counts[row["attack_type"] or "live"] += 1
        rows.append({**row, "split": "test"})

    subjects = sorted(train_live.keys() - test_subjects,
                      key=lambda subject: hashlib.sha256(subject.encode()).digest())
    requested = reference_subjects + calibration_subjects
    if len(subjects) < requested:
        raise ValueError(f"Need {requested} eligible train subjects, found {len(subjects)}")
    rows.extend({**train_live[subject], "split": "reference" if i < reference_subjects else "calibration"}
                for i, subject in enumerate(subjects[:requested]))
    missing_types = sorted({"live", *ATTACK_TYPES[1:]} - test_counts.keys())
    if missing_types:
        raise ValueError("Incomplete CelebA-Spoof test metadata. Missing: " + ", ".join(missing_types))

    print(json.dumps({
        "unit": "image",
        "reference": reference_subjects,
        "calibration": calibration_subjects,
        "official_test_subjects": len(test_subjects),
        "train_test_shared_subject_ids": len(train_subjects & test_subjects),
        "test": dict(sorted(test_counts.items())),
    }, indent=2))
    if metadata_only:
        return

    for row in rows:
        image = (root / row["image_path"]).resolve()
        if not image.is_relative_to(root) or not image.is_file():
            raise ValueError(f"Missing or invalid image in the selected release: {row['image_path']}")
        row["image_path"] = str(image)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    write_csv(manifest, rows, FIELDS)
    print(f"Wrote {len(rows)} rows to {manifest}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    download = commands.add_parser("download")
    download.add_argument("--parts", type=Path, default=Path("data/celeba-spoof-parts"))
    extract = commands.add_parser("extract")
    extract.add_argument("--parts", type=Path, default=Path("data/celeba-spoof-parts"))
    extract.add_argument("--output", type=Path, default=Path("data/celeba-spoof-raw"))
    importer = commands.add_parser("prepare")
    importer.add_argument("--root", type=Path, required=True)
    importer.add_argument("--manifest", type=Path, default=Path("data/celeba-spoof/manifest.csv"))
    importer.add_argument("--reference-subjects", type=int, default=50)
    importer.add_argument("--calibration-subjects", type=int, default=200)
    importer.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()
    try:
        if args.command == "download":
            get_shards(args.parts)
        elif args.command == "extract":
            extract_shards(args.parts, args.output)
        else:
            if min(args.reference_subjects, args.calibration_subjects) < 1:
                parser.error("Reference and calibration subject counts must be positive.")
            prepare(args.root, args.manifest, args.metadata_only,
                    args.reference_subjects, args.calibration_subjects)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()

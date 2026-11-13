"""Convert the licensed Idiap WMCA RGB release into a Grandtest benchmark manifest."""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from contextlib import closing
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from zipfile import ZipFile

from PIL import Image

from benchmark import FIELDS, ensure_asset, write_csv

BOB_ARCHIVE = "bob.db.batl-1.0.0.zip"
BOB_ASSET = (
    "https://files.pythonhosted.org/packages/35/be/"
    "28355d07374b39378c78acd45fffce5592d86c5dfa7e45be386022440600/" + BOB_ARCHIVE,
    "8d4c29a8d45f6457df96b956ae3ad11e4571f66689696d7163740025db759bd2",
)
SPLITS = {"train": "reference", "validation": "calibration", "test": "test"}
EXPECTED = {"train": (124, 441), "validation": (108, 449), "test": (115, 442)}
# BATL 1.0.0 taxonomy. The journal Grandtest excludes wigs and makeup.
ATTACKS = {
    (0, 0): "", (0, 1): "", (0, 2): "",
    **{(1, pai): "glasses" for pai in (1, 4, 5, 6, 7, 15, 16)},
    **{(1, pai): "3d_mask_rigid" for pai in (2, 13, 14)},
    **{(1, pai): None for pai in range(8, 13)},
    **{(2, pai): "fake_head" for pai in range(1, 8)},
    **{(2, pai): "paper_mask" for pai in range(8, 12)},
    **{(2, pai): "3d_mask_rigid" for pai in range(12, 18)},
    **{(2, pai): "3d_mask_flexible" for pai in (18, *range(75, 100))},
    **{(3, pai): "replay" if pai in (5, 10, 15) else "print" for pai in range(1, 16)},
    **{(4, pai): "replay" for pai in range(1, 19)},
}
QUERY = """
    SELECT pp."group", pp.purpose, v.path, v.client_id, v.session_id, v.type_id, v.pai_id
    FROM video_file v
    JOIN protocolPurpose_file_association a ON a.file_id = v.id
    JOIN protocolPurpose pp ON pp.id = a.protocolPurpose_id
    JOIN protocol p ON p.id = pp.protocol_id
    WHERE p.name = 'baseline'
    ORDER BY v.path
"""


def load_protocol(archive):
    member = "bob.db.batl-1.0.0/bob/db/batl/db.sql3"
    with TemporaryDirectory() as directory, ZipFile(archive) as package:
        database = Path(directory) / "protocol.sqlite"
        database.write_bytes(package.read(member))
        with closing(sqlite3.connect(database)) as connection:
            records = connection.execute(QUERY).fetchall()
    counts = Counter()
    client_splits, instrument_splits = defaultdict(set), defaultdict(set)
    rows = []
    for group, purpose, path, client, session, kind, pai in records:
        attack = None if kind == 5 else ATTACKS[kind, pai]
        if attack is None:
            continue
        label = "live" if kind == 0 else "spoof"
        if purpose != ("real" if label == "live" else "attack"):
            raise ValueError(f"Inconsistent protocol label for {path}")
        counts[group, label] += 1
        client_splits[client].add(group)
        if attack:
            instrument_splits[kind, pai].add(group)
        if group == "train" and attack:
            continue
        rows.append(dict(
            source_dataset="WMCA", split=SPLITS[group], label=label, attack_type=attack,
            subject_id=str(client), session_id=f"{client}/{session}", video_id=path,
            mask_id=f"{kind}/{pai}" if "mask" in attack else "",
        ))
    actual = {group: (counts[group, "live"], counts[group, "spoof"]) for group in SPLITS}
    if actual != EXPECTED or len({row["video_id"] for row in rows}) != len(rows):
        raise ValueError(f"Unexpected Grandtest counts or duplicate presentations: {actual}")
    if any(len(groups) > 1 for groups in (*client_splits.values(), *instrument_splits.values())):
        raise ValueError("Grandtest has a client or PAI in multiple splits")
    return rows


def match_files(rgb_root, rows):
    index = defaultdict(list)
    for path in rgb_root.rglob("*"):
        if path.suffix.lower() in (".h5", ".hdf5") and path.is_file():
            index[(path.parent.name, path.stem)].append(path)
    missing, matched = [], []
    for row in rows:
        clip = PurePosixPath(row["video_id"])
        paths = index[clip.parent.name, clip.name]
        if len(paths) > 1:
            raise ValueError(f"Ambiguous RGB files for {clip}. Point --rgb-root to the RGB release only.")
        if paths:
            matched.append((row, paths[0]))
        else:
            missing.append(str(clip))
    if missing:
        raise ValueError(f"Missing {len(missing)} of {len(rows)} RGB clips. Examples: {', '.join(missing[:4])}")
    return matched


def export_frames(h5py, source, target, limit, dataset_key, color_order):
    with h5py.File(source, "r") as handle:
        if dataset_key:
            stack = handle[dataset_key]
            if not isinstance(stack, h5py.Dataset) or stack.ndim != 4:
                raise ValueError(f"{dataset_key} must be a stack of RGB frames in {source}")
            count = len(stack)
        else:
            keys = sorted(
                (key for key in handle if re.fullmatch(r"Frame_\d+", key)),
                key=lambda key: int(key.partition("_")[2]),
            )
            count = len(keys)
        if not count:
            raise ValueError(f"No Frame_N groups in {source}. Inspect HDF5 and set --dataset-key for a 4D RGB stack.")
        target.mkdir(parents=True, exist_ok=True)
        take = min(limit, count)
        for i in range(take):
            index = (2 * i + 1) * count // (2 * take)
            if dataset_key:
                frame = stack[index]
            else:
                frame = handle[keys[index]]["array"][()]
            if frame.ndim != 3 or frame.dtype.name != "uint8":
                raise ValueError(f"Expected a uint8 color frame in {source}, found {frame.shape} {frame.dtype}")
            if frame.shape[0] == 3 and frame.shape[-1] != 3:
                frame = frame.transpose(1, 2, 0)
            if frame.shape[-1] != 3:
                raise ValueError(f"Expected exactly three color channels in {source}, found {frame.shape}")
            if color_order == "bgr":
                frame = frame[:, :, ::-1]
            output = target / f"{index:06d}.png"
            Image.fromarray(frame).save(output)
            yield output


def main(args):
    archive = ensure_asset(args.metadata_dir, BOB_ARCHIVE, *BOB_ASSET, download=True)
    rows = load_protocol(archive)
    counts = Counter((row["split"], row["label"], row["attack_type"]) for row in rows)
    summary = [
        dict(split=split, label=label, attack_type=attack, clips=count)
        for (split, label, attack), count in sorted(counts.items())
    ]
    if args.metadata_only:
        print(json.dumps(summary, indent=2))
        return

    import h5py

    matched = match_files(args.rgb_root, rows)
    manifest = args.manifest.resolve()
    frame_root = manifest.parent / f"{manifest.stem}_frames"
    output = []
    for row, source in matched:
        target = frame_root / row["video_id"]
        for frame in export_frames(h5py, source, target, args.frames_per_clip, args.dataset_key, args.color_order):
            output.append({**row, "image_path": frame.relative_to(manifest.parent).as_posix()})
    write_csv(manifest, output, FIELDS)
    manifest.with_suffix(".summary.json").write_text(
        json.dumps({
            "protocol": "WMCA Grandtest",
            "metadata_sha256": BOB_ASSET[1],
            "frames_per_clip_requested": args.frames_per_clip,
            "color_order": args.color_order,
            "dataset_key": args.dataset_key,
            "total_frames": len(output),
            "counts": summary,
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(output)} frames from {len(matched)} clips to {manifest}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata-dir", type=Path, default=Path("data/wmca/metadata"))
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--rgb-root", type=Path, help="Extracted preprocessed-face-station_RGB directory")
    parser.add_argument("--manifest", type=Path, default=Path("data/wmca/manifest.csv"))
    parser.add_argument("--frames-per-clip", type=int, default=8)
    parser.add_argument("--dataset-key", help="Explicit HDF5 dataset for an N×H×W×3 or N×3×H×W RGB stack")
    parser.add_argument("--color-order", choices=("rgb", "bgr"), default="rgb")
    args = parser.parse_args()
    if args.frames_per_clip < 1:
        parser.error("--frames-per-clip must be positive")
    if not args.metadata_only and (args.rgb_root is None or not args.rgb_root.is_dir()):
        parser.error("--rgb-root must point to an extracted WMCA RGB directory, or use --metadata-only")
    try:
        main(args)
    except (ImportError, KeyError, OSError, ValueError, sqlite3.Error) as error:
        parser.exit(1, f"Error: {error}\n")

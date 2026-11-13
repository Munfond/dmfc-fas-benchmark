"""Fetch upstream demonstration images and make a smoke-test manifest."""

import argparse
from pathlib import Path
from urllib.request import urlopen

from benchmark import FIELDS, MINI_URL, write_csv


SAMPLES = [
    ("image_T1.jpg", "reference", "live", ""),
    ("image_F1.jpg", "test", "spoof", "unknown"),
    ("image_F2.jpg", "test", "spoof", "unknown"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for filename, split, label, attack in SAMPLES:
        with urlopen(f"{MINI_URL}/images/sample/{filename}", timeout=45) as response:
            (args.output / filename).write_bytes(response.read())
        rows.append(dict(image_path=filename, source_dataset="minivision_demo", split=split,
                         label=label, attack_type=attack, video_id=filename))
    write_csv(args.output / "manifest.csv", rows, FIELDS)
    print(f"Wrote {args.output / 'manifest.csv'} for pipeline smoke testing only")


if __name__ == "__main__":
    main()

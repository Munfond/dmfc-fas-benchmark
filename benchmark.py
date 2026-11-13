"""Benchmark RGB face anti-spoofing checkpoints on a shared image manifest."""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from urllib.request import urlopen

import numpy as np
from PIL import Image, ImageOps


DMFC_REV = "ce33f053c9b9c9ca2d9daac9f9e35086a7a1198e"
MINI_REV = "b6d5f04ad78778917853b25c778acef6d5626d15"
MINI_URL = f"https://raw.githubusercontent.com/minivision-ai/Silent-Face-Anti-Spoofing/{MINI_REV}"
ASSETS = {
    "dmfc.pth": (
        f"https://raw.githubusercontent.com/wangjl1993/DMFC-FAS/{DMFC_REV}/exp/"
        "resnet34%2Boversampling%2Buse_mixup%2Bseed_42/model_epoch_150.pth",
        "fed51ee29763b2b17159862ec15dbf8bd812db1f59a6d4bec480119bb2aed548",
    ),
    "mn3.onnx": (
        "https://storage.openvinotoolkit.org/repositories/open_model_zoo/public/2022.1/"
        "anti-spoof-mn3/anti-spoof-mn3.onnx",
        "c4c99af04603b62d7e44f6f4daeb33e0daeccc696008c0b1d62f6f5cebbb3262",
    ),
    "mini.pth": (
        f"{MINI_URL}/resources/anti_spoof_models/2.7_80x80_MiniFASNetV2.pth",
        "a5eb02e1843f19b5386b953cc4c9f011c3f985d0ee2bb9819eea9a142099bec0",
    ),
    "minifasnet_upstream.py": (
        f"{MINI_URL}/src/model_lib/MiniFASNet.py",
        "e498c4ec5e1ddfaba62b941a126c19d65aa564999f3309661fe43ee8bf38acd7",
    ),
}
MODEL_ASSETS = {
    "dmfc": ("dmfc.pth",),
    "mn3": ("mn3.onnx",),
    "mini": ("mini.pth", "minifasnet_upstream.py"),
}
RESEARCH_MODELS = ("gac", "mvp", "flip-mcl")
MODEL_NAMES = (*MODEL_ASSETS, *RESEARCH_MODELS)
CLIP_SHA256 = "5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
FIELDS = ("image_path", "source_dataset", "split", "label", "attack_type",
          "subject_id", "session_id", "video_id", "mask_id")
PREDICTION_FIELDS = FIELDS + ("model", "status", "spoof_score", "threshold", "prediction",
                              "preprocess_ms", "model_ms")
METRIC_FIELDS = ("model", "source_dataset", "attack_type", "metric", "errors",
                 "scored", "total", "rate", "coverage", "threshold")
DMFC_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
DMFC_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
MN3_MEAN = np.array([151.2405, 119.5950, 107.8395], dtype=np.float32)
MN3_STD = np.array([63.0105, 56.4570, 55.0035], dtype=np.float32)
FACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
     [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def ensure_asset(directory, name, url, expected, download):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    if path.is_file() and sha256(path) == expected:
        return path
    if not download:
        raise ValueError(f"Missing or unexpected asset {path}. Run the fetch command.")
    print(f"Downloading {name}", flush=True)
    with NamedTemporaryFile(dir=directory, delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            with urlopen(url, timeout=60) as response:
                while block := response.read(1024 * 1024):
                    temporary.write(block)
            temporary.close()
            if sha256(temporary_path) != expected:
                raise ValueError(f"Checksum mismatch for {name}")
            temporary_path.replace(path)
        finally:
            temporary_path.unlink(missing_ok=True)
    return path


def prepare_assets(directory, models, download):
    for name in dict.fromkeys(asset for model in models for asset in MODEL_ASSETS.get(model, ())):
        ensure_asset(directory, name, *ASSETS[name], download=download)


def read_manifest(path):
    rows = []
    paths = set()
    videos = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "source_dataset", "split", "label", "attack_type"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest needs columns {', '.join(sorted(required))}")
        for line, record in enumerate(reader, start=2):
            row = {field: (record.get(field) or "").strip() for field in FIELDS}
            row["split"] = row["split"].lower()
            row["label"] = row["label"].lower()
            row["attack_type"] = row["attack_type"].lower() or (
                "unknown" if row["label"] == "spoof" else ""
            )
            row["source_dataset"] = row["source_dataset"] or "unknown"
            if not row["image_path"] or row["split"] not in {"reference", "calibration", "test"}:
                raise ValueError(f"Invalid path or split at CSV line {line}")
            if row["label"] not in {"", "live", "spoof"}:
                raise ValueError(f"Invalid label at CSV line {line}: expected live or spoof")
            if row["split"] == "reference" and row["label"] != "live":
                raise ValueError(f"Reference must be labeled live at CSV line {line}")
            image = (path.parent / row["image_path"]).resolve()
            if image in paths:
                raise ValueError(f"Duplicate image path at CSV line {line}: {image}")
            paths.add(image)
            if row["video_id"]:
                key = (row["source_dataset"], row["video_id"])
                if videos.setdefault(key, row["split"]) != row["split"]:
                    raise ValueError(f"Video in multiple splits at CSV line {line}: {key}")
            rows.append((row, image))
    if not any(row["split"] == "test" for row, _ in rows):
        raise ValueError("Manifest has no test rows")
    return rows


def resize(image, size):
    return np.asarray(image.resize((size, size), Image.Resampling.BILINEAR), dtype=np.float32)


def resize_cv(image, size, *, cubic=False):
    import cv2

    interpolation = cv2.INTER_CUBIC if cubic else cv2.INTER_LINEAR
    return cv2.resize(np.asarray(image), (size, size), interpolation=interpolation).astype(np.float32)


def chw(pixels):
    return np.ascontiguousarray(pixels.transpose(2, 0, 1)[None], dtype=np.float32)


def mini_crop(image, box):
    left, top, width, height = (float(value) for value in box)
    scale = min(2.7, (image.width - 1) / width, (image.height - 1) / height)
    width_scaled, height_scaled = width * scale, height * scale
    x = min(max(0, left + (width - width_scaled) / 2), image.width - 1 - width_scaled)
    y = min(max(0, top + (height - height_scaled) / 2), image.height - 1 - height_scaled)
    return image.crop((int(x), int(y), int(x + width_scaled) + 1, int(y + height_scaled) + 1))


class FaceCrops:
    def __init__(self, detector_path):
        self.detector = None
        if detector_path:
            import cv2

            self.cv2 = cv2
            self.detector = cv2.FaceDetectorYN.create(str(detector_path), "", (640, 640))

    def read(self, path):
        try:
            with Image.open(path) as image:
                rgb = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, ValueError, Image.DecompressionBombError):
            return "decode_error", {}
        if self.detector is None:
            return "ok", {"dmfc": rgb, "mn3": rgb, "mini": rgb}
        scale = min(1, 640 / max(rgb.size))
        size = tuple(max(1, round(value * scale)) for value in rgb.size)
        frame = np.ascontiguousarray(np.asarray(rgb.resize(size))[:, :, ::-1])
        self.detector.setInputSize(size)
        _, faces = self.detector.detect(frame)
        if faces is None or len(faces) == 0:
            return "no_face", {}
        if len(faces) != 1:
            return "multiple_faces", {}
        face = faces[0]
        scale_xy = np.asarray(rgb.size) / size
        box = face[:4] * np.tile(scale_xy, 2)
        left, top, width, height = box
        x0, y0 = max(0, int(left)), max(0, int(top))
        x1, y1 = min(rgb.width, math.ceil(left + width)), min(rgb.height, math.ceil(top + height))
        if x0 >= x1 or y0 >= y1:
            return "invalid_crop", {}
        landmarks = (face[4:14].reshape(5, 2) * scale_xy).astype(np.float32)
        landmarks[:2] = landmarks[:2][np.argsort(landmarks[:2, 0])]
        landmarks[3:] = landmarks[3:][np.argsort(landmarks[3:, 0])]
        affine, _ = self.cv2.estimateAffinePartial2D(landmarks, 3 * FACE_TEMPLATE, method=self.cv2.LMEDS)
        if affine is None:
            return "alignment_error", {}
        aligned = self.cv2.warpAffine(np.asarray(rgb), affine, (336, 336))
        return "ok", {
            "dmfc": Image.fromarray(aligned),
            "mn3": rgb.crop((x0, y0, x1, y1)),
            "mini": mini_crop(rgb, (x0, y0, x1 - x0, y1 - y0)),
        }


class TorchModel:
    def __init__(self, network, path, device):
        import torch

        self.torch = torch
        state = torch.load(path, map_location="cpu", weights_only=True)
        state = state.get("model_state_dict", state)
        network.load_state_dict({key.removeprefix("module."): value for key, value in state.items()},
                                strict=True)
        self.network = network.eval().to(device)
        self.device = device

    def forward(self, pixels):
        with self.torch.inference_mode():
            return self.network(self.torch.from_numpy(chw(pixels)).to(self.device))


class DMFC(TorchModel):
    def __init__(self, directory, device, tta):
        from torchvision.models import resnet34

        # Upstream wraps torchvision's six-class ResNet34 under the "model" key.
        from torch import nn

        backbone = resnet34(weights=None, num_classes=6)
        container = nn.ModuleDict({"model": backbone})
        super().__init__(container, directory / "dmfc.pth", device)
        self.network = nn.Sequential(*list(backbone.children())[:-1]).eval()
        self.tta = tta
        self.reference = None

    def embedding(self, image):
        pixels = (resize(image, 224) / 255 - DMFC_MEAN) / DMFC_STD
        features = self.forward(pixels).cpu().numpy().reshape(512)
        if self.tta:
            other = self.forward(pixels[:, ::-1]).cpu().numpy().reshape(512)
            features = (features + other) / 2
        return features

    def score(self, image):
        embedding = self.embedding(image)
        denominator = np.linalg.norm(embedding) * np.linalg.norm(self.reference)
        if not np.isfinite(denominator) or denominator <= 0:
            raise ValueError("Invalid DMFC embedding or reference norm")
        cosine = np.dot(embedding, self.reference) / denominator
        return float(1 - np.clip(cosine, 0, 1))


class MN3:
    def __init__(self, directory, threads):
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        self.session = ort.InferenceSession(str(directory / "mn3.onnx"), sess_options=options,
                                            providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def score(self, image):
        pixels = (resize_cv(image, 128, cubic=True) - MN3_MEAN) / MN3_STD
        output = self.session.run(None, {self.input: chw(pixels)})[0]
        if output.shape != (1, 2) or not np.isclose(output.sum(), 1, atol=1e-4):
            raise ValueError("Unexpected output from anti-spoof-mn3")
        return float(output[0, 1])


class Mini(TorchModel):
    def __init__(self, directory, device):
        # prepare_assets checks the SHA-256 of this pinned upstream module.
        spec = importlib.util.spec_from_file_location("minifasnet_upstream",
                                                      directory / "minifasnet_upstream.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        super().__init__(module.MiniFASNetV2(conv6_kernel=(5, 5)), directory / "mini.pth", device)

    def score(self, image):
        # Minivision's custom ToTensor casts the BGR byte array to float.
        output = self.forward(resize_cv(image, 80)[:, :, ::-1])
        return float(1 - output.softmax(dim=1)[0, 1].item())


def calibrate(live_scores, target):
    if not live_scores:
        return None
    # For score >= threshold, reject at most floor(target * n) live samples.
    allowed = min(math.floor(target * len(live_scores) + 1e-12), len(live_scores) - 1)
    boundary = sorted(live_scores, reverse=True)[allowed]
    return math.nextafter(boundary, math.inf)


def score_rows(rows, models, crops):
    for row, path in rows:
        if row["split"] == "reference":
            continue
        start = perf_counter()
        status, faces = crops.read(path)
        preprocess_ms = (perf_counter() - start) * 1000
        for name, model in models.items():
            score, model_ms = None, None
            if status == "ok":
                start = perf_counter()
                score = model.score(faces.get(name, faces["mn3"]))
                model_ms = (perf_counter() - start) * 1000
                if not math.isfinite(score) or not -1e-6 <= score <= 1 + 1e-6:
                    raise ValueError(f"Invalid spoof score from {name} for {path}")
                score = min(max(score, 0.0), 1.0)
            yield {**row, "model": name, "status": status, "spoof_score": score,
                   "preprocess_ms": preprocess_ms, "model_ms": model_ms}


def evaluate(predictions, thresholds, *, calibration=False):
    groups = defaultdict(list)
    split = "calibration" if calibration else "test"
    for row in predictions:
        if row["split"] != split or not row["label"]:
            continue
        if row["label"] == "live":
            for scope in (None, row["source_dataset"]):
                groups[row["model"], scope, None, "BPCER"].append(row)
        else:
            for scope in (None, row["source_dataset"]):
                for attack in (None, row["attack_type"]):
                    groups[row["model"], scope, attack, "APCER"].append(row)
    for (model, source, attack, metric), rows in groups.items():
        threshold = thresholds[model]
        scored = [row for row in rows if row["spoof_score"] is not None]
        wrong = "spoof" if metric == "BPCER" else "live"
        errors = sum(row["prediction"] == wrong for row in scored) if threshold is not None else None
        yield dict(model=model, source_dataset=source, attack_type=attack, metric=metric,
                   errors=errors, scored=len(scored), total=len(rows),
                   rate=errors / len(scored) if errors is not None and scored else None,
                   coverage=len(scored) / len(rows), threshold=threshold)


def aggregate_videos(predictions, thresholds):
    """A clip receives the median spoof score of its scorable sampled frames."""
    groups = defaultdict(list)
    for row in predictions:
        identifier = row["video_id"] or row["image_path"]
        groups[row["model"], row["split"], row["source_dataset"],
               bool(row["video_id"]), identifier].append(row)
    for rows in groups.values():
        first = rows[0]
        if any((row["label"], row["attack_type"]) != (first["label"], first["attack_type"])
               for row in rows):
            raise ValueError(f"Inconsistent labels within video {first['video_id']}")
        scores = [row["spoof_score"] for row in rows if row["spoof_score"] is not None]
        score = float(np.median(scores)) if scores else None
        threshold = thresholds.get(first["model"])
        prediction = classify(score, threshold)
        yield {**first, "image_path": "" if first["video_id"] else first["image_path"],
               "status": "ok" if scores else "unscored", "spoof_score": score,
               "prediction": prediction, "threshold": threshold,
               "preprocess_ms": None, "model_ms": None,
               "frames_scored": len(scores), "frames_total": len(rows)}


def classify(score, threshold):
    if score is None or threshold is None:
        return ""
    return "spoof" if score >= threshold else "live"


def fit_thresholds(predictions, models, target):
    scores = defaultdict(list)
    for row in predictions:
        if (row["split"] == "calibration" and row["label"] == "live"
                and row["spoof_score"] is not None):
            scores[row["model"]].append(row["spoof_score"])
    return ({name: calibrate(scores[name], target) for name in models},
            {name: len(scores[name]) for name in models})


def write_csv(path, rows, columns):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def run(args):
    rows = read_manifest(args.manifest)
    prepare_assets(args.weights_dir, args.models, download=False)
    extra = set(args.models).intersection(RESEARCH_MODELS)
    paths = {}
    if extra:
        from research_models import flip_mcl_model, gac_model, mvp_model

        for name in sorted(extra):
            path = getattr(args, name.replace("-", "_") + "_checkpoint")
            if path is None or not path.is_file():
                raise ValueError(f"Provide --{name}-checkpoint with an official FAS checkpoint")
            paths[name] = path
        if "mvp" in extra and (args.mvp_repo is None or args.clip_weights is None
                              or not args.clip_weights.is_file()):
            raise ValueError("MVP-FAS requires --mvp-repo and --clip-weights")
        if "mvp" in extra and sha256(args.clip_weights) != CLIP_SHA256:
            raise ValueError("Expected the original OpenAI CLIP ViT-B/16 weights")
        if "flip-mcl" in extra and args.flip_repo is None:
            raise ValueError("FLIP-MCL requires --flip-repo")
    if args.input_mode == "raw" and not args.detector_weights:
        raise ValueError("Raw mode requires --detector-weights with a YuNet ONNX file")
    if args.input_mode == "cropped" and args.detector_weights:
        raise ValueError("--detector-weights is only used in raw mode")
    if "dmfc" in args.models and not any(row["split"] == "reference" for row, _ in rows):
        raise ValueError("DMFC-FAS needs a separate set of reference live images")

    if any(name != "mn3" for name in args.models):
        import torch

        torch.set_num_threads(args.threads)
        if args.device == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but torch.cuda.is_available() is false")
    constructors = {
        "dmfc": lambda: DMFC(args.weights_dir, args.device, args.tta),
        "mn3": lambda: MN3(args.weights_dir, args.threads),
        "mini": lambda: Mini(args.weights_dir, args.device),
    }
    if extra:
        constructors.update({
            "gac": lambda: gac_model(paths["gac"], args.device),
            "mvp": lambda: mvp_model(args.mvp_repo, paths["mvp"], args.clip_weights, args.device),
            "flip-mcl": lambda: flip_mcl_model(args.flip_repo, paths["flip-mcl"], args.device),
        })
    crops = FaceCrops(args.detector_weights)
    models = {name: constructors[name]() for name in args.models}
    reference_images = []
    skipped_reference = []
    if "dmfc" in models:
        features = []
        for row, path in rows:
            if row["split"] != "reference":
                continue
            status, faces = crops.read(path)
            if status == "ok":
                features.append(models["dmfc"].embedding(faces["dmfc"]))
                reference_images.append(row["image_path"])
            else:
                skipped_reference.append({"image_path": row["image_path"], "status": status})
        if not features:
            raise ValueError("No scorable DMFC reference images")
        models["dmfc"].reference = np.mean(features, axis=0)

    predictions = list(score_rows(rows, models, crops))
    thresholds, counts = fit_thresholds(predictions, models, args.bpcer)
    for row in predictions:
        threshold, score = thresholds[row["model"]], row["spoof_score"]
        row["threshold"] = threshold
        row["prediction"] = classify(score, threshold)
    initial_clips = list(aggregate_videos(predictions, {}))
    clip_thresholds, clip_counts = fit_thresholds(initial_clips, models, args.bpcer)
    for row in initial_clips:
        threshold = clip_thresholds[row["model"]]
        row["threshold"] = threshold
        row["prediction"] = classify(row["spoof_score"], threshold)
    clip_predictions = initial_clips

    args.output.mkdir(parents=True, exist_ok=True)
    write_csv(args.output / "predictions.csv", predictions, PREDICTION_FIELDS)
    write_csv(args.output / "metrics.csv", evaluate(predictions, thresholds), METRIC_FIELDS)
    write_csv(args.output / "video_predictions.csv", clip_predictions,
              PREDICTION_FIELDS + ("frames_scored", "frames_total"))
    write_csv(args.output / "video_metrics.csv", evaluate(clip_predictions, clip_thresholds),
              METRIC_FIELDS)
    write_csv(args.output / "calibration_metrics.csv",
              evaluate(predictions, thresholds, calibration=True), METRIC_FIELDS)
    if "dmfc" in models:
        np.save(args.output / "dmfc_reference.npy", models["dmfc"].reference)
    metadata = {
        "models": args.models,
        "manifest_sha256": sha256(args.manifest),
        "dmfc_tta": args.tta,
        "input_mode": args.input_mode,
        "detector_sha256": sha256(args.detector_weights) if args.detector_weights else None,
        "device": {name: "cpu" if name == "mn3" else args.device for name in models},
        "assets_sha256": {asset: ASSETS[asset][1] for name in models
                          for asset in MODEL_ASSETS.get(name, ())},
        "research_checkpoints": {name: {"filename": path.name, "sha256": sha256(path)}
                                 for name, path in paths.items()},
        "reference_images": reference_images,
        "skipped_reference": skipped_reference,
        "target_bpcer": args.bpcer,
        "thresholds": thresholds,
        "calibration_live_count": counts,
        "video_thresholds": clip_thresholds,
        "video_calibration_live_count": clip_counts,
        "video_aggregation": "median of scorable frames; image used when video_id is empty",
    }
    (args.output / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(predictions)} prediction rows to {args.output}")
    if None in thresholds.values():
        print("At least one model has no live calibration scores. Predictions are score-only for that model.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--models", action="append", choices=MODEL_NAMES,
                        help="Select a model; repeat this option to select several")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("fetch", help="Download and verify official model assets")
    runner = subcommands.add_parser("run", help="Run inference on an image manifest")
    runner.add_argument("--manifest", type=Path, required=True)
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--input-mode", choices=["cropped", "raw"], required=True)
    runner.add_argument("--detector-weights", type=Path)
    runner.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    runner.add_argument("--threads", type=int, default=2)
    runner.add_argument("--tta", action=argparse.BooleanOptionalAction, default=True,
                        help="Average original and flipped DMFC embeddings")
    runner.add_argument("--bpcer", type=float, default=0.05)
    runner.add_argument("--gac-checkpoint", type=Path)
    runner.add_argument("--mvp-checkpoint", type=Path)
    runner.add_argument("--mvp-repo", type=Path)
    runner.add_argument("--clip-weights", type=Path)
    runner.add_argument("--flip-mcl-checkpoint", type=Path)
    runner.add_argument("--flip-repo", type=Path)
    args = parser.parse_args()
    args.models = list(dict.fromkeys(args.models or MODEL_ASSETS))
    if args.command == "run" and (args.threads < 1 or not 0 <= args.bpcer < 1):
        parser.error("--threads must be positive and --bpcer must be in [0, 1)")
    try:
        if args.command == "fetch":
            if not set(args.models).intersection(MODEL_ASSETS):
                raise ValueError("See README for research checkpoint downloads")
            prepare_assets(args.weights_dir, args.models, download=True)
        else:
            run(args)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    main()

"""Optional FAS adapters for author-provided research checkpoints."""

import importlib
import pickle
import sys

import numpy as np


MVP_REV = "fad83b94a326cec1925b73b43fb27708f1bdcf23"
FLIP_REV = "4f95def259e135a0cbaff1d770f559ca739c4c9f"


def checkpoint_state(path, *, gac=False):
    import torch

    scalar = np._core.multiarray.scalar if hasattr(np, "_core") else np.core.multiarray.scalar
    globals_to_allow = [
        (scalar, "numpy.core.multiarray.scalar"),
        (scalar, "numpy._core.multiarray.scalar"),
        np.dtype, type(np.dtype("float32")), type(np.dtype("float64")),
        type(np.dtype("int32")), type(np.dtype("int64")),
    ]
    if gac:
        from easydict import EasyDict

        globals_to_allow.append(EasyDict)
    try:
        with torch.serialization.safe_globals(globals_to_allow):
            data = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError(f"Cannot safely load {path}. Use a compatible tensor-only state dict.") from error
    if not isinstance(data, dict):
        raise ValueError(f"Expected a state dict in {path}")
    state = data.get("state_dict", data)
    if not isinstance(state, dict) or not state or not all(torch.is_tensor(v) for v in state.values()):
        raise ValueError(f"Invalid model state dict in {path}")
    return {key.removeprefix("module."): value for key, value in state.items()}


def import_upstream(directory, module_name, revision):
    import subprocess

    directory = directory.resolve()
    try:
        actual = subprocess.check_output(
            ["git", "-C", str(directory), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=all"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError(f"Expected an upstream Git checkout at {directory}") from error
    if actual != revision or dirty:
        raise ValueError(f"Use a clean checkout of {revision} at {directory}")
    sys.path.insert(0, str(directory))
    write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        return importlib.import_module(module_name)
    finally:
        sys.dont_write_bytecode = write_bytecode
        sys.path.remove(str(directory))


class ResearchScorer:
    def __init__(self, network, device, size, *, resize_tensor=False):
        import torch
        from torchvision import transforms

        self.torch = torch
        self.network = network.eval().float().to(device)
        self.device = device
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        resize = transforms.Resize((size, size), antialias=True)
        to_tensor = transforms.ToTensor()
        stages = [to_tensor, normalize, resize] if resize_tensor else [resize, to_tensor, normalize]
        self.preprocess = transforms.Compose(stages)

    def score(self, image):
        with self.torch.inference_mode():
            tensor = self.preprocess(image).unsqueeze(0).to(self.device)
            logits = self.network(tensor)
            if logits.shape != (1, 2):
                raise ValueError(f"Expected two ordered logits [spoof, live], got {logits.shape}")
            return logits.softmax(dim=-1)[0, 0].item()


def gac_model(path, device):
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torchvision.models import resnet18

    state = checkpoint_state(path, gac=True)
    head = state.get("fc.weight")
    if head is None or tuple(head.shape) not in {(1, 512), (2, 512)}:
        raise ValueError("GAC-FAS checkpoint has no supported normalized head")
    backbone = resnet18(weights=None)
    backbone_state = {k.removeprefix("model."): v for k, v in state.items() if k.startswith("model.")}
    backbone.load_state_dict(backbone_state, strict=True)
    backbone.fc = nn.Identity()

    class GACHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.register_buffer("weight", head)

        def forward(self, image):
            logits = F.linear(F.normalize(self.backbone(image), dim=1),
                              F.normalize(self.weight, dim=1))
            if logits.shape[1] == 1:
                # BCE variant: upstream evaluates the raw live logit. This mapping
                # preserves ranking and the benchmark calibrates its own threshold.
                return torch.cat([torch.zeros_like(logits), logits], dim=1)
            return logits

    return ResearchScorer(GACHead(), device, 256)


def mvp_model(directory, path, clip_weights, device):
    import torch
    from torch import nn

    upstream = import_upstream(directory, "models.MVP_FAS", MVP_REV)
    state = checkpoint_state(path)
    # The upstream constructor builds the matching CLIP architecture from base weights.
    # The official FAS checkpoint then replaces every parameter with strict loading.
    from unittest.mock import patch

    clip_load = upstream.clip.load
    with patch.object(upstream.clip, "load",
                      lambda _name, **kwargs: clip_load(str(clip_weights), device="cpu", **kwargs)):
        model = upstream.mspt(None)
    model.load_state_dict(state, strict=True)

    class MVPInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model.eval().float().to(device)
            tokens = upstream.clip.tokenize(upstream.spoof_templates + upstream.real_templates)
            with torch.inference_mode():
                text_features = self.model.model.encode_text(tokens.to(device))
            self.register_buffer("text_features", text_features)

        def forward(self, image):
            cls, _, patches = self.model.model.encode_image(image)
            cls = self.model.cls_projection(cls)
            patches = self.model.patch_projection(patches)
            slots = self.model.MVSlot(patches + cls.unsqueeze(1), self.text_features)
            return self.model.classifier(self.model.slot_projection(slots.mean(dim=1)))

    return ResearchScorer(MVPInference(), device, 224, resize_tensor=True)


def flip_mcl_model(directory, path, device):
    import torch
    from torch import nn
    from torch.nn import functional as F

    clip = import_upstream(directory, "clip.clip", FLIP_REV)
    prompts = import_upstream(directory, "prompt_templates", FLIP_REV)
    state = checkpoint_state(path)
    if not any(key.startswith("image_mlp.") for key in state):
        raise ValueError("Expected FLIP-MCL checkpoint with its training MLP weights")
    clip_state = {key.removeprefix("model."): value for key, value in state.items()
                  if key.startswith("model.")}
    if "visual.proj" not in clip_state or clip_state["visual.conv1.weight"].shape[-1] != 16:
        raise ValueError("Expected the ViT-B/16 FLIP-MCL checkpoint")
    model = clip.build_model(clip_state).float().eval().to(device)

    class FLIPInference(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = model
            with torch.inference_mode():
                centers = [model.encode_text(clip.tokenize(templates).to(device)).mean(0)
                           for templates in (prompts.spoof_templates, prompts.real_templates)]
                self.register_buffer("text_features", F.normalize(torch.stack(centers), dim=-1))

        def forward(self, image):
            features = F.normalize(self.model.encode_image(image), dim=-1)
            return self.model.logit_scale.exp() * features @ self.text_features.T

    return ResearchScorer(FLIPInference(), device, 224)

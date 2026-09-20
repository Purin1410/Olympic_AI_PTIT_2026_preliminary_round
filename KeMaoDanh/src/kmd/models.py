"""Image scorers derived from the immutable experiment trainer.

BatchNorm remains in evaluation mode, including during full fine-tuning.
Image logits are pooled before the right-minus-left pair score is formed.
"""
from pathlib import Path
import random
import urllib.parse
import numpy as np
import torch
from torch import nn
from torchvision import models

from .config import Config
from .data import Pairs, make_loader
from .core import pair_table, sha256

WEIGHTS = {
    "densenet121": models.DenseNet121_Weights.IMAGENET1K_V1,
    "efficientnet_b0": models.EfficientNet_B0_Weights.IMAGENET1K_V1,
    "efficientnet_b2": models.EfficientNet_B2_Weights.IMAGENET1K_V1,
    "resnet18": models.ResNet18_Weights.IMAGENET1K_V1,
    "resnet34": models.ResNet34_Weights.IMAGENET1K_V1,
}


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def model_for(c, device="cpu", pretrained=True, embedding=False):
    if c.backbone not in WEIGHTS:
        raise ValueError("Backbone outside the recorded ImageNet whitelist.")
    seed_all(c.seed)
    weights = WEIGHTS[c.backbone] if pretrained else None
    model = getattr(models, c.backbone)(weights=weights)
    weight_hash = None
    if weights is not None:
        path = Path(torch.hub.get_dir()) / "checkpoints" / Path(urllib.parse.urlparse(weights.url).path).name
        weight_hash = sha256(path)
        if not weight_hash.startswith(path.stem.rsplit("-", 1)[1]):
            raise ValueError("ImageNet pretrained weight hash mismatch.")
    if c.backbone.startswith("resnet"):
        dim = model.fc.in_features
        model.fc = nn.Identity() if embedding else nn.Linear(dim, 1)
    else:
        dim = model.classifier.in_features if c.backbone == "densenet121" else model.classifier[-1].in_features
        model.classifier = nn.Identity() if embedding else nn.Linear(dim, 1)
    return model.to(device), weight_hash


def head_name(c):
    return "fc." if c.backbone.startswith("resnet") else "classifier."


def train_mode(model, c, epoch):
    model.train(); head = head_name(c); unfreeze = epoch > c.warmup
    for name, parameter in model.named_parameters():
        enabled = name.startswith(head)
        if unfreeze and c.mode == "full":
            enabled = True
        elif unfreeze and c.mode == "partial":
            if c.backbone == "densenet121":
                enabled |= name.startswith(("features.denseblock4.", "features.norm5."))
            elif c.backbone.startswith("resnet"):
                enabled |= name.startswith("layer4.")
            else:
                enabled |= name.startswith(("features.7.", "features.8."))
        parameter.requires_grad_(enabled)
    for layer in model.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()
    assert all(not layer.training for layer in model.modules() if isinstance(layer, nn.modules.batchnorm._BatchNorm))


def logits(model, x, c):
    batch, images, views, channels, height, width = x.shape
    scores = model(x.reshape(batch * images * views, channels, height, width)).reshape(batch, images, views)
    return scores.topk(min(2, views), dim=2).values.mean(2) if c.pooling == "top2" else scores.mean(2)


def objective(scores, y, c):
    image = nn.functional.binary_cross_entropy_with_logits(scores, torch.stack([1-y, y], dim=1))
    pair = nn.functional.binary_cross_entropy_with_logits(scores[:, 1] - scores[:, 0], y)
    if c.objective == "image":
        return image
    if c.objective == "pair":
        return pair
    if c.objective == "mixed":
        return image + .25 * pair
    raise ValueError(c.objective)


@torch.inference_mode()
def predict_pairs(model, frame, c, data_root, norm=None, device="cpu"):
    device = torch.device(device); model.eval(); scores = []
    for x, _ in make_loader(Pairs(frame, data_root, c, norm=norm)):
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=c.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()):
            s = logits(model, x.to(device, non_blocking=True), c)
        scores.append(s.float().cpu().numpy())
    return pair_table(frame, np.concatenate(scores))



@torch.inference_mode()
def embedding_pairs(model, frame, c, data_root, device="cpu"):
    model.eval(); result = []
    device = torch.device(device)
    for tensor, _ in make_loader(Pairs(frame, data_root, c)):
        b, n, v, ch, h, w = tensor.shape
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=c.amp and device.type == "cuda" and torch.cuda.is_bf16_supported()):
            emb = model(tensor.to(device).reshape(b*n*v, ch, h, w)).reshape(b, n, v, -1).mean(2)
        result.append(emb.float().cpu().numpy())
    return np.concatenate(result)

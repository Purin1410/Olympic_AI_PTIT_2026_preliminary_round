"""Small model factories and explicit head/backbone optimizer groups."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from torch import nn
from torchvision import models as tv_models


_BUILDERS: dict[str, tuple[Callable[..., nn.Module], Any]] = {
    "resnet34": (tv_models.resnet34, tv_models.ResNet34_Weights),
    "resnet18": (tv_models.resnet18, tv_models.ResNet18_Weights),
}


def attach_optimizer_groups(model: nn.Module, head: nn.Module) -> nn.Module:
    """Mark the classifier parameters so AdamW can use a separate learning rate."""
    head_ids = {id(parameter) for parameter in head.parameters()}
    for parameter in model.parameters():
        parameter._ailaai_group = "head" if id(parameter) in head_ids else "backbone"  # type: ignore[attr-defined]
    return model


def model_factory(
    backbone: str = "resnet34",
    weights: str | None = "IMAGENET1K_V1",
    initialize: bool = True,
) -> nn.Module:
    """Build a two-class model, optionally starting from ImageNet weights."""
    if backbone not in _BUILDERS:
        raise ValueError(f"Unsupported backbone {backbone!r}; choose one of {sorted(_BUILDERS)}.")
    builder, enum = _BUILDERS[backbone]
    selected = enum[weights] if initialize and weights else None
    model = builder(weights=selected)
    if not hasattr(model, "fc") or not isinstance(model.fc, nn.Linear):
        raise ValueError(f"The {backbone} builder does not expose the expected linear fc head.")
    model.fc = nn.Linear(model.fc.in_features, 2)
    return attach_optimizer_groups(model, model.fc)


def optimizer_parameter_groups(model: nn.Module, backbone_lr: float, head_lr: float) -> list[dict[str, Any]]:
    """Return nonempty, disjoint optimizer groups for backbone and classifier."""
    groups = {"backbone": [], "head": []}
    for parameter in model.parameters():
        if parameter.requires_grad:
            group = getattr(parameter, "_ailaai_group", "backbone")
            if group not in groups:
                raise ValueError(f"Unknown optimizer parameter group {group!r}.")
            groups[group].append(parameter)
    if not groups["head"] or not groups["backbone"]:
        raise ValueError("Both classifier and backbone optimizer groups must contain parameters.")
    return [{"params": groups["backbone"], "lr": backbone_lr}, {"params": groups["head"], "lr": head_lr}]

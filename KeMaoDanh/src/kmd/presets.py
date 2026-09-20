"""Canonical experiment presets derived from the verified job configurations.

Preserves the 5 original preset names (frozen, center60, native, resampled, b2)
while registering descriptive names and aliases for all 23 ORIGINAL_JOBS configurations.
"""
from dataclasses import asdict, replace
from typing import Dict, Any, Optional

from .config import Config

# Base configurations extracted from ORIGINAL_JOBS.json (all 23 original candidate configs)
_BASE_CONFIGS: Dict[str, Dict[str, Any]] = {
    # 1. center60 / A_full_batch24 (matches configs/center60.json: microbatch 24, effective_batch 24, epochs 24)
    "center60": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 24, "patience": 6, "warmup": 2,
        "microbatch": 24, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 2. center60_micro8 / A_full_micro8 (microbatch 8, effective_batch 24, epochs 24)
    "center60_micro8": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 24, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 3. frozen / A_frozen_head
    "frozen": {
        "backbone": "densenet121", "mode": "frozen", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 24, "patience": 6, "warmup": 2,
        "microbatch": 24, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 4. frozen_lr / A_frozen_LR
    "frozen_lr": {
        "backbone": "densenet121", "mode": "frozen", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 24, "patience": 6, "warmup": 2,
        "microbatch": 24, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "ml", "feature": "embedding", "classifier": "lr",
    },
    # 5. center60_cap48 / B_cap48 (microbatch 8, effective_batch 24, epochs 48)
    "center60_cap48": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 6. partial_native / C_partial_native
    "partial_native": {
        "backbone": "densenet121", "mode": "partial", "view": "native", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 7. partial_resampled / C_partial_resampled
    "partial_resampled": {
        "backbone": "densenet121", "mode": "partial", "view": "resampled", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 8. native / C_full_native
    "native": {
        "backbone": "densenet121", "mode": "full", "view": "native", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 9. resampled / C_full_resampled
    "resampled": {
        "backbone": "densenet121", "mode": "full", "view": "resampled", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 10. loss_image / D_image_fixed
    "loss_image": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 11. loss_pair / D_pair
    "loss_pair": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 224,
        "objective": "pair", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 12. loss_mixed / D_mixed
    "loss_mixed": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 224,
        "objective": "mixed", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 13. repair / D_repair
    "repair": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 224,
        "objective": "mixed", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": True, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 14. dense288 / E_dense288
    "dense288": {
        "backbone": "densenet121", "mode": "full", "view": "center60", "size": 288,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 15. b2 / E_b2_288
    "b2": {
        "backbone": "efficientnet_b2", "mode": "full", "view": "center60", "size": 288,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 24. center72 / E_center72 (B_cap48 except view)
    "center72": {
        "backbone": "densenet121", "mode": "full", "view": "center72", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 25. b0_224 / E_efficientnet_b0_224 (B_cap48 except backbone)
    "b0_224": {
        "backbone": "efficientnet_b0", "mode": "full", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 26. b2_224 / E_efficientnet_b2_224 (B_cap48 except backbone)
    "b2_224": {
        "backbone": "efficientnet_b2", "mode": "full", "view": "center60", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 16. resnet18_rgb / F_resnet18_rgb
    "resnet18_rgb": {
        "backbone": "resnet18", "mode": "full", "view": "native", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "train",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 17. resnet18_gaussian / F_resnet18_gaussian
    "resnet18_gaussian": {
        "backbone": "resnet18", "mode": "full", "view": "native", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "gaussian", "normalization": "train",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 18. resnet18_npr / F_resnet18_npr
    "resnet18_npr": {
        "backbone": "resnet18", "mode": "full", "view": "native", "size": 224,
        "objective": "image", "pooling": "mean", "transform": "npr", "normalization": "train",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 19. top2 / G_top2
    "top2": {
        "backbone": "densenet121", "mode": "full", "view": "native", "size": 224,
        "objective": "image", "pooling": "top2", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 19, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 20. b2_resize / G_augmentation
    "b2_resize": {
        "backbone": "efficientnet_b2", "mode": "full", "view": "center60", "size": 288,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "resize", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 0, "fixed_epochs": False, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 21. data_25 / H_data_25
    "data_25": {
        "backbone": "efficientnet_b2", "mode": "full", "view": "center60", "size": 288,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 0.25,
        "fixed_updates": 391, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 22. data_50 / H_data_50
    "data_50": {
        "backbone": "efficientnet_b2", "mode": "full", "view": "center60", "size": 288,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 0.5,
        "fixed_updates": 391, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
    # 23. data_100 / H_data_100
    "data_100": {
        "backbone": "efficientnet_b2", "mode": "full", "view": "center60", "size": 288,
        "objective": "image", "pooling": "mean", "transform": "rgb", "normalization": "imagenet",
        "repair": False, "augmentation": "light", "epochs": 48, "patience": 6, "warmup": 2,
        "microbatch": 8, "effective_batch": 24, "workers": 4, "backbone_lr": 2e-05,
        "head_lr": 0.0003, "weight_decay": 0.0001, "amp": True, "train_fraction": 1.0,
        "fixed_updates": 391, "fixed_epochs": True, "kind": "deep", "feature": "rich", "classifier": "lr",
    },
}

# Aliases mapping alternative names / ORIGINAL_JOBS keys to canonical preset names
PRESET_ALIASES: Dict[str, str] = {
    # Original jobs mappings
    "A_full_batch24": "center60",
    "center60_batch24": "center60",
    "A_full_micro8": "center60_micro8",
    "A_frozen_head": "frozen",
    "A_frozen_LR": "frozen_lr",
    "B_cap48": "center60_cap48",
    "C_partial_native": "partial_native",
    "C_partial_resampled": "partial_resampled",
    "C_full_native": "native",
    "C_full_resampled": "resampled",
    "D_image_fixed": "loss_image",
    "D_pair": "loss_pair",
    "D_mixed": "loss_mixed",
    "D_repair": "repair",
    "E_dense288": "dense288",
    "E_center72": "center72",
    "E_efficientnet_b0_224": "b0_224",
    "E_efficientnet_b2_224": "b2_224",
    "E_b2_288": "b2",
    "F_resnet18_rgb": "resnet18_rgb",
    "F_resnet18_gaussian": "resnet18_gaussian",
    "F_resnet18_npr": "resnet18_npr",
    "G_top2": "top2",
    "G_augmentation": "b2_resize",
    "H_data_25": "data_25",
    "H_data_50": "data_50",
    "H_data_100": "data_100",

    # Convenient shortcuts
    "r18_rgb": "resnet18_rgb",
    "r18_gaussian": "resnet18_gaussian",
    "r18_npr": "resnet18_npr",
    "d_image": "loss_image",
    "d_pair": "loss_pair",
    "d_mixed": "loss_mixed",
    "d_repair": "repair",
    "densenet288": "dense288",
    "b2_288": "b2",
    "resize": "b2_resize",
    "native_top2": "top2",
    "fraction_25": "data_25",
    "fraction_50": "data_50",
    "fraction_100": "data_100",
}


def canonical_preset_name(name: str) -> str:
    """Resolve an alias or original job name to its canonical preset identifier."""
    cleaned = str(name).strip()
    return PRESET_ALIASES.get(cleaned, cleaned)


def is_registered_preset(name: str) -> bool:
    """Check if a preset name or alias is recognized."""
    return canonical_preset_name(name) in _BASE_CONFIGS


def list_registered_presets() -> list[str]:
    """Return sorted list of all canonical registered preset names."""
    return sorted(_BASE_CONFIGS.keys())


def get_preset_config(
    name: str,
    seed: Optional[int] = None,
    fold: Optional[int] = None,
    smoke: bool = False,
) -> Config:
    """Construct an immutable Config from a registered preset, with optional seed, fold and smoke adjustments."""
    canonical = canonical_preset_name(name)
    if canonical not in _BASE_CONFIGS:
        available = ", ".join(sorted(_BASE_CONFIGS.keys()))
        raise ValueError(f"Unknown preset: '{name}'. Available presets: {available}")

    params = dict(_BASE_CONFIGS[canonical])
    if seed is not None:
        params["seed"] = int(seed)
    if fold is not None:
        params["fold"] = int(fold)

    if smoke:
        params["workers"] = 0
        if params.get("train_fraction", 1.0) < 1.0 or params.get("fixed_updates", 0) > 0:
            params["fixed_updates"] = 4
            params["fixed_epochs"] = True
            params["warmup"] = 0
            params["epochs"] = 2
        else:
            params["epochs"] = 2
            params["warmup"] = 1
            params["patience"] = 2
    else:
        params["workers"] = params.get("workers", 4)

    return Config(**params)

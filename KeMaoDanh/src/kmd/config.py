"""Experiment configuration extracted from the verified job snapshot."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Config:
    backbone:str='densenet121'
    mode:str='full'
    view:str='center60'
    size:int=224
    objective:str='image'
    pooling:str='mean'
    transform:str='rgb'
    normalization:str='imagenet'
    repair:bool=False
    augmentation:str='light'
    seed:int=20260917
    fold:int=0
    epochs:int=24
    patience:int=6
    warmup:int=2
    microbatch:int=24
    effective_batch:int=24
    workers:int=4
    backbone_lr:float=2e-5
    head_lr:float=3e-4
    weight_decay:float=1e-4
    amp:bool=True
    train_fraction:float=1.
    fixed_updates:int=0
    fixed_epochs:bool=False
    kind:str='deep'
    feature:str='rich'
    classifier:str='lr'

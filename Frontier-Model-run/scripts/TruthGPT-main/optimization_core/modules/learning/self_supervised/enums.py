"""
Self-Supervised Learning Enums
==============================

Enums for SSL methods, pretext tasks, and loss types.
"""
from enum import Enum

class SSLMethod(Enum):
    """Self-supervised learning methods"""
    SIMCLR = "simclr"
    MOCo = "moco"
    SWAV = "swav"
    BYOL = "byol"
    DINO = "dino"
    Barlow_TWINS = "barlow_twins"
    VICREG = "vicreg"
    MAE = "mae"
    BEIT = "beit"
    MASKED_AUTOENCODER = "masked_autoencoder"

class PretextTaskType(Enum):
    """Pretext task types"""
    CONTRASTIVE_LEARNING = "contrastive_learning"
    RECONSTRUCTION = "reconstruction"
    PREDICTION = "prediction"
    CLUSTERING = "clustering"
    ROTATION_PREDICTION = "rotation_prediction"
    COLORIZATION = "colorization"
    INPAINTING = "inpainting"
    JIGSAW_PUZZLE = "jigsaw_puzzle"
    RELATIVE_POSITIONING = "relative_positioning"
    TEMPORAL_ORDERING = "temporal_ordering"

class ContrastiveLossType(Enum):
    """Contrastive loss types"""
    INFO_NCE = "info_nce"
    NT_XENT = "nt_xent"
    TRIPLET_LOSS = "triplet_loss"
    CONTRASTIVE_LOSS = "contrastive_loss"
    SUPERVISED_CONTRASTIVE = "supervised_contrastive"
    HARD_NEGATIVE_MINING = "hard_negative_mining"


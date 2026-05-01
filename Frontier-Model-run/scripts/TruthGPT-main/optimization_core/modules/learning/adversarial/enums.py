"""
Adversarial Learning Enums
==========================

Enums for attacks, GANs, and defense strategies.
"""
from enum import Enum

class AdversarialAttackType(Enum):
    """Adversarial attack types"""
    FGSM = "fgsm"
    PGD = "pgd"
    C_W = "c_w"
    DEEPFOOL = "deepfool"
    BIM = "bim"
    MIM = "mim"
    JSMA = "jsma"
    CWL2 = "cwl2"
    CWL0 = "cwl0"
    CWLINF = "cwlinf"

class GANType(Enum):
    """GAN types"""
    VANILLA_GAN = "vanilla_gan"
    DCGAN = "dcgan"
    WGAN = "wgan"
    WGAN_GP = "wgan_gp"
    LSGAN = "lsgan"
    BEGAN = "began"
    PROGRESSIVE_GAN = "progressive_gan"
    STYLEGAN = "stylegan"

class DefenseStrategy(Enum):
    """Defense strategies"""
    ADVERSARIAL_TRAINING = "adversarial_training"
    DISTILLATION = "distillation"
    DETECTION = "detection"
    INPUT_TRANSFORMATION = "input_transformation"
    CERTIFIED_DEFENSE = "certified_defense"
    RANDOMIZATION = "randomization"
    ENSEMBLE_DEFENSE = "ensemble_defense"


"""Utility functions for AdaptCLIP."""

import json
import os
import random
import re

import numpy as np
import torch
import torchvision.transforms as transforms

from adaptcliplib.constants import OPENAI_DATASET_MEAN, OPENAI_DATASET_STD
from adaptcliplib.transform import image_transform

IMG_FORMATS = {"bmp", "dng", "jpeg", "jpg", "mpo", "png", "tif", "tiff", "webp", "pfm"}
FORMATS_HELP_MSG = f"Supported formats are:\nimages: {IMG_FORMATS}"


def setup_seed(seed):
    """Set random seed for reproducibility.

    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(pred, max_value=None, min_value=None):
    """Normalize prediction values.

    Args:
        pred: Prediction tensor to normalize
        max_value: Maximum value for normalization (optional)
        min_value: Minimum value for normalization (optional)

    Returns:
        Normalized prediction tensor
    """
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def get_transform(image_size=518):
    """Get image preprocessing and target transforms.

    Args:
        args: Arguments containing image_size and other parameters

    Returns:
        tuple: (preprocess, target_transform) transforms
    """
    preprocess = image_transform(image_size, is_train=False, mean=OPENAI_DATASET_MEAN, std=OPENAI_DATASET_STD)

    target_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor()
    ])
    preprocess.transforms[0] = transforms.Resize(
        size=(image_size, image_size),
        interpolation=transforms.InterpolationMode.BICUBIC,
        max_size=None,
        antialias=None
    )
    preprocess.transforms[1] = transforms.CenterCrop(size=(image_size, image_size))
    return preprocess, target_transform


def get_position(file_name, dataset_name):
    """Extract position information from filename.

    Args:
        file_name: Input filename
        dataset_name: Name of the dataset

    Returns:
        str or None: Position string if found, None otherwise
    """
    if dataset_name == 'nmj':
        return file_name.split('_')[2]
    elif dataset_name == 'real-iad':
        pattern = r'_C\d+_'
        matches = re.findall(pattern, file_name)
        return matches[0][1: -1]
    else:
        return None

"""Torch-oriented interfaces for `utils.py`."""

# Original Code
# https://github.com/jpcbertoldo/aupimo
#
# Modified

import logging

import torch

logger = logging.getLogger(__name__)


def images_classes_from_masks(masks: torch.Tensor) -> torch.Tensor:
    """Deduce the image classes from the masks."""
    return (masks == 1).any(axis=(1, 2)).to(torch.int32)

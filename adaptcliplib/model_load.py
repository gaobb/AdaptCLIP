import hashlib
import os
import urllib
import warnings
from typing import Dict, Iterable, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from pkg_resources import packaging
from torchvision.transforms import (Compose, InterpolationMode, Normalize,
                                    Resize, ToTensor)
from tqdm import tqdm

from .build_model import build_model
from .simple_tokenizer import SimpleTokenizer as _Tokenizer

if packaging.version.parse(torch.__version__) < packaging.version.parse("1.7.1"):
    warnings.warn("PyTorch version 1.7.1 or higher is recommended")

from functools import partial

try:
    import safetensors.torch
    _has_safetensors = True
except ImportError:
    _has_safetensors = False

try:
    from huggingface_hub import hf_hub_download
    hf_hub_download = partial(hf_hub_download, library_name="open_clip")
    _has_hf_hub = True
except ImportError:
    hf_hub_download = None
    _has_hf_hub = False

__all__ = ["available_models", "load",
        ]  
# "get_similarity_map",  "compute_similarity","compute_norm_similarity", "calculate_visual_anomaly_score"

_tokenizer = _Tokenizer()

OPENAI_DATASET_MEAN = (0.48145466, 0.4578275, 0.40821073)
OPENAI_DATASET_STD = (0.26862954, 0.26130258, 0.27577711)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INCEPTION_MEAN = (0.5, 0.5, 0.5)
INCEPTION_STD = (0.5, 0.5, 0.5)

# Default name for a weights file hosted on the Huggingface Hub.
HF_WEIGHTS_NAME = "open_clip_pytorch_model.bin"  # default pytorch pkl
HF_SAFE_WEIGHTS_NAME = "open_clip_model.safetensors"  # safetensors version
HF_CONFIG_NAME = 'open_clip_config.json'


def _apcfg(url='', hf_hub='', **kwargs):
    # CLIPA defaults
    return {
        'url': url,
        'hf_hub': hf_hub,
        'mean': IMAGENET_MEAN,
        'std': IMAGENET_STD,
        'interpolation': 'bilinear',
        'resize_mode': 'squash',
        **kwargs,
    }

_MODELS = {
    "VITB16_PLUS_240": "https://github.com/mlfoundations/open_clip/releases/download/v0.2-weights/vit_b_16_plus_240-laion400m_e31-8fb26589.pt",
    "ViT-L/14@336px": "https://openaipublic.azureedge.net/clip/models/3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02/ViT-L-14-336px.pt",
    "ViT-L-14-CLIPA-336": 'UCSC-VLAA/ViT-L-14-CLIPA-336-datacomp1B',
}


def _download(
        url: str,
        cache_dir: Union[str, None] = None,
):

    if not cache_dir:
        cache_dir = os.path.expanduser("~/.cache/clip")
        #cache_dir = os.path.expanduser("/apdcephfs/private_jiangtyan/.cache/clip")
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.basename(url)

    if 'openaipublic' in url:
        expected_sha256 = url.split("/")[-2]
    elif 'mlfoundations' in url:
        expected_sha256 = os.path.splitext(filename)[0].split("-")[-1]
    else:
        expected_sha256 = ''

    download_target = os.path.join(cache_dir, filename)

    if os.path.exists(download_target) and not os.path.isfile(download_target):
        raise RuntimeError(f"{download_target} exists and is not a regular file")

    if os.path.isfile(download_target):
        if expected_sha256:
            if hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
                return download_target
            else:
                warnings.warn(f"{download_target} exists, but the SHA256 checksum does not match; re-downloading the file")
        else:
            return download_target

    with urllib.request.urlopen(url) as source, open(download_target, "wb") as output:
        with tqdm(total=int(source.headers.get("Content-Length")), ncols=80, unit='iB', unit_scale=True) as loop:
            while True:
                buffer = source.read(8192)
                if not buffer:
                    break

                output.write(buffer)
                loop.update(len(buffer))

    if expected_sha256 and not hashlib.sha256(open(download_target, "rb").read()).hexdigest().startswith(expected_sha256):
        raise RuntimeError(f"Model has been downloaded but the SHA256 checksum does not not match")

    return download_target



def has_hf_hub(necessary=False):
    if not _has_hf_hub and necessary:
        # if no HF Hub module installed, and it is necessary to continue, raise error
        raise RuntimeError(
            'Hugging Face hub model specified but package not installed. Run `pip install huggingface_hub`.')
    return _has_hf_hub


def _get_safe_alternatives(filename: str) -> Iterable[str]:
    """Returns potential safetensors alternatives for a given filename.

    Use case:
        When downloading a model from the Huggingface Hub, we first look if a .safetensors file exists and if yes, we use it.
    """
    if filename == HF_WEIGHTS_NAME:
        yield HF_SAFE_WEIGHTS_NAME

    if filename not in (HF_WEIGHTS_NAME,) and (filename.endswith(".bin") or filename.endswith(".pth")):
        yield filename[:-4] + ".safetensors"


def download_pretrained_from_hf(
        model_id: str,
        filename: Optional[str] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
):
    #has_hf_hub(True)

    filename = filename or HF_WEIGHTS_NAME

    # Look for .safetensors alternatives and load from it if it exists
    if _has_safetensors:
        for safe_filename in _get_safe_alternatives(filename):
            try:
                cached_file = hf_hub_download(
                    repo_id=model_id,
                    filename=safe_filename,
                    revision=revision,
                    cache_dir=cache_dir,
                )
                return cached_file
            except Exception:
                pass

    try:
        # Attempt to download the file
        cached_file = hf_hub_download(
            repo_id=model_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir,
        )
        return cached_file  # Return the path to the downloaded file if successful
    except Exception as e:
        raise FileNotFoundError(f"Failed to download file ({filename}) for {model_id}. Last error: {e}")



def _convert_image_to_rgb(image):
    return image.convert("RGB")


def _transform(n_px):
    return Compose([
        Resize((n_px, n_px), interpolation=InterpolationMode.BICUBIC),
        #CenterCrop(n_px), # rm center crop to explain whole image
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


def available_models() -> List[str]:
    """Returns the names of available CLIP models"""
    return list(_MODELS.keys())


def load_state_dict(checkpoint_path: str, map_location='cpu'):
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    if next(iter(state_dict.items()))[0].startswith('module'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}
    return state_dict


def load_checkpoint(model, checkpoint_path, strict=True):
    state_dict = load_state_dict(checkpoint_path)
    # detect old format and make compatible with new format
    if 'positional_embedding' in state_dict and not hasattr(model, 'positional_embedding'):
        state_dict = convert_to_custom_text_state_dict(state_dict)
    resize_pos_embed(state_dict, model)
    incompatible_keys = model.load_state_dict(state_dict, strict=strict)
    return incompatible_keys


def load(name: str, device: Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu", design_details = None, jit: bool = False, download_root: str = None):
    """Load a CLIP model

    Parameters
    ----------
    name : str
        A model name listed by `clip.available_models()`, or the path to a model checkpoint containing the state_dict

    device : Union[str, torch.device]
        The device to put the loaded model

    jit : bool
        Whether to load the optimized JIT model or more hackable non-JIT model (default).

    download_root: str
        path to download the model files; by default, it uses "~/.cache/clip"

    Returns
    -------
    model : torch.nn.Module
        The CLIP model

    preprocess : Callable[[PIL.Image], torch.Tensor]
        A torchvision transform that converts a PIL image into a tensor that the returned model can take as its input
    """
    print("name", name)
    if name in _MODELS:
        if name == "ViT-L-14-CLIPA-336":
            model_path = download_pretrained_from_hf(_MODELS[name])
        else:
            model_path = _download(_MODELS[name], download_root or os.path.expanduser("~/.cache/clip"))
        #model_path = _download(_MODELS[name], download_root or os.path.expanduser("/apdcephfs/private_jiangtyan/.cache/clip"))
    elif os.path.isfile(name):
        model_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {available_models()}")


    if name == 'VITB16_PLUS_240':
        state_dict = torch.load(model_path, map_location="cpu")
    elif name == 'ViT-L-14-CLIPA-336':
        if str(model_path).endswith(".safetensors"):
            from safetensors.torch import load_file
            state_dict = load_file(model_path, device=device)
    else:
        with open(model_path, 'rb') as opened_file:
            try:
                # loading JIT archive
                model = torch.jit.load(opened_file, map_location=device if jit else "cpu").eval()
                state_dict = None
            except RuntimeError:
                # loading saved state dict
                if jit:
                    warnings.warn(f"File {model_path} is not a JIT archive. Loading as a state dict instead")
                    jit = False
                state_dict = torch.load(opened_file, map_location="cpu")

    if not jit:
        model = build_model(name, state_dict or model.state_dict(), design_details).to(device)
        if str(device) == "cpu":
            model.float()
        return model, _transform(model.visual.input_resolution)

    # patch the device names
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device(device)), example_inputs=[])
    device_node = [n for n in device_holder.graph.findAllNodes("prim::Constant") if "Device" in repr(n)][-1]

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []

        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)

        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(node["value"]).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    # patch dtype to float32 on CPU
    if str(device) == "cpu":
        float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
        float_input = list(float_holder.graph.findNode("aten::to").inputs())[1]
        float_node = float_input.node()

        def patch_float(module):
            try:
                graphs = [module.graph] if hasattr(module, "graph") else []
            except RuntimeError:
                graphs = []

            if hasattr(module, "forward1"):
                graphs.append(module.forward1.graph)

            for graph in graphs:
                for node in graph.findAllNodes("aten::to"):
                    inputs = list(node.inputs())
                    for i in [1, 2]:  # dtype can be the second or third argument to aten::to()
                        if inputs[i].node()["value"] == 5:
                            inputs[i].node().copyAttributes(float_node)

        model.apply(patch_float)
        patch_float(model.encode_image)
        patch_float(model.encode_text)

        model.float()

    return model, _transform(model.input_resolution.item())

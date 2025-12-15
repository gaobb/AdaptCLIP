from collections import OrderedDict
from typing import Union, List, Tuple, Union
from pkg_resources import packaging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()


class VVAttention_Block(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim//4, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim//4, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

        self.ln_1 = LayerNorm(dim)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(dim, dim // 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(dim // 4, dim))
        ]))
        self.ln_2 = LayerNorm(dim)

    def forward(self, x):
        x_ori = x
        x = self.ln_1(x)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 1, self.num_heads, C // self.num_heads // 4).permute(2, 0, 3, 1, 4)
        v = qkv[0]

        # replace k & q by v
        k = v
        q = v

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C//4)
        x = self.proj_drop(self.proj(x))
        x = x + x_ori
        x = x + self.mlp(self.ln_2(x))
        return x


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


# implement attention module for v-v self-attention
class Attention(nn.Module):
    def __init__(self, out_dim, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., settings=''):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.settings = settings

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # original self-attention for the original path
        attn_ori = (q @ k.transpose(-2, -1)) * self.scale
        attn_ori = attn_ori.softmax(dim=-1)
        attn_ori = self.attn_drop(attn_ori)

        # replace k & q by v
        k = v
        q = k

        # self-attention, higher temperate for resnets performs better
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = (attn).softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_ori = (attn_ori @ v).transpose(1, 2).reshape(B, N, C)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        x_ori = self.proj_drop(self.proj(x_ori))
        return [x, x_ori]



class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x, whole = False, ffn = False):
        # print("xxxxx",x.shape)
        # dual paths for blocks deeper than "d"

        if isinstance(self.attn, Attention):
            if isinstance(x, list):
                if not ffn:
                    x, x_ori = x
                    x_res = self.attention(self.ln_1(x_ori))
                    x_res, x_ori_res = x_res
                    x_ori += x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res  # skip ffn for the new path
                    return [x, x_ori]
                else:
                    x, x_ori_1 = x
                    x_res = self.attention(self.ln_1(x_ori_1))
                    x_res, x_ori_res = x_res
                    x_ori = x_ori_1 +  x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res  # skip ffn for the new path
                    x = x_res + x_ori_1
                    x = x + self.mlp(self.ln_2(x))
                    return [x, x_ori]
            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = x + self.attention(self.ln_1(x))
            x = x + self.mlp(self.ln_2(x))
        return x


class ResidualAttentionBlock_learnable_token(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None, design_details=None,
            text_layer=False, i = 0):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

        self.i = i
        #self.compound_prompt_nctx = design_details['learnabel_text_embedding_length']
        self.text_layer = text_layer
        if i == 0:
            self.first_layer = True
        else:
            self.first_layer = False

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        if isinstance(self.attn, Attention):
            x = x.transpose(0, 1)
            x, x_ori = self.attn(x)
            return [x.transpose(0, 1), x_ori.transpose(0, 1)]
        else:
            return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, inputs):

        # dual paths for blocks deeper than "d"
        if isinstance(self.attn, Attention):
            x = inputs[0]
            if isinstance(x, list):
                x, x_ori = x
                x_res = self.attention(self.ln_1(x_ori))
                x_res, x_ori_res = x_res
                x_ori += x_ori_res
                x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                x += x_res  # skip ffn for the new path
                return [x, x_ori]

            # start of dual path
            else:
                x_res = self.attention(self.ln_1(x))
                if isinstance(x_res, list):
                    x_res, x_ori_res = x_res
                    x_ori = x + x_ori_res
                    x_ori = x_ori + self.mlp(self.ln_2(x_ori))
                    x += x_res
                    return [x, x_ori]

        # singl path before "d"
        else:
            x = inputs + self.attention(self.ln_1(inputs))
            x = x + self.mlp(self.ln_2(x))
            return x
        #     x = inputs[0]
        #     compound_prompts_deeper = inputs[1]
        #     counter = inputs[2]
        #     if not self.first_layer:
        #         # First check if the ith layer needs compound prompts or not
        #         if not (counter > len(compound_prompts_deeper) - 1):
        #             # Appending the learnable tokens in different way
        #             # x -> [77, NCLS, DIM]
        #             # First remove the learnable tokens from previous layer
        #             prefix = x[:1, :, :]
        #             suffix = x[1 + self.compound_prompt_nctx:, :, :]
        #             textual_context = compound_prompts_deeper[counter]
        #             textual_context = textual_context.expand(x.shape[1], -1, -1).permute(1, 0, 2).half()
        #             # Add the learnable tokens of this layer with the input, replaced by previous
        #             # layer learnable tokens
        #             x = torch.cat([prefix, textual_context, suffix], dim=0)
        #             # Once done, update the counter, so that the next time, it does not use same learnable tokens
        #             counter += 1
        #     x = x + self.attention(self.ln_1(x))
        #     x = x + self.mlp(self.ln_2(x))
        # return [x, compound_prompts_deeper, counter]


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None, need_weights: bool = False, design_details = None , text_layer = False):
        super().__init__()
        self.width = width
        self.layers = layers
        self.text_layer = text_layer
        self.design_deatails = design_details
        if self.text_layer:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock_learnable_token(width, heads, attn_mask, design_details, text_layer, i=i) for i in range(layers)])
        else:
            self.resblocks = nn.ModuleList([ResidualAttentionBlock(width, heads, attn_mask,) for i in range(layers)])

    def ori_CLIP_with_patch_forward(self, x, out_layers):
        idx = 0
        out_tokens = []
        for r in self.resblocks:
            idx += 1
            x = r(x)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[1])
                else:
                    out_tokens.append(x)

        return [x, x], out_tokens

    def ADCLIP_forward(self, x, out_layers, ffn):
        idx = 0
        out_tokens = []

        for r in self.resblocks:
            idx += 1
            x = r(x, ffn = ffn)
            if idx in out_layers:
                if isinstance(x, list):
                    out_tokens.append(x[0].clone())  # clone?
                else:
                    out_tokens.append(x)
        return x, out_tokens

    def forward(self, x: torch.Tensor, out_layers = None, DPAM_layer = None, ffn = False):
        # visual encoder forward
        if out_layers is None:
            out_layers = [6, 12, 18, 24]
        if not self.text_layer:
            out_tokens = []

            if DPAM_layer is None:
                [x, x], out_tokens = self.ori_CLIP_with_patch_forward(x, out_layers)
                return [x, x], out_tokens
            else:
                x, out_tokens = self.ADCLIP_forward(x, out_layers, ffn)
                return x, out_tokens
        # text encoder forward
        # ori text embedding
        else:
            for idx, r in enumerate(self.resblocks):
                x = r(x)
            return x
        # # insert learnable text embedding
        # elif self.design_deatails is not None:
        #     for idx, r in enumerate(self.resblocks):
        #         x = r(x)
        #     return x[0]
    def get_cast_dtype(self) -> torch.dtype:
        return self.resblocks[0].mlp.c_fc.weight.dtype


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads, need_weights=True)
        self.attn = None
        self.embed_dim = width
        self.num_heads = heads

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))


    @torch.no_grad()
    def DAPM_replace(self, DPAM_layer):
        if DPAM_layer is not None:
            for i in range(1, DPAM_layer):
                self.attn = Attention(self.embed_dim, self.embed_dim, self.num_heads, True)
                self.attn.qkv.weight.data = self.transformer.resblocks[-i].attn.in_proj_weight.clone()
                self.attn.qkv.bias.data = self.transformer.resblocks[-i].attn.in_proj_bias.clone()
                self.attn.proj.weight.data = self.transformer.resblocks[-i].attn.out_proj.weight.clone()
                self.attn.proj.bias.data = self.transformer.resblocks[-i].attn.out_proj.bias.clone()
                self.transformer.resblocks[-i].attn = self.attn

    @torch.no_grad()
    def forward(self, x: torch.Tensor, features_list, ori_patch = False, proj_use = True, DPAM_layer = None, ffn = False):

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        side = int((self.positional_embedding.shape[0] - 1) ** 0.5)
        new_side = int((x.shape[1] - 1) ** 0.5)

        # update the position embedding during inference for varied input size
        if side != new_side:
            new_pos = self.positional_embedding[1:, :].reshape(-1, side, side, x.shape[-1]).permute(0, 3, 1, 2)
            new_pos = torch.nn.functional.interpolate(new_pos, (new_side, new_side), mode='bilinear')
            new_pos = new_pos.reshape(-1, x.shape[-1], new_side * new_side).transpose(1, 2)
            self.positional_embedding.data = torch.cat([self.positional_embedding[:1, :], new_pos[0]], 0)

        pos = self.positional_embedding.to(x.dtype)
        x = x + pos
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        [x, x_ori], patch_tokens = self.transformer(x, features_list, DPAM_layer = DPAM_layer, ffn = ffn)

        if True:
            patch_token_list = []
            for patch_token in patch_tokens:
                patch_token = self.ln_post(patch_token.permute(1, 0, 2)) @ self.proj  # LND -> NLD
                patch_token_list.append(patch_token)
            patch_tokens = patch_token_list

            '''
            ori_patch_token_list = []
            for patch_token in ori_patch_tokens:
                patch_token = self.ln_post(patch_token.permute(1, 0, 2)) @ self.proj  # LND -> NLD
                ori_patch_token_list.append(patch_token)
            ori_patch_tokens = ori_patch_token_list
            '''
            return x_ori[0, :, :] @ self.proj, patch_tokens


        return x


class AdaptCLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 design_details = None
                 ):
        super().__init__()

        self.context_length = context_length

        if isinstance(vision_layers, (tuple, list)):
            vision_heads = vision_width * 32 // 64
            self.visual = ModifiedResNet(
                layers=vision_layers,
                output_dim=embed_dim,
                heads=vision_heads,
                input_resolution=image_resolution,
                width=vision_width
            )
        else:
            vision_heads = vision_width // 64
            self.visual = VisionTransformer(  # vision branch
                input_resolution=image_resolution,
                patch_size=vision_patch_size,
                width=vision_width,
                layers=vision_layers,
                heads=vision_heads,
                output_dim=embed_dim
            )

        self.transformer = Transformer(  # text branch
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(), text_layer=True, design_details=design_details
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image, feature_list = None, ori_patch = False, proj_use = True, DPAM_layer = None, ffn = False):
        return self.visual(image.type(self.dtype), feature_list, ori_patch = ori_patch, proj_use = proj_use, DPAM_layer = DPAM_layer, ffn = ffn)


    def encode_text(self, text, tokenized_prompts = None):
        cast_dtype = self.transformer.get_cast_dtype()
        if tokenized_prompts is None:
            x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        else:
            x = text
        x = x + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection if tokenized_prompts is not None else x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x

    def encode_text_learn(self, prompts, tokenized_prompts, deep_compound_prompts_text = None, normalize: bool = False):
        cast_dtype = self.transformer.get_cast_dtype()
        x = prompts + self.positional_embedding.to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND (77, 2, 768)

        if deep_compound_prompts_text is None:
            x = self.transformer(x)
        else:
            x = self.transformer([x, deep_compound_prompts_text, 0])
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection
        return x

    def forward(self, image, text):
        image_features = self.encode_image(image)
        text_features = self.encode_text(text)

        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_image, logits_per_text



def harmonic_mean(tensor_list):
    # 将列表中的张量堆叠成一个四维张量
    stacked_tensors = torch.stack(tensor_list)  # shape: (N, B, C, H, W)

    # 计算每个元素的倒数
    reciprocal_tensors = 1.0 / stacked_tensors  # shape: (N, B, C, H, W)

    # 计算倒数的和
    reciprocal_sum = torch.sum(reciprocal_tensors, dim=0)  # shape: (B, C, H, W)

    # 计算调和平均
    n = stacked_tensors.size(0)  # N: 张量的数量
    h_mean = n / reciprocal_sum  # shape: (B, C, H, W)

    return h_mean


def average_mean(tensor_list):
    stacked_tensors = torch.stack(tensor_list)  # shape: (N, B, C, H, W)

    a_mean = stacked_tensors.mean(dim=0)  # shape: (B, C, H, W)

    return a_mean


def fusion_fun(tensor_list, fusion_type = 'harmonic_mean'):
    if fusion_type == 'harmonic_mean':
        mean_tensor = harmonic_mean(tensor_list)
    elif fusion_type == 'average_mean':
        mean_tensor = average_mean(tensor_list)
    else:
        print("please correct fusion function~")

    return mean_tensor


def get_similarity_map(sm, shape):
    side = int(sm.shape[1] ** 0.5)
    sm = sm.reshape(sm.shape[0], side, side, -1).permute(0, 3, 1, 2)
    sm = torch.nn.functional.interpolate(sm, shape, mode='bilinear')
    sm = sm.permute(0, 2, 3, 1)
    return sm


def compute_similarity(image_features, text_features, t=2):
    prob_1 = image_features[:, :1, :] @ text_features.t()
    b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * text_features.reshape(1, 1, n_t, c)
    similarity = feats.sum(-1)
    return (similarity/0.07).softmax(-1), prob_1


def calculate_visual_anomaly_score(visual_features, feature_gallery1, grid_size=(37, 37)):
    N = visual_features.shape[0]

    score1, _ = (1.0 - visual_features @ feature_gallery1.transpose(-1, -2)).min(dim=-1)
    score1 /= 2.0

    score = torch.zeros((N, grid_size[0] * grid_size[1] + 1)) + score1.cpu()

    return score.unsqueeze(-1)


def compute_norm_similarity(image_features, disc_features):
    b, n_t, n_i, c = image_features.shape[0], disc_features.shape[0], image_features.shape[1], image_features.shape[2]
    feats = image_features.reshape(b, n_i, 1, c) * disc_features.reshape(b, 1, n_t, c)
    similarity = feats.sum(-1)
    return similarity


class ResMLP(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(ResMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.BatchNorm1d(c_in // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
        )

    def forward(self, x):
        if x.dim() == 2:
            out = self.fc(x)
        else:
            batch_size, seq_len, feature_dim = x.shape
            x = x.view(batch_size * seq_len, feature_dim)
            out = self.fc(x)
            out = out.view(batch_size, seq_len, feature_dim)
            x = x.view(batch_size, seq_len, feature_dim)
        x = x + out
        return x


class VisualAdapter(nn.Module):
    def __init__(self, img_size, patch_size, input_dim=768, reduction=4, layers_num=1):
        super().__init__()

        self.img_size = img_size
        self.patch_size =  patch_size
        self.input_dim = input_dim
        self.layers_num = layers_num
        self.reduction = reduction

        self.local_adater = ResMLP(input_dim, reduction=reduction)
        self.global_adapter = ResMLP(input_dim, reduction=reduction)

    def forward(self, image_features, patch_features, static_text_features):

        static_text_features = torch.stack(torch.chunk(static_text_features, dim = 0, chunks = 2), dim = 1)  # ?
        static_text_features = static_text_features/static_text_features.norm(dim=-1, keepdim=True)

        image_feature = self.global_adapter(image_features)
        patch_feature = self.local_adater(patch_features[-1])  # use the last stage

        image_feature = image_feature / image_feature.norm(dim = -1, keepdim = True)
        patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)

        static_text_probs = image_feature.unsqueeze(1) @ static_text_features.permute(0, 2, 1)
        static_text_probs = static_text_probs[:, 0, ...]/0.07

        similarity, _ = compute_similarity(patch_feature, static_text_features[0])
        similarity_map = get_similarity_map(similarity[:, 1:, :], self.img_size).permute(0, 3, 1, 2)

        return static_text_probs, similarity_map


class PQAdapter(nn.Module):
    def __init__(self, img_size, patch_size, context=True, input_dim=768, mid_dim=128, layers_num=4):
        super().__init__()
        self.img_size = img_size
        self.patch_size =  patch_size
        self.input_dim = input_dim
        self.layers_num = layers_num
        self.mid_dim = mid_dim
        self.context = context

        self.sharebn = nn.ModuleList([nn.BatchNorm2d(input_dim) for i in range(layers_num)])

        local_adapter = nn.Sequential(
            nn.Conv2d(input_dim, mid_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_dim, mid_dim, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(mid_dim, mid_dim//2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(mid_dim//2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(mid_dim//2, mid_dim//2, kernel_size=2, stride=2, padding=0),
            nn.Conv2d(mid_dim//2, 2, kernel_size=1, stride=1, padding=0)
        )

        global_adapter = nn.Sequential(
            nn.Linear(input_dim, mid_dim, bias=False),
            nn.BatchNorm1d(mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, mid_dim//2, bias=False),
            nn.BatchNorm1d(mid_dim//2),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim//2, 2, bias=False)
        )


        self.local_adapter = nn.ModuleList([local_adapter for i in range(layers_num)])
        self.global_adapter = nn.ModuleList([global_adapter for i in range(layers_num)])


    def forward(self, query_feats, query_patch_feats, prompt_feats, prompt_patch_feats):
        global_logits, local_scores, align_scores = [], [], []
        # patch-level
        for lay_idx, query_patch_feat in enumerate(query_patch_feats):
            prompt_patch_feat = prompt_patch_feats[lay_idx]
            query_patch_feat = F.normalize(query_patch_feat, dim = -1)  # BLD
            prompt_patch_feat = F.normalize(prompt_patch_feat, dim = -1)  # BLD ——> BSLD

            b, dim = query_patch_feat.shape[0], query_patch_feat.shape[-1]

            # patch-level
            query_patch_feat = query_patch_feat[:, 1:, :]  # B*L*D
            #prompt_patch_feat = prompt_patch_feat[:, 1:, :]  # B*L*D note assume shot=1
            prompt_patch_feat = prompt_patch_feat[:, :, 1:, :]  # B*L*D note assume shot >= 1
            prompt_patch_feat = prompt_patch_feat.reshape(b, -1, dim)

            align_score, min_idx = torch.min(1.0 - torch.bmm(query_patch_feat, prompt_patch_feat.permute(0, 2, 1)), dim = -1)
            align_score = align_score.reshape(b, 1, self.img_size//self.patch_size, self.img_size//self.patch_size)
            align_score = F.interpolate(align_score, size=(self.img_size, self.img_size), mode='bilinear')
            align_scores.append(align_score)

            align_prompt_feat = prompt_patch_feat[torch.arange(min_idx.size(0)).unsqueeze(1), min_idx]

            query_patch_feat = query_patch_feat.permute(0, 2, 1).reshape(b, dim, self.img_size//self.patch_size, self.img_size//self.patch_size)
            align_prompt_feat = align_prompt_feat.permute(0, 2, 1).reshape(b, dim, self.img_size//self.patch_size, self.img_size//self.patch_size)

            if self.context:
                fusion_patch_feat = query_patch_feat + torch.abs(query_patch_feat - align_prompt_feat)
            else:
                fusion_patch_feat = torch.abs(query_patch_feat - align_prompt_feat)

            fusion_patch_feat = self.sharebn[lay_idx](fusion_patch_feat)  # bach normalization

            local_score = self.local_adapter[lay_idx](fusion_patch_feat)
            local_score = local_score.softmax(dim=1)

            local_score = F.interpolate(local_score, size=(self.img_size, self.img_size), mode='bilinear')
            local_scores.append(local_score)

            # image-level
            fusion_img_feat = fusion_patch_feat.view(b, dim, -1)
            fusion_img_feat =  (fusion_img_feat.mean(dim=-1) + fusion_img_feat.topk(10, dim=-1)[0].mean(dim=-1))/2.0
            global_logit = self.global_adapter[lay_idx](fusion_img_feat)

            global_logits.append(global_logit)

        return global_logits, local_scores, align_scores


def tokenize(texts: Union[str, List[str]], context_length: int = 77, truncate: bool = False) -> Union[torch.IntTensor, torch.LongTensor]:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length].
    We return LongTensor when torch version is <1.8.0, since older index_select requires indices to be long.
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token] for text in texts]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.int)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {texts[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


class TextualAdapter(nn.Module):
    def __init__(self, clip_model, img_size, prompt_length):
        super().__init__()

        self.img_size = img_size
        self.n_ctx = prompt_length
        n_ctx_pos = self.n_ctx
        n_ctx_neg = self.n_ctx

        dtype = clip_model.transformer.get_cast_dtype()
        ctx_dim = clip_model.ln_final.weight.shape[0]

        self.learn_normal_list = [
            "{}",
        ]

        self.learn_anomaly_list = [
            "{}",
        ]

        self.static_normal_list = [
            "{}",
            "flawless {}",
            "perfect {}",
            "unblemished {}",
            "{} without flaw",
            "{} without defect",
            "{} without damage"
        ]

        self.static_anomaly_list = [
            "damaged {}",
            "{} with flaw",
            "{} with defect",
            "{} with damage"
        ]

        self.template_list = [
        "a cropped photo of the {}.",
        "a close-up photo of a {}.",
        "a close-up photo of the {}.",
        "a bright photo of a {}.",
        "a bright photo of the {}.",
        "a dark photo of the {}.",
        "a dark photo of a {}.",
        "a jpeg corrupted photo of the {}.",
        "a jpeg corrupted photo of the {}.",
        "a blurry photo of the {}.",
        "a blurry photo of a {}.",
        "a photo of a {}.",
        "a photo of the {}.",
        "a photo of a small {}.",
        "a photo of the small {}.",
        "a photo of a large {}.",
        "a photo of the large {}.",
        "a photo of the {} for visual inspection.",
        "a photo of a {} for visual inspection.",
        "a photo of the {} for anomaly detection.",
        "a photo of a {} for anomaly detection."
        ]

        normal_num = len(self.learn_normal_list)
        anormaly_num = len(self.learn_anomaly_list)

        # Random Initialization
        print("Initializing class-specific contexts")
        # n_ctx_pos代表learnable token的长度，ctx_dim表示prompt的dimension
        ctx_vectors_pos = torch.empty(1, 1, n_ctx_pos, ctx_dim, dtype=dtype)
        ctx_vectors_neg = torch.empty(1, 1, n_ctx_neg, ctx_dim, dtype=dtype)
        nn.init.normal_(ctx_vectors_pos, std=0.02)
        nn.init.normal_(ctx_vectors_neg, std=0.02)
        prompt_prefix_pos = " ".join(["X"] * n_ctx_pos)
        prompt_prefix_neg = " ".join(["X"] * n_ctx_neg)

        self.ctx_pos = nn.Parameter(ctx_vectors_pos)  # to be optimized
        self.ctx_neg = nn.Parameter(ctx_vectors_neg)  # to be optimized

        prompts_pos = [prompt_prefix_pos + "."]
        prompts_neg = [prompt_prefix_neg + "."]

        tokenized_prompts_pos = []
        tokenized_prompts_neg = []

        for p_pos in prompts_pos:
            tokenized_prompts_pos.append(tokenize(p_pos))
        for p_neg in prompts_neg:
            tokenized_prompts_neg.append(tokenize(p_neg))
        tokenized_prompts_pos = torch.cat(tokenized_prompts_pos)
        tokenized_prompts_neg = torch.cat(tokenized_prompts_neg)
        #生成相应的text embedding
        with torch.no_grad():
            embedding_pos = clip_model.token_embedding(tokenized_prompts_pos).type(dtype)
            embedding_neg = clip_model.token_embedding(tokenized_prompts_neg).type(dtype)
            n, l, d = embedding_pos.shape
            print("embedding_pos", embedding_pos.shape)
            embedding_pos = embedding_pos.reshape(normal_num, 1, l, d).permute(1, 0, 2, 3)
            embedding_neg = embedding_neg.reshape(anormaly_num, 1, l, d).permute(1, 0, 2, 3)

        self.register_buffer("token_prefix_pos", embedding_pos[:, :, :1, :] )
        self.register_buffer("token_suffix_pos", embedding_pos[:, :, 1 + n_ctx_pos:, :])
        self.register_buffer("token_prefix_neg", embedding_neg[:, :, :1, :])
        self.register_buffer("token_suffix_neg", embedding_neg[:, :, 1 + n_ctx_neg:, :])

        n, d = tokenized_prompts_pos.shape
        tokenized_prompts_pos = tokenized_prompts_pos.reshape(normal_num, 1, d).permute(1, 0, 2)

        n, d = tokenized_prompts_neg.shape
        tokenized_prompts_neg = tokenized_prompts_neg.reshape(anormaly_num, 1, d).permute(1, 0, 2)

        self.n_ctx_pos = n_ctx_pos
        self.n_ctx_neg = n_ctx_neg

        self.register_buffer("tokenized_prompts_pos", tokenized_prompts_pos)
        self.register_buffer("tokenized_prompts_neg", tokenized_prompts_neg)
        print("tokenized_prompts shape", self.tokenized_prompts_pos.shape, self.tokenized_prompts_neg.shape)

    def forward(self):
        ctx_pos = self.ctx_pos
        ctx_neg = self.ctx_neg

        prefix_pos = self.token_prefix_pos
        prefix_neg = self.token_prefix_neg
        suffix_pos = self.token_suffix_pos
        suffix_neg = self.token_suffix_neg

        prompts_pos = torch.cat(
            [
                # N(the number of template), 1, dim
                prefix_pos,  # (n_cls, 1, dim)
                ctx_pos,  # (n_cls, n_ctx, dim)
                suffix_pos,  # (n_cls, *, dim)
            ],
            dim=2,
        )

        prompts_neg = torch.cat(
            [
                prefix_neg,  # (n_cls, 1, dim)
                ctx_neg,  # (n_cls, n_ctx, dim)
                suffix_neg,  # (n_cls, *, dim)
            ],
            dim=2,
        )
        _, _, l, d = prompts_pos.shape
        prompts_pos = prompts_pos.reshape(-1, l, d)
        _, _, l, d = prompts_neg.shape
        prompts_neg = prompts_neg.reshape(-1, l, d)
        prompts = torch.cat([prompts_pos, prompts_neg], dim=0)


        _, l, d = self.tokenized_prompts_pos.shape
        tokenized_prompts_pos = self.tokenized_prompts_pos.reshape(-1,  d)
        _, l, d = self.tokenized_prompts_neg.shape
        tokenized_prompts_neg = self.tokenized_prompts_neg.reshape(-1,  d)
        tokenized_prompts = torch.cat((tokenized_prompts_pos, tokenized_prompts_neg), dim = 0)

        return prompts, tokenized_prompts

    def prompt(self):
        norm_class_state = [ele.format('object') for ele in self.static_normal_list]
        normal_static_template = [class_template.format(ele) for ele in norm_class_state for class_template in self.template_list]
        abnormal_class_state = [ele.format('object') for ele in self.static_anomaly_list]
        anomaly_static_template = [class_template.format(ele) for ele in abnormal_class_state for class_template in self.template_list]

        return normal_static_template, anomaly_static_template

    def prepare_static_text_feature(self, model):
        normal_description, abnormal_description = self.prompt()
        normal_tokens = tokenize(normal_description)
        abnormal_tokens = tokenize(abnormal_description)
        with torch.no_grad():
            normal_text_features = model.encode_text(normal_tokens.cuda()).float()
            abnormal_text_features = model.encode_text(abnormal_tokens.cuda()).float()

        avg_normal_text_features = torch.mean(normal_text_features, dim = 0, keepdim= True)
        avg_abnormal_text_features = torch.mean(abnormal_text_features, dim = 0, keepdim= True)

        self.static_text_features = torch.cat((avg_normal_text_features, avg_abnormal_text_features), dim = 0)  # (2, 768)

    def compute_global_local_score(self, query_feats, query_patch_feats, text_features):

        text_features = torch.stack(torch.chunk(text_features, dim = 0, chunks = 2), dim = 1)   # [1, 2, 768]
        text_features = text_features/text_features.norm(dim=-1, keepdim=True)

        query_feats = query_feats / query_feats.norm(dim=-1, keepdim=True)
        text_probs = query_feats.unsqueeze(1) @ text_features.permute(0, 2, 1)
        text_probs = text_probs[:, 0, ...]/0.07


        # similarity_map_list.append(similarity_map)
        patch_feature = query_patch_feats[-1]  # use the last feature
        patch_feature = patch_feature / patch_feature.norm(dim = -1, keepdim = True)  # [8, 1370, 768]
        # patch_feature [8, 1370, 768], text_features[0] [2, 768]
        similarity, _ = compute_similarity(patch_feature, text_features[0])  # [8, 1370, 2]
        similarity_map = get_similarity_map(similarity[:, 1:, :], self.img_size).permute(0, 3, 1, 2)  # [8, 2, 518, 518]

        return text_probs, similarity_map

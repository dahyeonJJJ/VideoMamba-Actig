# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import os
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
import torch.utils.checkpoint as checkpoint

from einops import rearrange
from timm.models.vision_transformer import _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_

from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights

import math

from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


MODEL_PATH = 'your_model_path'
_MODELS = {
    "videomamba_t16_in1k": os.path.join(MODEL_PATH, "videomamba_t16_in1k_res224.pth"),
    "videomamba_s16_in1k": os.path.join(MODEL_PATH, "videomamba_s16_in1k_res224.pth"),
    "videomamba_m16_in1k": os.path.join(MODEL_PATH, "videomamba_m16_in1k_res224.pth"),
}


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False,drop_path=0.,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None,
        use_checkpoint=False
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (residual + self.drop_path(hidden_states)) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states if residual is None else self.drop_path(hidden_states),
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        if use_checkpoint:
            hidden_states = checkpoint.checkpoint(self.mixer, hidden_states, inference_params)
        else:
            hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.,
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True,
    layer_idx=None,
    bimamba=True,
    device=None,
    dtype=None,
):
    factory_kwargs = {"device": device, "dtype": dtype}
    if ssm_cfg is None:
        ssm_cfg = {}
    mixer_cls = partial(Mamba, layer_idx=layer_idx, bimamba=bimamba, **ssm_cfg, **factory_kwargs)
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        self.proj = nn.Conv3d(
            in_chans, embed_dim, 
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        x = self.proj(x)
        return x
    
class LateralConnection(nn.Module):
    def __init__(self, fast_shape, slow_shape, fast_channels=32, slow_channels=64):
        super(LateralConnection, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(fast_channels, slow_channels, [3, 1, 1], stride=[2, 1, 1], padding=[1,0,0]),   
            nn.BatchNorm3d(slow_channels),
            nn.ReLU(),
        )
        self.fast_shape = fast_shape
        self.slow_shape = slow_shape
        
    def forward(self, slow_path, fast_path):
        B, L, D = fast_path.shape
        fast_path = fast_path.permute(0,2,1).reshape(B, D, self.fast_shape[0], self.fast_shape[1], self.fast_shape[2])
        fast_path = self.conv(fast_path)
        
        B, L, D = slow_path.shape
        slow_path = slow_path.permute(0,2,1).reshape(B, D, self.slow_shape[0], self.slow_shape[1], self.slow_shape[2])
        
        return (fast_path + slow_path).reshape(B, D, L).permute(0,2,1)
    
class Upsample(nn.Module):
    def __init__(self, in_embed_dim, out_embed_dim, fast_shape, slow_shape, scale_factor=(2,1,1)):
        super(Upsample, self).__init__()
        self.upsample = nn.Upsample(scale_factor=scale_factor)
        self.conv = nn.Conv3d(in_embed_dim, out_embed_dim, [3, 1, 1], stride=1, padding=(1,0,0))
        self.fast_shape = fast_shape
        self.slow_shape = slow_shape
        
    def forward(self, slow_path, fast_path):
        B, L, D = slow_path.shape
        slow_path = slow_path.permute(0,2,1).reshape(B, D, self.slow_shape[0], self.slow_shape[1], self.slow_shape[2])
        slow_path = self.upsample(slow_path)
        
        B, L, D = fast_path.shape
        fast_path = fast_path.permute(0,2,1).reshape(B, D, self.fast_shape[0], self.fast_shape[1], self.fast_shape[2])
        
        path = torch.cat([fast_path, slow_path], dim=1)
        path = self.conv(path).reshape(B, -1, L).permute(0,2,1)
        return path

class VisionMamba_slowfast(nn.Module):
    def __init__(
            self, 
            img_size=224, 
            patch_size=16, 
            depth=24, 
            embed_dim=192, 
            channels=3, 
            num_classes=1000,
            drop_rate=0.,
            drop_path_rate=0.1,
            ssm_cfg=None, 
            norm_epsilon=1e-5, 
            initializer_cfg=None,
            fused_add_norm=True,
            rms_norm=True, 
            residual_in_fp32=True,
            bimamba=True,
            # video
            kernel_size=1, 
            num_frames=8, 
            fc_drop_rate=0., 
            device=None,
            dtype=None,
            # checkpoint
            use_checkpoint=False,
            checkpoint_num=0,
        ):
        factory_kwargs = {"device": device, "dtype": dtype} # follow MambaLMHeadModel
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.use_checkpoint = use_checkpoint
        self.checkpoint_num = checkpoint_num
        print(f'Use checkpoint: {use_checkpoint}')
        print(f'Checkpoint number: {checkpoint_num}')

        # pretrain parameters
        self.num_classes = num_classes
        self.d_model = self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=(kernel_size*2),
            in_chans=channels, embed_dim=embed_dim
        )
        self.patch_embed_fast = PatchEmbed(
            img_size=img_size, patch_size=patch_size, 
            kernel_size=kernel_size,
            in_chans=channels, embed_dim=(embed_dim//2)
        )
        num_patches_slow = self.patch_embed.num_patches
        num_patches_fast = self.patch_embed_fast.num_patches
        self.shape_slow = shape_slow = [num_frames // (kernel_size*2), img_size//patch_size, img_size//patch_size]
        self.shape_fast = shape_fast = [num_frames // kernel_size, img_size//patch_size, img_size//patch_size]

        # self.cls_token_slow = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
        # self.cls_token_fast = nn.Parameter(torch.zeros(1, 1, (self.embed_dim//2)))
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches_slow, self.embed_dim))
        self.pos_embed_fast = nn.Parameter(torch.zeros(1, num_patches_fast, (self.embed_dim//2)))
        
        self.temporal_pos_embedding = nn.Parameter(torch.zeros(1, num_frames // (kernel_size*2), embed_dim))
        self.temporal_pos_embedding_fast = nn.Parameter(torch.zeros(1, num_frames // kernel_size, (embed_dim//2)))
        
        self.pos_drop_slow = nn.Dropout(p=drop_rate)
        self.pos_drop_fast = nn.Dropout(p=drop_rate)

        self.head_drop = nn.Dropout(fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        # mamba blocks
        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dim,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.layers_fast = nn.ModuleList(
            [
                create_block(
                    (embed_dim//2),
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    bimamba=bimamba,
                    drop_path=inter_dpr[i],
                    **factory_kwargs,
                )
                for i in range(depth)
            ]
        )
        self.fusions = nn.ModuleList(
            [
                LateralConnection(fast_shape=shape_fast, slow_shape=shape_slow, fast_channels=(embed_dim//2), slow_channels=embed_dim)
                for i in range(depth-1)
            ]
        )
        self.upsample = Upsample(in_embed_dim=embed_dim + (embed_dim//2), out_embed_dim=embed_dim, 
                                 fast_shape=shape_fast, slow_shape=shape_slow, scale_factor=(2,1,1))
        
        # output head
        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(embed_dim, eps=norm_epsilon, **factory_kwargs)
        self.norm_f_fast = (nn.LayerNorm if not rms_norm else RMSNorm)((embed_dim//2), eps=norm_epsilon, **factory_kwargs)

        # original init
        self.apply(segm_init_weights)
        self.head.apply(segm_init_weights)
        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.pos_embed_fast, std=.02)
        trunc_normal_(self.temporal_pos_embedding, std=.02)
        trunc_normal_(self.temporal_pos_embedding_fast, std=.02)
        

        # mamba init
        self.apply(
            partial(
                _init_weights,
                n_layer=depth,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "temporal_pos_embedding", "pos_embed_fast", "temporal_pos_embedding_fast"}
    
    def get_num_layers(self):
        return len(self.layers)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward_features(self, x, inference_params=None):
        # slow path
        x_s = self.patch_embed(x)
        B, C, T, H, W = x_s.shape
        x_s = x_s.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # cls_token = self.cls_token_slow.expand(x_s.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x_s = torch.cat((cls_token, x_s), dim=1)
        x_s = x_s + self.pos_embed

        # temporal pos
        # cls_tokens = x_s[:B, :1, :]
        # x_s = x_s[:, 1:]
        x_s = rearrange(x_s, '(b t) n m -> (b n) t m', b=B, t=T)
        x_s = x_s + self.temporal_pos_embedding
        x_s = rearrange(x_s, '(b n) t m -> b (t n) m', b=B, t=T)
        # x_s = torch.cat((cls_tokens, x_s), dim=1)

        x_s = self.pos_drop_slow(x_s)
        
        
        # fast path
        x_f = self.patch_embed_fast(x)
        B, C, T, H, W = x_f.shape
        x_f = x_f.permute(0, 2, 3, 4, 1).reshape(B * T, H * W, C)

        # cls_token = self.cls_token_fast.expand(x_f.shape[0], -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        # x_f = torch.cat((cls_token, x_f), dim=1)
        x_f = x_f + self.pos_embed_fast

        # temporal pos
        # cls_tokens = x_f[:B, :1, :]
        # x_f = x_f[:, 1:]
        x_f = rearrange(x_f, '(b t) n m -> (b n) t m', b=B, t=T)
        x_f = x_f + self.temporal_pos_embedding_fast
        x_f = rearrange(x_f, '(b n) t m -> b (t n) m', b=B, t=T)
        # x_f = torch.cat((cls_tokens, x_f), dim=1)

        x_f = self.pos_drop_fast(x_f)
        

        # mamba impl
        residual = None
        hidden_states = x_s
        residual_fast = None
        hidden_states_fast = x_f
        
        for idx, layer in enumerate(self.layers):
            if self.use_checkpoint and idx < self.checkpoint_num:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params,
                    use_checkpoint=True
                )
            else:
                hidden_states, residual = layer(
                    hidden_states, residual, inference_params=inference_params
                )
                layer_fast = self.layers_fast[idx]
                hidden_states_fast, residual_fast = layer_fast(
                    hidden_states_fast, residual_fast, inference_params=inference_params
                )
                if idx < self.get_num_layers() - 1:
                    hidden_states = self.fusions[idx](hidden_states, hidden_states_fast)

        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                self.drop_path(hidden_states),
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
            # Set prenorm=False here since we don't need the residual
            fused_add_norm_fn_fast = rms_norm_fn if isinstance(self.norm_f_fast, RMSNorm) else layer_norm_fn
            hidden_states_fast = fused_add_norm_fn_fast(
                self.drop_path(hidden_states_fast),
                self.norm_f_fast.weight,
                self.norm_f_fast.bias,
                eps=self.norm_f_fast.eps,
                residual=residual_fast,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )
        # slow path upsample & slowfast fusion
        hidden_states = self.upsample(hidden_states, hidden_states_fast) 


        # return THW Averaging Token (B, D)
        return hidden_states.mean(dim=1, keepdim=False)

    def forward(self, x, inference_params=None):
        x = self.forward_features(x, inference_params)
        x = self.head(self.head_drop(x))
        return x


def inflate_weight(weight_2d, time_dim, center=True):
    print(f'Init center: {center}')
    if center:
        weight_3d = torch.zeros(*weight_2d.shape)
        weight_3d = weight_3d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        middle_idx = time_dim // 2
        weight_3d[:, :, middle_idx, :, :] = weight_2d
    else:
        weight_3d = weight_2d.unsqueeze(2).repeat(1, 1, time_dim, 1, 1)
        weight_3d = weight_3d / time_dim
    return weight_3d


def load_state_dict(model, state_dict, center=True):
    state_dict_3d = model.state_dict()
    for k in state_dict.keys():
        if k in state_dict_3d.keys() and state_dict[k].shape != state_dict_3d[k].shape:
            if len(state_dict_3d[k].shape) <= 3:
                print(f'Ignore: {k}')
                continue
            print(f'Inflate: {k}, {state_dict[k].shape} => {state_dict_3d[k].shape}')
            time_dim = state_dict_3d[k].shape[2]
            state_dict[k] = inflate_weight(state_dict[k], time_dim, center=center)
    
    del state_dict['head.weight']
    del state_dict['head.bias']
    msg = model.load_state_dict(state_dict, strict=False)
    print(msg)


@register_model
def videomamba_slowfast_tiny(pretrained=False, **kwargs):
    model = VisionMamba_slowfast(
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_t16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_slowfast_small(pretrained=False, **kwargs):
    model = VisionMamba_slowfast(
        patch_size=16, 
        embed_dim=384, 
        depth=24, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_s16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


@register_model
def videomamba_slowfast_middle(pretrained=False, **kwargs):
    model = VisionMamba_slowfast(
        patch_size=16, 
        embed_dim=576, 
        depth=32, 
        rms_norm=True, 
        residual_in_fp32=True, 
        fused_add_norm=True, 
        **kwargs
    )
    model.default_cfg = _cfg()
    if pretrained:
        print('load pretrained weights')
        state_dict = torch.load(_MODELS["videomamba_m16_in1k"], map_location='cpu')
        load_state_dict(model, state_dict, center=True)
    return model


if __name__ == '__main__':
    import time
    from fvcore.nn import FlopCountAnalysis
    from fvcore.nn import flop_count_table
    import numpy as np

    seed = 4217
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    num_frames = 8
    img_size = 224

    # To evaluate GFLOPs, pleaset set `rms_norm=False` and `fused_add_norm=False`
    model = videomamba_middle(num_frames=num_frames).cuda()
    flops = FlopCountAnalysis(model, torch.rand(1, 3, num_frames, img_size, img_size).cuda())
    s = time.time()
    print(flop_count_table(flops, max_depth=1))
    print(time.time()-s)

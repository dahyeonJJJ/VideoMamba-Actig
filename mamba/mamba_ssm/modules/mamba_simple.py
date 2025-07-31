# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_fn_out, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj, mamba_inner_fn_no_out_proj_out
except ImportError:
    selective_scan_fn, selective_scan_fn_out, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj, mamba_inner_fn_no_out_proj_out = None, None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None
    
from .model_utils import shift_size_generate, window_partition, window_expansion, window_reverse, Predictor, local_scan,local_scan_quad,local_scan_quad_quad,local_reverse, local_reverse_quad, local_reverse_quad_quad, Scan_FB, Merge_FB, NONROI_ROI_split, NONROI_ROI_merge, apply_hilbert_curve_2d_quad, reverse_hilbert_curve_2d_quad

from .hilbert_2d import HilbertCurve

class Mamba_pruned2_frame_pre(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        # original
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, T=8):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                # pre-trained ssm out
                out, ssm_out = mamba_inner_fn_no_out_proj_out(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b, ssm_out_b = mamba_inner_fn_no_out_proj_out(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                # token pruning using token importance evaluation (from 'Exploring Token Pruning in Vision State Space Models')
                score_out = torch.relu(ssm_out+ssm_out_b.flip([-1])).mean(dim=1, keepdim=True)  # b, 1, l
                
                # Seperate by frame
                frames = T
                spatial = seqlen//frames
                score_out = rearrange(score_out, "b d (t s) -> b d t s", t=frames, s=spatial)  # b, 1, t, s
                
                # fix prunning ratio
                thres_k = max(1, int(spatial * 1.0))

                _, indices_out = score_out.topk(thres_k, dim=-1, largest=True, sorted=True)  # b, 1, t, k
                
                # post 단계에서 필요한 것: indices_out
                indices=[indices_out]
                
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out, indices

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Mamba_pruned2_frame_post(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        # original
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, indices, inference_params=None, T=8):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                
                # post 단계에서 필요한 것: indices_out
                # Seperate by frame
                frames = T
                spatial = seqlen//frames
                
                indices_out = indices[0]
                xz_tokens = rearrange(xz, "b d (t s) -> b d t s", t=frames, s=spatial)
                xz_fg = xz_tokens.gather(dim=-1, index=indices_out.expand(-1, self.d_inner * 2, -1, -1))  # b, d, t, k

                xz_pruned = rearrange(xz_fg, "b d t s -> b d (t s)")
                
                # actual videomamba scan after token pruning
                out_pruned, ssm_out = mamba_inner_fn_no_out_proj_out(
                    xz_pruned,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_pruned_b, ssm_out_b = mamba_inner_fn_no_out_proj_out(
                    xz_pruned.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out_pruned = torch.cat((out_pruned, ssm_out), dim=1)
                out_pruned_b = torch.cat((out_pruned_b, ssm_out_b), dim=1)
                
                # fill hidden states of pruned tokens with zero.
                out_pruned_tokens = rearrange(out_pruned, "b d (t s) -> b d t s", t=frames, s=spatial)
                out_tmp = torch.zeros_like(out_pruned_tokens)
                out_b_tmp = torch.zeros_like(out_pruned_tokens)
                
                out_fg = out_pruned_tokens
                out_pruned_b_tokens = rearrange(out_pruned_b.flip([-1]), "b d (t s) -> b d t s", t=frames, s=spatial)
                out_fg_b = out_pruned_b_tokens

                out_tmp.scatter_(dim=-1, index=indices_out.expand(-1, self.d_inner*2, -1, -1), src=out_fg)  # b, d, t, k
                out_b_tmp.scatter_(dim=-1, index=indices_out.expand(-1, self.d_inner*2, -1, -1), src=out_fg_b)  # b, d, t, k
                
                ssm_out = rearrange(out_tmp[:, self.d_inner:], "b d t s -> b d (t s)")
                ssm_out_b = rearrange(out_b_tmp[:, self.d_inner:], "b d t s -> b d (t s)")
                out = rearrange(out_tmp[:, :self.d_inner], "b d t s -> b d (t s)")
                out_b = rearrange(out_b_tmp[:, :self.d_inner], "b d t s -> b d (t s)")

                # token pruning using token importance evaluation (from 'Exploring Token Pruning in Vision State Space Models')
                score_out = torch.relu(ssm_out+ssm_out_b).mean(dim=1, keepdim=True)  # b, 1, l
                
                # Seperate by frame
                score_out = rearrange(score_out, "b d (t s) -> b d t s", t=frames, s=spatial)  # b, 1, t, s
                
                # fix prunning ratio
                thres_k = max(1, int(spatial * 1.0))

                _, indices_out = score_out.topk(thres_k, dim=-1, largest=True, sorted=True)  # b, 1, t, k
                
                # post 단계에서 필요한 것: indices_out
                indices=[indices_out]
                
                out = F.linear(rearrange(out + out_b, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out, indices

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Mamba2d(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
        vertical=True,
        token_seq=[8,14,14],
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba
        self.vertical = vertical
        self.token_seq = token_seq

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
         
        # vertical
        if self.vertical:
            self.in_proj_v = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            A_v = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_v_log = torch.log(A_v)  # Keep A_v_log in fp32
            self.A_v_log = nn.Parameter(A_v_log)
            self.A_v_log._no_weight_decay = True 

            self.conv1d_v = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_v = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_v = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_v = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_v._no_weight_decay = True
            
            
            A_bv = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_bv_log = torch.log(A_bv)  # Keep A_bv_log in fp32
            self.A_bv_log = nn.Parameter(A_bv_log)
            self.A_bv_log._no_weight_decay = True 

            self.conv1d_bv = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_bv = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_bv = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_bv = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_bv._no_weight_decay = True
            

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, T=1):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba and self.vertical:
                # hidden_states_v = torch.cat((hidden_states[:, :1, :], 
                #                             rearrange(hidden_states[:, 1:, :], 'b (t h w) d -> b (t w h) d', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])), dim=1)
                hidden_states_v = rearrange(hidden_states, 'b (t h w) d -> b (t w h) d', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])
                
                xz_v = rearrange(
                    self.in_proj_v.weight @ rearrange(hidden_states_v, "b l d -> d (b l)"),
                    "d (b l) -> b d l",
                    l=seqlen,
                )
                if self.in_proj_v.bias is not None:
                    xz_v = xz_v + rearrange(self.in_proj_v.bias.to(dtype=xz_v.dtype), "d -> d 1")
                
                A_b = -torch.exp(self.A_b_log.float())
                A_v = -torch.exp(self.A_v_log.float())
                A_bv = -torch.exp(self.A_bv_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out_v = mamba_inner_fn_no_out_proj(
                    xz_v,
                    self.conv1d_v.weight,
                    self.conv1d_v.bias,
                    self.x_proj_v.weight,
                    self.dt_proj_v.weight,
                    A_v,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D_v.float(),
                    delta_bias=self.dt_proj_v.bias.float(),
                    delta_softplus=True,
                )
                out_bv = mamba_inner_fn_no_out_proj(
                    xz_v.flip([-1]),
                    self.conv1d_bv.weight,
                    self.conv1d_bv.bias,
                    self.x_proj_bv.weight,
                    self.dt_proj_bv.weight,
                    A_bv,
                    None,
                    None,
                    self.D_bv.float(),
                    delta_bias=self.dt_proj_bv.bias.float(),
                    delta_softplus=True,
                )
                out_b = out_b.flip([-1])
                # out_v = torch.cat((out_v[:, :, :1], 
                #                 rearrange(out_v[:, :, 1:], 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])), dim=-1)
                out_v = rearrange(out_v, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])
                out_bv = out_bv.flip([-1])
                # out_bv = torch.cat((out_bv[:, :, :1], 
                #                 rearrange(out_bv[:, :, 1:], 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])), dim=-1)
                out_bv = rearrange(out_bv, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])
                
                out = F.linear(rearrange(out + out_b + out_v + out_bv, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            elif self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

# class Mamba_ti(nn.Module):
#     def __init__(
#         self,
#         d_model,
#         d_state=16,
#         d_conv=4,
#         expand=2,
#         dt_rank="auto",
#         dt_min=0.001,
#         dt_max=0.1,
#         dt_init="random",
#         dt_scale=1.0,
#         dt_init_floor=1e-4,
#         conv_bias=True,
#         bias=False,
#         use_fast_path=True,  # Fused kernel options
#         layer_idx=None,
#         device=None,
#         dtype=None,
#         bimamba=True,
#         tiscan=False,
#         tipred=False,
#         krratio=0.2,
#         token_seq=[8,14,14],
#     ):
#         factory_kwargs = {"device": device, "dtype": dtype}
#         super().__init__()
#         self.d_model = d_model
#         self.d_state = d_state
#         self.d_conv = d_conv
#         self.expand = expand
#         self.d_inner = int(self.expand * self.d_model)
#         self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
#         self.use_fast_path = use_fast_path
#         self.layer_idx = layer_idx
#         self.bimamba = bimamba
#         self.tiscan = tiscan
#         self.tipred = tipred
#         self.krratio = krratio
#         self.token_seq = token_seq

#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

#         self.conv1d = nn.Conv1d(
#             in_channels=self.d_inner,
#             out_channels=self.d_inner,
#             bias=conv_bias,
#             kernel_size=d_conv,
#             groups=self.d_inner,
#             padding=d_conv - 1,
#             **factory_kwargs,
#         )

#         self.activation = "silu"
#         self.act = nn.SiLU()

#         self.x_proj = nn.Linear(
#             self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
#         )
#         self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = self.dt_rank**-0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(self.dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError

#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             self.dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         self.dt_proj.bias._no_reinit = True

#         # S4D real initialization
#         # NOTE: why plus 1?
#         A = repeat(
#             torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=self.d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         self.A_log = nn.Parameter(A_log)
#         self.A_log._no_weight_decay = True

#         # D "skip" parameter
#         self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
#         self.D._no_weight_decay = True

#         # bidirectional
#         # forked from https://github.com/hustvl/Vim
#         if self.bimamba:
#             A_b = repeat(
#                 torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
#                 "n -> d n",
#                 d=self.d_inner,
#             ).contiguous()
#             A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
#             self.A_b_log = nn.Parameter(A_b_log)
#             self.A_b_log._no_weight_decay = True 

#             self.conv1d_b = nn.Conv1d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 groups=self.d_inner,
#                 padding=d_conv - 1,
#                 **factory_kwargs,
#             )

#             self.x_proj_b = nn.Linear(
#                 self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
#             )
#             self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

#             self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
#             self.D_b._no_weight_decay = True
         
#         # tiscan
#         if self.tiscan:
#             self.alphalogit = nn.Parameter(torch.tensor(-3.0, device=device))
#             A_ti = repeat(
#                 torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
#                 "n -> d n",
#                 d=self.d_inner,
#             ).contiguous()
#             A_ti_log = torch.log(A_ti)
#             self.A_ti_log = nn.Parameter(A_ti_log)
#             self.A_ti_log._no_weight_decay = True 

#             self.conv1d_ti = nn.Conv1d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 groups=self.d_inner,
#                 padding=d_conv - 1,
#                 **factory_kwargs,
#             )

#             self.x_proj_ti = nn.Linear(
#                 self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
#             )
#             self.dt_proj_ti = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

#             self.D_ti = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
#             self.D_ti._no_weight_decay = True
            
            
#             A_ti_b = repeat(
#                 torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
#                 "n -> d n",
#                 d=self.d_inner,
#             ).contiguous()
#             A_ti_b_log = torch.log(A_ti_b)
#             self.A_ti_b_log = nn.Parameter(A_ti_b_log)
#             self.A_ti_b_log._no_weight_decay = True 

#             self.conv1d_ti_b = nn.Conv1d(
#                 in_channels=self.d_inner,
#                 out_channels=self.d_inner,
#                 bias=conv_bias,
#                 kernel_size=d_conv,
#                 groups=self.d_inner,
#                 padding=d_conv - 1,
#                 **factory_kwargs,
#             )

#             self.x_proj_ti_b = nn.Linear(
#                 self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
#             )
#             self.dt_proj_ti_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

#             self.D_ti_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
#             self.D_ti_b._no_weight_decay = True
            

#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

#     def forward(self, hidden_states, ind, inference_params=None, T=8):
#         """
#         hidden_states: (B, L, D)
#         Returns: same shape as hidden_states
#         """
#         batch, seqlen, dim = hidden_states.shape
#         frames = self.token_seq[0]
#         spatial = seqlen//frames

#         conv_state, ssm_state = None, None
#         if inference_params is not None:
#             conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
#             if inference_params.seqlen_offset > 0:
#                 # The states are updated inplace
#                 out, _, _ = self.step(hidden_states, conv_state, ssm_state)
#                 return out

#         # We do matmul and transpose BLH -> HBL at the same time
#         # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
#         xz = rearrange(
#             self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
#             "d (b l) -> b d l",
#             l=seqlen,
#         )
#         if self.in_proj.bias is not None:
#             xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

#         A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
#         # In the backward pass we write dx and dz next to each other to avoid torch.cat
#         if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
#             if self.bimamba and self.tipred:
                
#                 A_b = -torch.exp(self.A_b_log.float())
#                 out, so = mamba_inner_fn_no_out_proj_out(
#                     xz,
#                     self.conv1d.weight,
#                     self.conv1d.bias,
#                     self.x_proj.weight,
#                     self.dt_proj.weight,
#                     A,
#                     None,  # input-dependent B
#                     None,  # input-dependent C
#                     self.D.float(),
#                     delta_bias=self.dt_proj.bias.float(),
#                     delta_softplus=True,
#                 )
#                 out_b, so_b = mamba_inner_fn_no_out_proj_out(
#                     xz.flip([-1]),
#                     self.conv1d_b.weight,
#                     self.conv1d_b.bias,
#                     self.x_proj_b.weight,
#                     self.dt_proj_b.weight,
#                     A_b,
#                     None,
#                     None,
#                     self.D_b.float(),
#                     delta_bias=self.dt_proj_b.bias.float(),
#                     delta_softplus=True,
#                 )
                
#                 if self.tiscan:
#                     # ti scan 수행하기
#                     xz_temp = rearrange(xz, "b d (t s) -> b d t s", t=frames, s=spatial)
#                     ind = ind.to(xz.device).expand(-1, self.d_inner * 2, -1, -1)
#                     xz_ti = xz_temp.gather(dim=-1, index=ind)
#                     xz_ti = xz_ti.flatten(2)
                    
#                     A_ti = -torch.exp(self.A_ti_log.float())
#                     A_ti_b = -torch.exp(self.A_ti_b_log.float())
#                     out_ti, so_ti = mamba_inner_fn_no_out_proj_out(
#                         xz_ti,
#                         self.conv1d_ti.weight,
#                         self.conv1d_ti.bias,
#                         self.x_proj_ti.weight,
#                         self.dt_proj_ti.weight,
#                         A_ti,
#                         None,  # input-dependent B
#                         None,  # input-dependent C
#                         self.D_ti.float(),
#                         delta_bias=self.dt_proj_ti.bias.float(),
#                         delta_softplus=True,
#                     )
#                     out_ti_b, so_ti_b = mamba_inner_fn_no_out_proj_out(
#                         xz_ti.flip([-1]),
#                         self.conv1d_ti_b.weight,
#                         self.conv1d_ti_b.bias,
#                         self.x_proj_ti_b.weight,
#                         self.dt_proj_ti_b.weight,
#                         A_ti_b,
#                         None,
#                         None,
#                         self.D_ti_b.float(),
#                         delta_bias=self.dt_proj_ti_b.bias.float(),
#                         delta_softplus=True,
#                     )
#                     # ti scan 수행 전 배열 순서로 되돌리기
#                     outso_ti = rearrange(torch.cat((out_ti, so_ti), dim=1), "b d (t s) -> b d t s", t=frames, s=spatial)
#                     outso_ti_b = rearrange(torch.cat((out_ti_b.flip([-1]), so_ti_b.flip([-1])), dim=1), "b d (t s) -> b d t s", t=frames, s=spatial)

#                     outso_ti_temp = torch.zeros_like(outso_ti).scatter_(dim=-1, index=ind, src=outso_ti).flatten(2)
#                     outso_ti_b_temp = torch.zeros_like(outso_ti_b).scatter_(dim=-1, index=ind, src=outso_ti_b).flatten(2)

#                     out_ti, so_ti = outso_ti_temp[:, :self.d_inner], outso_ti_temp[:, self.d_inner:]
#                     out_ti_b, so_ti_b = outso_ti_b_temp[:, :self.d_inner], outso_ti_b_temp[:, self.d_inner:]
                    
#                     # 4가지 out를 활용해서, 최종 out 구함.
#                     alpha = torch.sigmoid(self.alphalogit)  # always in (0, 1)
#                     # print(f'layers {self.layer_idx} - alpha {alpha}')
#                     out = F.linear(rearrange((1-alpha)*(out + out_b.flip([-1])) + alpha*(out_ti + out_ti_b), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                    
#                     # 4가지 so를 활용해서, score 구함.
#                     # token pruning using token importance evaluation (from 'Exploring Token Pruning in Vision State Space Models')
#                     tiscore = torch.relu((1-alpha)*(so + so_b.flip([-1])) + alpha*(so_ti + so_ti_b)).mean(dim=1, keepdim=True)  # b, 1, l
#                 else:
#                     # 2가지 out을 활용해서, 최종 out 구함.
#                     out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                    
#                     # 2가지 so를 활용해서, score 구함.
#                     # token pruning using token importance evaluation (from 'Exploring Token Pruning in Vision State Space Models')
#                     tiscore = torch.relu(so + so_b.flip([-1])).mean(dim=1, keepdim=True)  # b, 1, l
                
#                 # tiscore 기반으로 ti prediction 수행하기
#                 nind=None
#                 topk_num = int(spatial * self.krratio)
#                 tiscore = rearrange(tiscore, "b d (t s) -> b d t s", t=frames, s=spatial)
                
#                 # Key Region 픽셀 위치 추출하기 (largest token importance score)
#                 _, nind = tiscore.topk(topk_num, dim=-1, largest=True, sorted=False)  # b, 1, t, k
#                 key_nind, _ = nind.sort(dim=-1) # ascending
#                 # Non-Key Region 픽셀 위치 추출하기 (smallest token importance score)
#                 _, nind_ = tiscore.topk(spatial - topk_num, dim=-1, largest=False, sorted=False)  # b, 1, t, k
#                 nonkey_nind, _ = nind_.sort(dim=-1) # ascending

#                 # Key Region -> Non-Key Region 순서로 정렬하기
#                 nind = torch.cat((nonkey_nind, key_nind),dim=-1)
                
#             elif self.bimamba:
#                 A_b = -torch.exp(self.A_b_log.float())
#                 out = mamba_inner_fn_no_out_proj(
#                     xz,
#                     self.conv1d.weight,
#                     self.conv1d.bias,
#                     self.x_proj.weight,
#                     self.dt_proj.weight,
#                     A,
#                     None,  # input-dependent B
#                     None,  # input-dependent C
#                     self.D.float(),
#                     delta_bias=self.dt_proj.bias.float(),
#                     delta_softplus=True,
#                 )
#                 out_b = mamba_inner_fn_no_out_proj(
#                     xz.flip([-1]),
#                     self.conv1d_b.weight,
#                     self.conv1d_b.bias,
#                     self.x_proj_b.weight,
#                     self.dt_proj_b.weight,
#                     A_b,
#                     None,
#                     None,
#                     self.D_b.float(),
#                     delta_bias=self.dt_proj_b.bias.float(),
#                     delta_softplus=True,
#                 )
#                 out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
#                 nind=None
#             else:
#                 out = mamba_inner_fn(
#                     xz,
#                     self.conv1d.weight,
#                     self.conv1d.bias,
#                     self.x_proj.weight,
#                     self.dt_proj.weight,
#                     self.out_proj.weight,
#                     self.out_proj.bias,
#                     A,
#                     None,  # input-dependent B
#                     None,  # input-dependent C
#                     self.D.float(),
#                     delta_bias=self.dt_proj.bias.float(),
#                     delta_softplus=True,
#                 )
#         else:
#             x, z = xz.chunk(2, dim=1)
#             # Compute short convolution
#             if conv_state is not None:
#                 conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
#             if causal_conv1d_fn is None:
#                 x = self.act(self.conv1d(x)[..., :seqlen])
#             else:
#                 assert self.activation in ["silu", "swish"]
#                 x = causal_conv1d_fn(
#                     x,
#                     rearrange(self.conv1d.weight, "d 1 w -> d w"),
#                     self.conv1d.bias,
#                     self.activation,
#                 )

#             # We're careful here about the layout, to avoid extra transposes.
#             # We want dt to have d as the slowest moving dimension
#             # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
#             x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
#             dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#             dt = self.dt_proj.weight @ dt.t()
#             dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
#             B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
#             C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
#             assert self.activation in ["silu", "swish"]
#             y = selective_scan_fn(
#                 x,
#                 dt,
#                 A,
#                 B,
#                 C,
#                 self.D.float(),
#                 z=z,
#                 delta_bias=self.dt_proj.bias.float(),
#                 delta_softplus=True,
#                 return_last_state=ssm_state is not None,
#             )
#             if ssm_state is not None:
#                 y, last_state = y
#                 ssm_state.copy_(last_state)
#             y = rearrange(y, "b d l -> b l d")
#             out = self.out_proj(y)
#         return out, nind

#     def step(self, hidden_states, conv_state, ssm_state):
#         dtype = hidden_states.dtype
#         assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
#         xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
#         x, z = xz.chunk(2, dim=-1)  # (B D)

#         # Conv step
#         if causal_conv1d_update is None:
#             conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
#             conv_state[:, :, -1] = x
#             x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
#             if self.conv1d.bias is not None:
#                 x = x + self.conv1d.bias
#             x = self.act(x).to(dtype=dtype)
#         else:
#             x = causal_conv1d_update(
#                 x,
#                 conv_state,
#                 rearrange(self.conv1d.weight, "d 1 w -> d w"),
#                 self.conv1d.bias,
#                 self.activation,
#             )

#         x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
#         dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
#         # Don't add dt_bias here
#         dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
#         A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

#         # SSM step
#         if selective_state_update is None:
#             # Discretize A and B
#             dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
#             dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
#             dB = torch.einsum("bd,bn->bdn", dt, B)
#             ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
#             y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
#             y = y + self.D.to(dtype) * x
#             y = y * self.act(z)  # (B D)
#         else:
#             y = selective_state_update(
#                 ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
#             )

#         out = self.out_proj(y)
#         return out.unsqueeze(1), conv_state, ssm_state

#     def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
#         device = self.out_proj.weight.device
#         conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
#         conv_state = torch.zeros(
#             batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
#         )
#         ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
#         # ssm_dtype = torch.float32
#         ssm_state = torch.zeros(
#             batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
#         )
#         return conv_state, ssm_state

#     def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
#         assert self.layer_idx is not None
#         if self.layer_idx not in inference_params.key_value_memory_dict:
#             batch_shape = (batch_size,)
#             conv_state = torch.zeros(
#                 batch_size,
#                 self.d_model * self.expand,
#                 self.d_conv,
#                 device=self.conv1d.weight.device,
#                 dtype=self.conv1d.weight.dtype,
#             )
#             ssm_state = torch.zeros(
#                 batch_size,
#                 self.d_model * self.expand,
#                 self.d_state,
#                 device=self.dt_proj.weight.device,
#                 dtype=self.dt_proj.weight.dtype,
#                 # dtype=torch.float32,
#             )
#             inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
#         else:
#             conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
#             # TODO: What if batch size changes between generation, and we reuse the same states?
#             if initialize_states:
#                 conv_state.zero_()
#                 ssm_state.zero_()
#         return conv_state, ssm_state


class Mamba_ti(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=False,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
        tiscan=False,
        tipred=False,
        krratio=0.2,
        token_seq=[8,14,14],
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba
        self.tiscan = tiscan
        self.tipred = tipred
        self.krratio = krratio
        self.token_seq = token_seq

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
        
        # tiscan
        if self.tiscan:
            A_ti = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_ti_log = torch.log(A_ti)
            self.A_ti_log = nn.Parameter(A_ti_log)
            self.A_ti_log._no_weight_decay = True

            self.x_proj_ti = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_ti = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_ti = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_ti._no_weight_decay = True
            
            
            A_ti_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_ti_b_log = torch.log(A_ti_b)
            self.A_ti_b_log = nn.Parameter(A_ti_b_log)
            self.A_ti_b_log._no_weight_decay = True 

            self.x_proj_ti_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_ti_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_ti_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_ti_b._no_weight_decay = True
        
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_norm_b = nn.LayerNorm(self.d_inner)
        self.out_norm_ti = nn.LayerNorm(self.d_inner)
        self.out_norm_ti_b = nn.LayerNorm(self.d_inner)
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, indk, inference_params=None, T=8):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape
        frames = self.token_seq[0]
        spatial = seqlen//frames

        conv_state, ssm_state = None, None
        # if inference_params is not None:
        #     conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
        #     if inference_params.seqlen_offset > 0:
        #         # The states are updated inplace
        #         out, _, _ = self.step(hidden_states, conv_state, ssm_state)
        #         return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else: # Not fast-path
            if self.bimamba:
                x, z = xz.chunk(2, dim=1)
                # Compute short convolution
                # if conv_state is not None:
                #     conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x,
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias,
                        self.activation,
                    )

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                # *********************************forward********************************************
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=None,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
                y = y.permute(0, 2, 1)  # (b l d)
                y = self.out_norm(y)
                y = y.permute(0, 2, 1)  # (b d l)
                yz = y * F.silu(z)
                
                # **********************************backward********************************************
                xz_b = xz.flip([-1])
                x_b, z_b = xz_b.chunk(2, dim=1)
                x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
                dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt_b = self.dt_proj_b.weight @ dt_b.t()
                dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
                B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
                y_b = selective_scan_fn(
                    x_b,
                    dt_b,
                    A_b,
                    B_b,
                    C_b,
                    self.D_b.float(),
                    z=None,
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
                y_b = y_b.permute(0, 2, 1)  # (b l d)
                y_b = self.out_norm_b(y_b)
                y_b = y_b.permute(0, 2, 1)  # (b d l)
                yz_b = y_b * F.silu(z_b)
                y_b = y_b.flip([-1])
                yz_b = yz_b.flip([-1])
                
                if self.tiscan:
                    # *********************************forward********************************************
                    # Rearrange by token importance score
                    xz_ti = xz
                    xz_ti_cls = xz_ti[:, :, 0:1]
                    xz_ti = xz_ti[:, :, 1:]
                    xz_ti = rearrange(xz_ti, "b d (t s) -> b d t s", t=frames, s=spatial)
                    xz_ti = xz_ti.gather(dim=-1, index=indk).flatten(2)
                    xz_ti = torch.cat((xz_ti_cls, xz_ti), dim=-1)
                    
                    x_ti, z_ti = xz_ti.chunk(2, dim=1)
                    x_dbl_ti = self.x_proj_ti(rearrange(x_ti, "b d l -> (b l) d"))  # (bl d)
                    dt_ti, B_ti, C_ti = torch.split(x_dbl_ti, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt_ti = self.dt_proj_ti.weight @ dt_ti.t()
                    dt_ti = rearrange(dt_ti, "d (b l) -> b d l", l=seqlen)
                    B_ti = rearrange(B_ti, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C_ti = rearrange(C_ti, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    A_ti = -torch.exp(self.A_ti_log.float())  # (d_inner, d_state)
                    y_ti = selective_scan_fn(
                        x_ti,
                        dt_ti,
                        A_ti,
                        B_ti,
                        C_ti,
                        self.D_ti.float(),
                        z=None,
                        delta_bias=self.dt_proj_ti.bias.float(),
                        delta_softplus=True,
                        return_last_state=ssm_state is not None,
                    )
                    y_ti = y_ti.permute(0, 2, 1)  # (b l d)
                    y_ti = self.out_norm_ti(y_ti)
                    y_ti = y_ti.permute(0, 2, 1)  # (b d l)
                    yz_ti = y_ti * F.silu(z_ti)
                    
                    y_ti_cls = y_ti[:, :, 0:1]
                    y_ti = y_ti[:, :, 1:]
                    yz_ti_cls = yz_ti[:, :, 0:1]
                    yz_ti = yz_ti[:, :, 1:]
                    
                    # **********************************backward********************************************
                    xz_ti_b = xz_ti.flip([-1])
                    x_ti_b, z_ti_b = xz_ti_b.chunk(2, dim=1)
                    x_dbl_ti_b = self.x_proj_ti_b(rearrange(x_ti_b, "b d l -> (b l) d"))  # (bl d)
                    dt_ti_b, B_ti_b, C_ti_b = torch.split(x_dbl_ti_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                    dt_ti_b = self.dt_proj_ti_b.weight @ dt_ti_b.t()
                    dt_ti_b = rearrange(dt_ti_b, "d (b l) -> b d l", l=seqlen)
                    B_ti_b = rearrange(B_ti_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    C_ti_b = rearrange(C_ti_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                    assert self.activation in ["silu", "swish"]
                    A_ti_b = -torch.exp(self.A_ti_b_log.float())  # (d_inner, d_state)
                    y_ti_b = selective_scan_fn(
                        x_ti_b,
                        dt_ti_b,
                        A_ti_b,
                        B_ti_b,
                        C_ti_b,
                        self.D_ti_b.float(),
                        z=None,
                        delta_bias=self.dt_proj_ti_b.bias.float(),
                        delta_softplus=True,
                        return_last_state=ssm_state is not None,
                    )
                    y_ti_b = y_ti_b.permute(0, 2, 1)  # (b l d)
                    y_ti_b = self.out_norm_ti_b(y_ti_b)
                    y_ti_b = y_ti_b.permute(0, 2, 1)  # (b d l)
                    yz_ti_b = y_ti_b * F.silu(z_ti_b)
                    y_ti_b = y_ti_b.flip([-1])
                    yz_ti_b = yz_ti_b.flip([-1])
                    
                    y_ti_b_cls = y_ti_b[:, :, 0:1]
                    y_ti_b = y_ti_b[:, :, 1:]
                    yz_ti_b_cls = yz_ti_b[:, :, 0:1]
                    yz_ti_b = yz_ti_b[:, :, 1:]
                    
                    # Rearrange by token importance score
                    y_yz_ti = rearrange(torch.cat((y_ti, yz_ti), dim=1), "b d (t s) -> b d t s", t=frames, s=spatial)
                    y_yz_ti_b = rearrange(torch.cat((y_ti_b, yz_ti_b), dim=1), "b d (t s) -> b d t s", t=frames, s=spatial)

                    y_yz_ti = torch.zeros_like(y_yz_ti).scatter_(dim=-1, index=indk, src=y_yz_ti).flatten(2)
                    y_yz_ti_b = torch.zeros_like(y_yz_ti_b).scatter_(dim=-1, index=indk, src=y_yz_ti_b).flatten(2)

                    y_ti, yz_ti = y_yz_ti[:, :self.d_inner], y_yz_ti[:, self.d_inner:]
                    y_ti_b, yz_ti_b = y_yz_ti_b[:, :self.d_inner], y_yz_ti_b[:, self.d_inner:]
                    
                    yz_ti = torch.cat((yz_ti_cls, yz_ti), dim=-1)
                    yz_ti_b = torch.cat((yz_ti_b_cls, yz_ti_b), dim=-1)
                    
                    out = yz + yz_b + yz_ti + yz_ti_b
                    ys = y[:, :, 1:] + y_b[:, :, 1:] + y_ti + y_ti_b
                else:
                    out = yz + yz_b
                    ys = y[:, :, 1:] + y_b[:, :, 1:]

                # output calulation
                out = self.out_proj(rearrange(out, "b d l -> b l d"))
                # token importance score prediction (from 'Exploring Token Pruning in Vision State Space Models')
                indk = None
                if self.tipred:
                    tiscore = torch.relu(ys).mean(dim=1, keepdim=True)  # b, 1, l
                    tiscore = rearrange(tiscore, "b d (t s) -> b d t s", t=frames, s=spatial)
                    topk_num = int(spatial * self.krratio)
                    
                    # Key Region tokens extracting
                    _, ind = tiscore.topk(topk_num, dim=-1, largest=True, sorted=False)  # b, 1, t, k
                    key_ind, _ = ind.sort(dim=-1) # ascending
                    # Non-Key Region tokens extracting
                    _, ind_ = tiscore.topk(spatial - topk_num, dim=-1, largest=False, sorted=False)  # b, 1, t, k
                    nonkey_ind, _ = ind_.sort(dim=-1) # ascending

                    # sequence rearrange (Key Region => Non-Key Region)
                    indk = torch.cat((nonkey_ind, key_ind),dim=-1).expand(-1, self.d_inner * 2, -1, -1)

            else:
                x, z = xz.chunk(2, dim=1)
                # Compute short convolution
                if conv_state is not None:
                    conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
                if causal_conv1d_fn is None:
                    x = self.act(self.conv1d(x)[..., :seqlen])
                else:
                    assert self.activation in ["silu", "swish"]
                    x = causal_conv1d_fn(
                        x,
                        rearrange(self.conv1d.weight, "d 1 w -> d w"),
                        self.conv1d.bias,
                        self.activation,
                    )

                # We're careful here about the layout, to avoid extra transposes.
                # We want dt to have d as the slowest moving dimension
                # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
                x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
                dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
                dt = self.dt_proj.weight @ dt.t()
                dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
                B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
                assert self.activation in ["silu", "swish"]
                y = selective_scan_fn(
                    x,
                    dt,
                    A,
                    B,
                    C,
                    self.D.float(),
                    z=z,
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                    return_last_state=ssm_state is not None,
                )
                if ssm_state is not None:
                    y, last_state = y
                    ssm_state.copy_(last_state)
                y = rearrange(y, "b d l -> b l d")
                out = self.out_proj(y)
        return out, indk

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Mamba2d_krs(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        layer_krs=False,
        device=None,
        dtype=None,
        bimamba=True,
        vertical=True,
        token_seq=[8,14,14],
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.layer_krs = layer_krs
        self.bimamba = bimamba
        self.vertical = vertical
        self.token_seq = token_seq

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
         
        # vertical
        if self.vertical:
            A_v = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_v_log = torch.log(A_v)  # Keep A_v_log in fp32
            self.A_v_log = nn.Parameter(A_v_log)
            self.A_v_log._no_weight_decay = True 

            self.conv1d_v = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_v = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_v = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_v = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_v._no_weight_decay = True
            
            
            A_bv = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_bv_log = torch.log(A_bv)  # Keep A_bv_log in fp32
            self.A_bv_log = nn.Parameter(A_bv_log)
            self.A_bv_log._no_weight_decay = True 

            self.conv1d_bv = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_bv = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_bv = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_bv = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_bv._no_weight_decay = True
            

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inds, inference_params=None, T=8):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba and self.vertical:
                xz_v = rearrange(xz, 'b d (t h w)-> b d (t w h)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])

                if (inds is not None) and self.layer_krs:
                    frames = T
                    spatial = seqlen//frames

                    xz_temp = rearrange(xz, "b d (t s) -> b d t s", t=frames, s=spatial)
                    xz_v_temp = rearrange(xz_v, "b d (t s) -> b d t s", t=frames, s=spatial)
                    
                    inds0 = inds[0].to(xz.device).expand(-1, self.d_inner * 2, -1, -1)
                    inds1 = inds[1].to(xz.device).expand(-1, self.d_inner * 2, -1, -1)

                    xz = xz_temp.gather(dim=-1, index=inds0)
                    xz_v = xz_v_temp.gather(dim=-1, index=inds1)

                    # fast flatten
                    xz = xz.flatten(2)
                    xz_v = xz_v.flatten(2)
                
                A_b = -torch.exp(self.A_b_log.float())
                A_v = -torch.exp(self.A_v_log.float())
                A_bv = -torch.exp(self.A_bv_log.float())
                out, so = mamba_inner_fn_no_out_proj_out(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b, so_b = mamba_inner_fn_no_out_proj_out(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out_v, so_v = mamba_inner_fn_no_out_proj_out(
                    xz_v,
                    self.conv1d_v.weight,
                    self.conv1d_v.bias,
                    self.x_proj_v.weight,
                    self.dt_proj_v.weight,
                    A_v,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D_v.float(),
                    delta_bias=self.dt_proj_v.bias.float(),
                    delta_softplus=True,
                )
                out_bv, so_bv = mamba_inner_fn_no_out_proj_out(
                    xz_v.flip([-1]),
                    self.conv1d_bv.weight,
                    self.conv1d_bv.bias,
                    self.x_proj_bv.weight,
                    self.dt_proj_bv.weight,
                    A_bv,
                    None,
                    None,
                    self.D_bv.float(),
                    delta_bias=self.dt_proj_bv.bias.float(),
                    delta_softplus=True,
                )

                if (inds is not None) and self.layer_krs:
                    outso = torch.cat((out, so), dim=1)
                    outso_b = torch.cat((out_b.flip([-1]), so_b.flip([-1])), dim=1)
                    outso_v = torch.cat((out_v, so_v), dim=1)
                    outso_bv = torch.cat((out_bv.flip([-1]), so_bv.flip([-1])), dim=1)
                    
                    # fill hidden states of pruned tokens with zero.
                    outso = rearrange(outso, "b d (t s) -> b d t s", t=frames, s=spatial)
                    outso_b = rearrange(outso_b, "b d (t s) -> b d t s", t=frames, s=spatial)
                    outso_v = rearrange(outso_v, "b d (t s) -> b d t s", t=frames, s=spatial)
                    outso_bv = rearrange(outso_bv, "b d (t s) -> b d t s", t=frames, s=spatial)

                    # faster scatter with inplace
                    outso_temp = torch.zeros_like(outso).scatter_(dim=-1, index=inds0, src=outso)
                    outso_b_temp = torch.zeros_like(outso_b).scatter_(dim=-1, index=inds0, src=outso_b)
                    outso_v_temp = torch.zeros_like(outso_v).scatter_(dim=-1, index=inds1, src=outso_v)
                    outso_bv_temp = torch.zeros_like(outso_bv).scatter_(dim=-1, index=inds1, src=outso_bv)

                    # flatten back
                    outso_temp = outso_temp.flatten(2)
                    outso_b_temp = outso_b_temp.flatten(2)
                    outso_v_temp = outso_v_temp.flatten(2)
                    outso_bv_temp = outso_bv_temp.flatten(2)

                    out, so = outso_temp[:, :self.d_inner], outso_temp[:, self.d_inner:]
                    out_b, so_b = outso_b_temp[:, :self.d_inner], outso_b_temp[:, self.d_inner:]
                    out_v, so_v = outso_v_temp[:, :self.d_inner], outso_v_temp[:, self.d_inner:]
                    out_bv, so_bv = outso_bv_temp[:, :self.d_inner], outso_bv_temp[:, self.d_inner:]
                else:
                    out_b = out_b.flip([-1])
                    so_b = so_b.flip([-1])
                    out_bv = out_bv.flip([-1])
                    so_bv = so_bv.flip([-1])
                
                out_v = rearrange(out_v, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])
                out_bv = rearrange(out_bv, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])
                so_v = rearrange(so_v, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])
                so_bv = rearrange(so_bv, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])
                
                out = F.linear(rearrange(out + out_b + out_v + out_bv, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

                ninds=None
                if self.layer_krs:
                    # token pruning using token importance evaluation (from 'Exploring Token Pruning in Vision State Space Models')
                    tiscore = torch.relu(so+so_b+so_v+so_bv).mean(dim=1, keepdim=True)  # b, 1, l
                    
                    # Seperate by frame
                    frames = T
                    spatial = seqlen//frames
                    score = rearrange(tiscore, "b d (t h w) -> b d t (h w)", t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])  # b, 1, t, s
                    score_v = rearrange(tiscore, "b d (t h w) -> b d t (w h)", t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])  # b, 1, t, s
                    
                    # Precompute how many top tokens
                    topk_num = int(spatial * 0.2)
                    bkg_num = spatial - topk_num
                    
                    # Foreground (largest scores)
                    _, nind = score.topk(topk_num, dim=-1, largest=True, sorted=False)  # b, 1, t, k
                    _, nind_v = score_v.topk(topk_num, dim=-1, largest=True, sorted=False)  # b, 1, t, k
                    srt_nind, _ = nind.sort(dim=-1) # ascending
                    srt_nind_v, _ = nind_v.sort(dim=-1) # ascending
                    # background (smallest scores)
                    _, bgnind = score.topk(bkg_num, dim=-1, largest=False, sorted=False)  # b, 1, t, k
                    _, bgnind_v = score_v.topk(bkg_num, dim=-1, largest=False, sorted=False)  # b, 1, t, k
                    srt_bgnind, _ = bgnind.sort(dim=-1) # ascending
                    srt_bgnind_v, _ = bgnind_v.sort(dim=-1) # ascending

                    # Concatenate sorted indices
                    ninds=[
                        torch.cat((srt_bgnind, srt_nind),dim=-1), 
                        torch.cat((srt_bgnind_v, srt_nind_v),dim=-1)
                    ]
                
            elif self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
                ninds=None
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out, ninds

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Mamba2d_krs_prog(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        layer_krs=False,
        layer_fgr=0.0,
        device=None,
        dtype=None,
        bimamba=True,
        vertical=True,
        token_seq=[8,14,14],
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.layer_krs = layer_krs
        self.layer_fgr = layer_fgr
        self.bimamba = bimamba
        self.vertical = vertical
        self.token_seq = token_seq

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
         
        # vertical
        if self.vertical:
            self.in_proj_v = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
            A_v = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_v_log = torch.log(A_v)  # Keep A_v_log in fp32
            self.A_v_log = nn.Parameter(A_v_log)
            self.A_v_log._no_weight_decay = True 

            self.conv1d_v = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_v = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_v = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_v = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_v._no_weight_decay = True
            
            
            A_bv = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_bv_log = torch.log(A_bv)  # Keep A_bv_log in fp32
            self.A_bv_log = nn.Parameter(A_bv_log)
            self.A_bv_log._no_weight_decay = True 

            self.conv1d_bv = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_bv = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_bv = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_bv = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_bv._no_weight_decay = True
            

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inds, inference_params=None, T=8):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba and self.vertical:
                hidden_states_v = torch.cat((hidden_states[:, :1, :], 
                                            rearrange(hidden_states[:, 1:, :], 'b (t h w) d -> b (t w h) d', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])), dim=1)
                
                xz_v = rearrange(
                    self.in_proj_v.weight @ rearrange(hidden_states_v, "b l d -> d (b l)"),
                    "d (b l) -> b d l",
                    l=seqlen,
                )
                if self.in_proj_v.bias is not None:
                    xz_v = xz_v + rearrange(self.in_proj_v.bias.to(dtype=xz_v.dtype), "d -> d 1")

                if inds is not None:
                    frames = T
                    spatial = seqlen//frames
                    
                    xz_cls = xz[:, :, :1]; xz = xz[:, :, 1:]
                    xz_v_cls = xz_v[:, :, :1]; xz_v = xz_v[:, :, 1:]
                    xz_temp = rearrange(xz, "b d (t s) -> b d t s", t=frames, s=spatial)
                    xz_v_temp = rearrange(xz_v, "b d (t s) -> b d t s", t=frames, s=spatial)
                    xz = xz_temp.gather(dim=-1, index=inds[0].expand(-1, self.d_inner * 2, -1, -1))  # b, d, t, k
                    xz_v = xz_v_temp.gather(dim=-1, index=inds[1].expand(-1, self.d_inner * 2, -1, -1))  # b, d, t, k
                    xz = rearrange(xz, "b d t s -> b d (t s)")
                    xz_v = rearrange(xz_v, "b d t s -> b d (t s)")
                    xz = torch.cat((xz_cls, xz), dim=-1)
                    xz_v = torch.cat((xz_v_cls, xz_v), dim=-1)
                
                A_b = -torch.exp(self.A_b_log.float())
                A_v = -torch.exp(self.A_v_log.float())
                A_bv = -torch.exp(self.A_bv_log.float())
                out, so = mamba_inner_fn_no_out_proj_out(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b, so_b = mamba_inner_fn_no_out_proj_out(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out_v, so_v = mamba_inner_fn_no_out_proj_out(
                    xz_v,
                    self.conv1d_v.weight,
                    self.conv1d_v.bias,
                    self.x_proj_v.weight,
                    self.dt_proj_v.weight,
                    A_v,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D_v.float(),
                    delta_bias=self.dt_proj_v.bias.float(),
                    delta_softplus=True,
                )
                out_bv, so_bv = mamba_inner_fn_no_out_proj_out(
                    xz_v.flip([-1]),
                    self.conv1d_bv.weight,
                    self.conv1d_bv.bias,
                    self.x_proj_bv.weight,
                    self.dt_proj_bv.weight,
                    A_bv,
                    None,
                    None,
                    self.D_bv.float(),
                    delta_bias=self.dt_proj_bv.bias.float(),
                    delta_softplus=True,
                )

                out_b = out_b.flip([-1])
                so_b = so_b.flip([-1])
                out_bv = out_bv.flip([-1])
                so_bv = so_bv.flip([-1])
                
                out_cls=out[:, :, :1]; out_b_cls=out_b[:, :, :1]; out_v_cls=out_v[:, :, :1]; out_bv_cls=out_bv[:, :, :1]
                out=out[:, :, 1:]; out_b=out_b[:, :, 1:]; out_v=out_v[:, :, 1:]; out_bv=out_bv[:, :, 1:]
                so=so[:, :, 1:]; so_b=so_b[:, :, 1:]; so_v=so_v[:, :, 1:]; so_bv=so_bv[:, :, 1:]
                
                if inds is not None:
                    outso = torch.cat((out, so), dim=1)
                    outso_b = torch.cat((out_b, so_b), dim=1)
                    outso_v = torch.cat((out_v, so_v), dim=1)
                    outso_bv = torch.cat((out_bv, so_bv), dim=1)
                    
                    # fill hidden states of pruned tokens with zero.
                    outso = rearrange(outso, "b d (t s) -> b d t s", t=frames, s=spatial)
                    outso_b = rearrange(outso_b, "b d (t s) -> b d t s", t=frames, s=spatial)
                    outso_v = rearrange(outso_v, "b d (t s) -> b d t s", t=frames, s=spatial)
                    outso_bv = rearrange(outso_bv, "b d (t s) -> b d t s", t=frames, s=spatial)
                    
                    outso_temp = torch.zeros_like(outso)
                    outso_b_temp = torch.zeros_like(outso_b)
                    outso_v_temp = torch.zeros_like(outso_v)
                    outso_bv_temp = torch.zeros_like(outso_bv)

                    outso_temp = outso_temp.scatter(dim=-1, index=inds[0].expand(-1, self.d_inner*2, -1, -1), src=outso)  # b, d, t, k
                    outso_b_temp = outso_b_temp.scatter(dim=-1, index=inds[0].expand(-1, self.d_inner*2, -1, -1), src=outso_b)  # b, d, t, k
                    outso_v_temp = outso_v_temp.scatter(dim=-1, index=inds[1].expand(-1, self.d_inner*2, -1, -1), src=outso_v)  # b, d, t, k
                    outso_bv_temp = outso_bv_temp.scatter(dim=-1, index=inds[1].expand(-1, self.d_inner*2, -1, -1), src=outso_bv)  # b, d, t, k
                    
                    outso_temp = rearrange(outso_temp, "b d t s -> b d (t s)")
                    outso_b_temp = rearrange(outso_b_temp, "b d t s -> b d (t s)")
                    outso_v_temp = rearrange(outso_v_temp, "b d t s -> b d (t s)")
                    outso_bv_temp = rearrange(outso_bv_temp, "b d t s -> b d (t s)")

                    out, so = outso_temp[:, :self.d_inner], outso_temp[:, self.d_inner:]
                    out_b, so_b = outso_b_temp[:, :self.d_inner], outso_b_temp[:, self.d_inner:]
                    out_v, so_v = outso_v_temp[:, :self.d_inner], outso_v_temp[:, self.d_inner:]
                    out_bv, so_bv = outso_bv_temp[:, :self.d_inner], outso_bv_temp[:, self.d_inner:]
                    
                
                out = torch.cat((out_cls, out), dim=-1)
                out_b = torch.cat((out_b_cls, out_b), dim=-1)
                out_v = torch.cat((out_v_cls, rearrange(out_v, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])), dim=-1)
                out_bv = torch.cat((out_bv_cls, rearrange(out_bv, 'b d (t w h) -> b d (t h w)', t=self.token_seq[0], h=self.token_seq[1], w=self.token_seq[2])), dim=-1)
                out = F.linear(rearrange(out + out_b + out_v + out_bv, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)

                ninds=None
                if self.layer_krs:
                    # token pruning using token importance evaluation (from 'Exploring Token Pruning in Vision State Space Models')
                    score = torch.relu(so+so_b).mean(dim=1, keepdim=True)  # b, 1, l
                    score_v = torch.relu(so_v+so_bv).mean(dim=1, keepdim=True)  # b, 1, l
                    
                    # Seperate by frame
                    frames = T
                    spatial = seqlen//frames
                    score = rearrange(score, "b d (t s) -> b d t s", t=frames, s=spatial)  # b, 1, t, s
                    score_v = rearrange(score_v, "b d (t s) -> b d t s", t=frames, s=spatial)  # b, 1, t, s
                    # foreground
                    _, nind = score.topk(int(spatial*self.layer_fgr), dim=-1, largest=True, sorted=False)  # b, 1, t, k
                    _, nind_v = score_v.topk(int(spatial*self.layer_fgr), dim=-1, largest=True, sorted=False)  # b, 1, t, k
                    srt_nind, _ = nind.sort(dim=-1) # ascending
                    srt_nind_v, _ = nind_v.sort(dim=-1) # ascending
                    # background
                    _, bgnind = score.topk(spatial-int(spatial*self.layer_fgr), dim=-1, largest=False, sorted=False)  # b, 1, t, k
                    _, bgnind_v = score_v.topk(spatial-int(spatial*self.layer_fgr), dim=-1, largest=False, sorted=False)  # b, 1, t, k
                    srt_bgnind, _ = bgnind.sort(dim=-1) # ascending
                    srt_bgnind_v, _ = bgnind_v.sort(dim=-1) # ascending

                    ninds=[torch.cat((srt_nind,srt_bgnind),dim=-1), torch.cat((srt_nind_v,srt_bgnind_v),dim=-1)]
                
            elif self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out, ninds

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Mamba_quadtree(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
        # ========================
        token_seq=[8,14,14],
        stage_num=0,
        depth_num=0,
        block_depth=[0],
        # ========================
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Quadtree
        self.token_seq = token_seq
        self.stage_num = stage_num
        self.depth_num = depth_num
        self.block_index = (sum(block_depth[0:stage_num]) + depth_num)if stage_num>=1 else depth_num
        self.quad_flag = False
        self.shift_flag = False
        if self.stage_num == 0:
            self.score_predictor = Predictor(self.d_inner)
            self.quad_flag = True
            if self.depth_num % 2 == 1:
                self.shift_flag = True
        

    def forward(self, hidden_states, inference_params=None, T=1):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        
        if self.bimamba:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            
            if self.quad_flag:
                # reshape
                b, d, l = x.shape
                t, h, w = self.token_seq
                x = x.reshape(b, d, t, h, w)
                score = self.score_predictor(x)
                score = rearrange(score, "b d t h w -> (b t) d h w", t=t)
                x = rearrange(x, "b d t h w -> (b t) d h w", t=t)
                
                b, d, h, w = x.shape
                # score prediction+
                quad_size = int(2)
                quad_number = quad_size * quad_size
                # score = self.score_predictor(x)
                if self.shift_flag:
                    shift_size, reverse_size = shift_size_generate(self.block_index, h)

                    x = torch.roll(x, shifts=shift_size, dims=(2, 3))
                    # print(f"shift: {shift_size}, reverse: {reverse_size}")

                if h % quad_number != 0 or w % quad_number != 0:
                    # print("h % quad_number != 0 or w % quad_number != 0")
                    newH, newW = math.ceil(h / quad_number) * quad_number, math.ceil(w / quad_number) * quad_number
                    diff_H, diff_W = newH - h, newW - w
                    x = F.pad(x, (0, diff_H, 0, diff_W, 0, 0))
                    score = F.pad(score, (0, diff_H, 0, diff_W, 0, 0))

                    b, d, h, w = x.shape
                    diff_flag = True
                else:
                    diff_flag = False

                ### quad_one_stage
                x_rs = x.reshape(b, d, -1)
                #score_window = window_partition(score[:, 0:1, :, :], quad_size=quad_size)  # [b, 4, h/4, w/4, 1]
                #score_window_sum = score_window.reshape(B, quad_number, -1, 1).sum(dim=-2, keepdims=True)  # b, 4, 1, 1
                score_window = F.adaptive_avg_pool2d(score[:, 0:1, :, :], (2, 2)) # b, 1, 2, 2
                hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, hard=True, tau=2.0).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]
                
                # print(f"Shape: {hard_keep_decision.shape}")
                # for _ in range(hard_keep_decision.shape[0]):
                #     print(f"Mask: {hard_keep_decision[_].squeeze(-1).squeeze(-1)}")
                #     print(f"Score: {score_window[_].squeeze(0).view(4)}")

                hard_keep_decision_mask = window_expansion(hard_keep_decision, H=int(h), W=int(w))  # [b, 1, l]
                x_masked_select = x_rs * hard_keep_decision_mask
                x_masked_nonselect = x_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_masked_select_localscan = local_scan_quad_quad(x_masked_select, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_masked_nonselect_localscan = local_scan_quad(x_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_quad_window = x_masked_nonselect_localscan + x_masked_select_localscan  # B, C, L
                
                # reshape
                x = rearrange(x_quad_window, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                seqlen = t * h * w
                

            # bi-directional scan (forward)
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            
            # bi-directional scan (backward)
            A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
            
            x_dbl_b = self.x_proj_b(rearrange(x.flip([-1]), "b d l -> (b l) d"))  # (bl d)
            dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_b.weight @ dt_b.t()
            dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y_b = selective_scan_fn(
                x.flip([-1]),
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=None,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y_b = y_b.flip([-1])
            y += y_b
            
            if self.quad_flag:
                # reshape
                y = rearrange(y, "b d (t h w) -> (b t) d (h w)", t=t, h=h, w=w)
                
                # for quad
                y_select = local_reverse_quad_quad(y, H=int(h), W=int(w))
                y_nonselect = local_reverse_quad(y, H=int(h), W=int(w))
                y_hard_keep_decision_mask = hard_keep_decision_mask.clone()
                y_masked_select = y_select * y_hard_keep_decision_mask
                y_masked_nonselect = y_nonselect * (1.0 - y_hard_keep_decision_mask)

                y = y_masked_select + y_masked_nonselect  # B, C, L

                if diff_flag:
                    y = y.reshape(b, d, h, -1)
                    y = y[:, :, 0:-diff_H, 0:-diff_W].contiguous()
                    h, w = h - diff_H, w - diff_W
                else:
                    y = y.view(b, d, h, -1)

                if self.shift_flag:
                    y = torch.roll(y, shifts=reverse_size, dims=(2, 3))

                # reshape
                y = rearrange(y, "(b t) d h w -> b d (t h w)", t=t, h=h, w=w)
            
            
            y = y.permute(0, 2, 1)  # (b l d)
            y = self.out_norm(y)
            
            z = z.permute(0, 2, 1)  # (b l d)
            y = y * F.silu(z)
            
            out = self.out_proj(y)
        
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
class Mamba_quadtreev2(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
        # ========================
        token_seq=[8,14,14],
        stage_num=0,
        depth_num=0,
        block_depth=[0],
        # ========================
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_norm_b = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Quadtree
        self.token_seq = token_seq
        self.stage_num = stage_num
        self.depth_num = depth_num
        self.block_index = (sum(block_depth[0:stage_num]) + depth_num)if stage_num>=1 else depth_num
        self.quad_flag = False
        self.shift_flag = False
        if self.stage_num == 0:
            self.quad_flag = True
            if self.depth_num % 2 == 1:
                self.shift_flag = True
                
            # Hibert curve gen.
            H, W = self.token_seq[1], self.token_seq[2]
            quad_size = int(2)
            quad_number = quad_size * 8
            if H % quad_number != 0 or W % quad_number != 0:
                # print("h % quad_number != 0 or w % quad_number != 0")
                H, W = math.ceil(H / quad_number) * quad_number, math.ceil(W / quad_number) * quad_number
            
            H //= 2
            W //= 2
            p = int(np.log2(H))  # 힐베르트 곡선의 단계 (order)
            n = 2  # 2차원

            # 힐베르트 곡선 객체 생성
            hilbert_curve = HilbertCurve(p, n)

            # 힐베르트 곡선의 전체 좌표 계산
            coords = []
            for y in range(H):
                for x in range(W):
                    coords.append((x, y))

            # 각 좌표에 대한 힐베르트 인덱스 계산
            hilbert_indices = []
            
            for coord in coords:
                x, y = coord
                # 힐베르트 곡선의 크기에 맞게 좌표 조정
                hilbert_index = hilbert_curve.distance_from_point([x, y])
                hilbert_indices.append(hilbert_index)

            # 힐베르트 인덱스에 따라 정렬
            hilbert_indices = np.array(hilbert_indices)
            self.hibert_sorted_indices = sorted_indices = np.argsort(hilbert_indices)
            # 역순서 인덱스 계산
            self.hibert_inverse_indices = np.argsort(sorted_indices)
            
            

    def forward(self, hidden_states, prev_score, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        
        if self.bimamba:
            x, z = xz.chunk(2, dim=1)
            x_b = x.flip([-1])
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            
            if causal_conv1d_fn is None:
                x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x_b = causal_conv1d_fn(
                    x_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.activation,
                )
            
            if self.quad_flag:
                # reshape
                b, d, l = x.shape
                t, h, w = self.token_seq
                x_cls = x[:, :, 0:1]
                x = x[:, :, 1:]
                x = x.reshape(b, d, t, h, w)
                x = rearrange(x, "b d t h w -> (b t) d h w", t=t)
                
                x_b = x_b.flip([-1]) 
                x_b_cls = x_b[:, :, 0:1]
                x_b = x_b[:, :, 1:]
                x_b = x_b.reshape(b, d, t, h, w)
                x_b = rearrange(x_b, "b d t h w -> (b t) d h w", t=t)
                
                b, d, h, w = x.shape

                quad_size = int(2)
                quad_number = quad_size * 8
                # 이전 stage에서 받아온 score를 활용
                if prev_score is None:
                    score = torch.relu(x+x_b).mean(dim=1, keepdim=True)  # (b, 1, h, w)
                else:
                    score = prev_score  # (b, 1, h, w)
                if self.shift_flag:
                    shift_size, reverse_size = shift_size_generate(self.block_index, h)

                    x = torch.roll(x, shifts=shift_size, dims=(2, 3))
                    x_b = torch.roll(x_b, shifts=shift_size, dims=(2, 3))
                
                if h % quad_number != 0 or w % quad_number != 0:
                    # print("h % quad_number != 0 or w % quad_number != 0")
                    newH, newW = math.ceil(h / quad_number) * quad_number, math.ceil(w / quad_number) * quad_number
                    diff_H, diff_W = newH - h, newW - w
                    # Apply padding evenly on all four sides (top, bottom, left, and right)            
                    pad_top = diff_H // 2
                    pad_bottom = diff_H - pad_top
                    pad_left = diff_W // 2
                    pad_right = diff_W - pad_left

                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
                    x_b = F.pad(x_b, (pad_left, pad_right, pad_top, pad_bottom))
                    # score = F.pad(score, (pad_left, pad_right, pad_top, pad_bottom))

                    b, d, h, w = x.shape
                    diff_flag = True
                else:
                    diff_flag = False

                ### quad_one_stage
                x_rs = x.reshape(b, d, -1)
                x_b_rs = x_b.reshape(b, d, -1)
                score_window = F.adaptive_avg_pool2d(score[:, 0:1, :, :], (2, 2)) # b, 1, 2, 2
                # print(score_window)
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=2.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # smoothness
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=1.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]
                hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=0.3, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # sharpness
                # print(hard_keep_decision)
                
                # (forward)
                hard_keep_decision_mask = window_expansion(hard_keep_decision, H=int(h), W=int(w))  # [b, 1, l]
                x_masked_select = x_rs * hard_keep_decision_mask
                x_masked_nonselect = x_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_masked_select_localscan = apply_hilbert_curve_2d_quad(x_masked_select, self.hibert_sorted_indices, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_masked_nonselect_localscan = local_scan_quad(x_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_quad_window = x_masked_nonselect_localscan + x_masked_select_localscan  # B, C, L
                # Non-ROI & ROI Region Split
                x_quad_window_split = NONROI_ROI_split(x_quad_window, hard_keep_decision, H=int(h), W=int(w))
                
                # (backward)
                x_b_masked_select = x_b_rs * hard_keep_decision_mask
                x_b_masked_nonselect = x_b_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_b_masked_select_localscan = apply_hilbert_curve_2d_quad(x_b_masked_select, self.hibert_sorted_indices, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_masked_nonselect_localscan = local_scan_quad(x_b_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_quad_window = x_b_masked_nonselect_localscan + x_b_masked_select_localscan  # B, C, L
                # Non-ROI & ROI Region Split
                x_b_quad_window_split = NONROI_ROI_split(x_b_quad_window, hard_keep_decision, H=int(h), W=int(w))
                
                # reshape
                x = rearrange(x_quad_window_split, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x = torch.cat((x_cls, x), dim=-1)
                x_b = rearrange(x_b_quad_window_split, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x_b = torch.cat((x_b_cls, x_b), dim=-1)
                x_b = x_b.flip([-1])
                
                seqlen = 1 + (t * h * w)
                

            # bi-directional scan (forward)
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            
            # bi-directional scan (backward)
            A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
            
            x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
            dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_b.weight @ dt_b.t()
            dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y_b = selective_scan_fn(
                x_b,
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=None,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y_b = y_b.flip([-1])
            
            y = self.out_norm(y.permute(0, 2, 1))  # (b l d)
            y_b = self.out_norm_b(y_b.permute(0, 2, 1))  # (b l d)
            y += y_b
            y = y.permute(0, 2, 1)  # (b d l)
            
            y_cls = y[:, :, 0:1]
            y = y[:, :, 1:]
            curr_score = None
            if self.quad_flag:
                # reshape
                y = rearrange(y, "b d (t h w) -> (b t) d (h w)", t=t, h=h, w=w)
                
                # Non-ROI & ROI Region Merge
                y_hard_keep_decision = hard_keep_decision.clone()
                y = NONROI_ROI_merge(y, y_hard_keep_decision, H=int(h), W=int(w))
                
                # for quad
                y_select = reverse_hilbert_curve_2d_quad(y, self.hibert_inverse_indices, H=int(h), W=int(w))
                y_nonselect = local_reverse_quad(y, H=int(h), W=int(w))
                y_hard_keep_decision_mask = hard_keep_decision_mask.clone()
                y_masked_select = y_select * y_hard_keep_decision_mask
                y_masked_nonselect = y_nonselect * (1.0 - y_hard_keep_decision_mask)

                y = y_masked_select + y_masked_nonselect  # B, C, L

                if diff_flag:
                    y = y.reshape(b, d, h, -1)
                    # Remove padding evenly on all four sides (top, bottom, left, and right)            
                    y = y[:, :, pad_top:-pad_bottom, pad_left:-pad_right].contiguous()
                    h, w = h - (pad_top+pad_bottom), w - (pad_left+pad_right)
                    
                else:
                    y = y.view(b, d, h, -1)

                if self.shift_flag:
                    y = torch.roll(y, shifts=reverse_size, dims=(2, 3))

                # reshape
                curr_score = torch.relu(y).mean(dim=1, keepdim=True)  # (bt, 1, h, w)
                y = rearrange(y, "(b t) d h w -> b d (t h w)", t=t, h=h, w=w)
                
            y = torch.cat((y_cls, y), dim=-1)
            
            y = y.permute(0, 2, 1)  # (b l d)
            z = z.permute(0, 2, 1)  # (b l d)
            y = y * F.silu(z)
            
            out = self.out_proj(y)
        
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            curr_score = None
            
        return out, curr_score

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
class Mamba_quadtreev2_nosplit(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
        # ========================
        token_seq=[8,14,14],
        stage_num=0,
        depth_num=0,
        block_depth=[0],
        # ========================
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_norm_b = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Quadtree
        self.token_seq = token_seq
        self.stage_num = stage_num
        self.depth_num = depth_num
        self.block_index = (sum(block_depth[0:stage_num]) + depth_num)if stage_num>=1 else depth_num
        self.quad_flag = False
        self.shift_flag = False
        if self.stage_num == 0:
            self.quad_flag = True
            if self.depth_num % 2 == 1:
                self.shift_flag = True
                
            # Hibert curve gen.
            H, W = self.token_seq[1], self.token_seq[2]
            quad_size = int(2)
            quad_number = quad_size * 8
            if H % quad_number != 0 or W % quad_number != 0:
                # print("h % quad_number != 0 or w % quad_number != 0")
                H, W = math.ceil(H / quad_number) * quad_number, math.ceil(W / quad_number) * quad_number
            
            H //= 2
            W //= 2
            p = int(np.log2(H))  # 힐베르트 곡선의 단계 (order)
            n = 2  # 2차원

            # 힐베르트 곡선 객체 생성
            hilbert_curve = HilbertCurve(p, n)

            # 힐베르트 곡선의 전체 좌표 계산
            coords = []
            for y in range(H):
                for x in range(W):
                    coords.append((x, y))

            # 각 좌표에 대한 힐베르트 인덱스 계산
            hilbert_indices = []
            
            for coord in coords:
                x, y = coord
                # 힐베르트 곡선의 크기에 맞게 좌표 조정
                hilbert_index = hilbert_curve.distance_from_point([x, y])
                hilbert_indices.append(hilbert_index)

            # 힐베르트 인덱스에 따라 정렬
            hilbert_indices = np.array(hilbert_indices)
            self.hibert_sorted_indices = sorted_indices = np.argsort(hilbert_indices)
            # 역순서 인덱스 계산
            self.hibert_inverse_indices = np.argsort(sorted_indices)
            
            

    def forward(self, hidden_states, prev_score, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        
        if self.bimamba:
            x, z = xz.chunk(2, dim=1)
            x_b = x.flip([-1])
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            
            if causal_conv1d_fn is None:
                x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x_b = causal_conv1d_fn(
                    x_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.activation,
                )
            
            if self.quad_flag:
                # reshape
                b, d, l = x.shape
                t, h, w = self.token_seq
                x_cls = x[:, :, 0:1]
                x = x[:, :, 1:]
                x = x.reshape(b, d, t, h, w)
                x = rearrange(x, "b d t h w -> (b t) d h w", t=t)
                
                x_b = x_b.flip([-1]) 
                x_b_cls = x_b[:, :, 0:1]
                x_b = x_b[:, :, 1:]
                x_b = x_b.reshape(b, d, t, h, w)
                x_b = rearrange(x_b, "b d t h w -> (b t) d h w", t=t)
                
                b, d, h, w = x.shape

                quad_size = int(2)
                quad_number = quad_size * 8
                # 이전 stage에서 받아온 score를 활용
                if prev_score is None:
                    score = torch.relu(x+x_b).mean(dim=1, keepdim=True)  # (b, 1, h, w)
                else:
                    score = prev_score  # (b, 1, h, w)
                if self.shift_flag:
                    shift_size, reverse_size = shift_size_generate(self.block_index, h)

                    x = torch.roll(x, shifts=shift_size, dims=(2, 3))
                    x_b = torch.roll(x_b, shifts=shift_size, dims=(2, 3))
                
                if h % quad_number != 0 or w % quad_number != 0:
                    # print("h % quad_number != 0 or w % quad_number != 0")
                    newH, newW = math.ceil(h / quad_number) * quad_number, math.ceil(w / quad_number) * quad_number
                    diff_H, diff_W = newH - h, newW - w
                    # Apply padding evenly on all four sides (top, bottom, left, and right)            
                    pad_top = diff_H // 2
                    pad_bottom = diff_H - pad_top
                    pad_left = diff_W // 2
                    pad_right = diff_W - pad_left

                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
                    x_b = F.pad(x_b, (pad_left, pad_right, pad_top, pad_bottom))
                    # score = F.pad(score, (pad_left, pad_right, pad_top, pad_bottom))

                    b, d, h, w = x.shape
                    diff_flag = True
                else:
                    diff_flag = False

                ### quad_one_stage
                x_rs = x.reshape(b, d, -1)
                x_b_rs = x_b.reshape(b, d, -1)
                score_window = F.adaptive_avg_pool2d(score[:, 0:1, :, :], (2, 2)) # b, 1, 2, 2
                # print(score_window)
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=2.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # smoothness
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=1.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]
                hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=0.3, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # sharpness
                # print(hard_keep_decision)
                
                # (forward)
                hard_keep_decision_mask = window_expansion(hard_keep_decision, H=int(h), W=int(w))  # [b, 1, l]
                x_masked_select = x_rs * hard_keep_decision_mask
                x_masked_nonselect = x_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_masked_select_localscan = apply_hilbert_curve_2d_quad(x_masked_select, self.hibert_sorted_indices, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_masked_nonselect_localscan = local_scan_quad(x_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_quad_window = x_masked_nonselect_localscan + x_masked_select_localscan  # B, C, L
                
                # (backward)
                x_b_masked_select = x_b_rs * hard_keep_decision_mask
                x_b_masked_nonselect = x_b_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_b_masked_select_localscan = apply_hilbert_curve_2d_quad(x_b_masked_select, self.hibert_sorted_indices, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_masked_nonselect_localscan = local_scan_quad(x_b_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_quad_window = x_b_masked_nonselect_localscan + x_b_masked_select_localscan  # B, C, L
                
                # reshape
                x = rearrange(x_quad_window, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x = torch.cat((x_cls, x), dim=-1)
                x_b = rearrange(x_b_quad_window, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x_b = torch.cat((x_b_cls, x_b), dim=-1)
                x_b = x_b.flip([-1])
                
                seqlen = 1 + (t * h * w)
                

            # bi-directional scan (forward)
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            
            # bi-directional scan (backward)
            A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
            
            x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
            dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_b.weight @ dt_b.t()
            dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y_b = selective_scan_fn(
                x_b,
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=None,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y_b = y_b.flip([-1])
            
            y = self.out_norm(y.permute(0, 2, 1))  # (b l d)
            y_b = self.out_norm_b(y_b.permute(0, 2, 1))  # (b l d)
            y += y_b
            y = y.permute(0, 2, 1)  # (b d l)
            
            y_cls = y[:, :, 0:1]
            y = y[:, :, 1:]
            curr_score = None
            if self.quad_flag:
                # reshape
                y = rearrange(y, "b d (t h w) -> (b t) d (h w)", t=t, h=h, w=w)
                
                # for quad
                y_select = reverse_hilbert_curve_2d_quad(y, self.hibert_inverse_indices, H=int(h), W=int(w))
                y_nonselect = local_reverse_quad(y, H=int(h), W=int(w))
                y_hard_keep_decision_mask = hard_keep_decision_mask.clone()
                y_masked_select = y_select * y_hard_keep_decision_mask
                y_masked_nonselect = y_nonselect * (1.0 - y_hard_keep_decision_mask)

                y = y_masked_select + y_masked_nonselect  # B, C, L

                if diff_flag:
                    y = y.reshape(b, d, h, -1)
                    # Remove padding evenly on all four sides (top, bottom, left, and right)            
                    y = y[:, :, pad_top:-pad_bottom, pad_left:-pad_right].contiguous()
                    h, w = h - (pad_top+pad_bottom), w - (pad_left+pad_right)
                    
                else:
                    y = y.view(b, d, h, -1)

                if self.shift_flag:
                    y = torch.roll(y, shifts=reverse_size, dims=(2, 3))

                # reshape
                curr_score = torch.relu(y).mean(dim=1, keepdim=True)  # (bt, 1, h, w)
                y = rearrange(y, "(b t) d h w -> b d (t h w)", t=t, h=h, w=w)
                
            y = torch.cat((y_cls, y), dim=-1)
            
            y = y.permute(0, 2, 1)  # (b l d)
            z = z.permute(0, 2, 1)  # (b l d)
            y = y * F.silu(z)
            
            out = self.out_proj(y)
        
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            curr_score = None
            
        return out, curr_score

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    
class Mamba_quadtreev2_nohibert_nosplit(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
        # ========================
        token_seq=[8,14,14],
        stage_num=0,
        depth_num=0,
        block_depth=[0],
        # ========================
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_norm_b = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Quadtree
        self.token_seq = token_seq
        self.stage_num = stage_num
        self.depth_num = depth_num
        self.block_index = (sum(block_depth[0:stage_num]) + depth_num)if stage_num>=1 else depth_num
        self.quad_flag = False
        self.shift_flag = False
        if self.stage_num == 0:
            self.quad_flag = True
            if self.depth_num % 2 == 1:
                self.shift_flag = True
            

    def forward(self, hidden_states, prev_score, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        
        if self.bimamba:
            x, z = xz.chunk(2, dim=1)
            x_b = x.flip([-1])
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            
            if causal_conv1d_fn is None:
                x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x_b = causal_conv1d_fn(
                    x_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.activation,
                )
            
            if self.quad_flag:
                # reshape
                b, d, l = x.shape
                t, h, w = self.token_seq
                x_cls = x[:, :, 0:1]
                x = x[:, :, 1:]
                x = x.reshape(b, d, t, h, w)
                x = rearrange(x, "b d t h w -> (b t) d h w", t=t)
                
                x_b = x_b.flip([-1]) 
                x_b_cls = x_b[:, :, 0:1]
                x_b = x_b[:, :, 1:]
                x_b = x_b.reshape(b, d, t, h, w)
                x_b = rearrange(x_b, "b d t h w -> (b t) d h w", t=t)
                
                b, d, h, w = x.shape

                quad_size = int(2)
                quad_number = quad_size * 8
                # 이전 stage에서 받아온 score를 활용
                if prev_score is None:
                    score = torch.relu(x+x_b).mean(dim=1, keepdim=True)  # (b, 1, h, w)
                else:
                    score = prev_score  # (b, 1, h, w)
                if self.shift_flag:
                    shift_size, reverse_size = shift_size_generate(self.block_index, h)

                    x = torch.roll(x, shifts=shift_size, dims=(2, 3))
                    x_b = torch.roll(x_b, shifts=shift_size, dims=(2, 3))
                
                if h % quad_number != 0 or w % quad_number != 0:
                    # print("h % quad_number != 0 or w % quad_number != 0")
                    newH, newW = math.ceil(h / quad_number) * quad_number, math.ceil(w / quad_number) * quad_number
                    diff_H, diff_W = newH - h, newW - w
                    # Apply padding evenly on all four sides (top, bottom, left, and right)            
                    pad_top = diff_H // 2
                    pad_bottom = diff_H - pad_top
                    pad_left = diff_W // 2
                    pad_right = diff_W - pad_left

                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
                    x_b = F.pad(x_b, (pad_left, pad_right, pad_top, pad_bottom))
                    # score = F.pad(score, (pad_left, pad_right, pad_top, pad_bottom))

                    b, d, h, w = x.shape
                    diff_flag = True
                else:
                    diff_flag = False

                ### quad_one_stage
                x_rs = x.reshape(b, d, -1)
                x_b_rs = x_b.reshape(b, d, -1)
                score_window = F.adaptive_avg_pool2d(score[:, 0:1, :, :], (2, 2)) # b, 1, 2, 2
                # print(score_window)
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=2.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # smoothness
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=1.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]
                hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=0.3, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # sharpness
                # print(hard_keep_decision)
                
                # (forward)
                hard_keep_decision_mask = window_expansion(hard_keep_decision, H=int(h), W=int(w))  # [b, 1, l]
                x_masked_select = x_rs * hard_keep_decision_mask
                x_masked_nonselect = x_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_masked_select_localscan = local_scan_quad_quad(x_masked_select, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_masked_nonselect_localscan = local_scan_quad(x_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_quad_window = x_masked_nonselect_localscan + x_masked_select_localscan  # B, C, L
                
                # (backward)
                x_b_masked_select = x_b_rs * hard_keep_decision_mask
                x_b_masked_nonselect = x_b_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_b_masked_select_localscan = local_scan_quad_quad(x_b_masked_select, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_masked_nonselect_localscan = local_scan_quad(x_b_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_quad_window = x_b_masked_nonselect_localscan + x_b_masked_select_localscan  # B, C, L
                
                # reshape
                x = rearrange(x_quad_window, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x = torch.cat((x_cls, x), dim=-1)
                x_b = rearrange(x_b_quad_window, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x_b = torch.cat((x_b_cls, x_b), dim=-1)
                x_b = x_b.flip([-1])
                
                seqlen = 1 + (t * h * w)
                

            # bi-directional scan (forward)
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            
            # bi-directional scan (backward)
            A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
            
            x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
            dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_b.weight @ dt_b.t()
            dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y_b = selective_scan_fn(
                x_b,
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=None,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y_b = y_b.flip([-1])
            
            y = self.out_norm(y.permute(0, 2, 1))  # (b l d)
            y_b = self.out_norm_b(y_b.permute(0, 2, 1))  # (b l d)
            y += y_b
            y = y.permute(0, 2, 1)  # (b d l)
            
            y_cls = y[:, :, 0:1]
            y = y[:, :, 1:]
            curr_score = None
            if self.quad_flag:
                # reshape
                y = rearrange(y, "b d (t h w) -> (b t) d (h w)", t=t, h=h, w=w)
                
                # for quad
                y_select = local_reverse_quad_quad(y, H=int(h), W=int(w))
                y_nonselect = local_reverse_quad(y, H=int(h), W=int(w))
                y_hard_keep_decision_mask = hard_keep_decision_mask.clone()
                y_masked_select = y_select * y_hard_keep_decision_mask
                y_masked_nonselect = y_nonselect * (1.0 - y_hard_keep_decision_mask)

                y = y_masked_select + y_masked_nonselect  # B, C, L

                if diff_flag:
                    y = y.reshape(b, d, h, -1)
                    # Remove padding evenly on all four sides (top, bottom, left, and right)            
                    y = y[:, :, pad_top:-pad_bottom, pad_left:-pad_right].contiguous()
                    h, w = h - (pad_top+pad_bottom), w - (pad_left+pad_right)
                    
                else:
                    y = y.view(b, d, h, -1)

                if self.shift_flag:
                    y = torch.roll(y, shifts=reverse_size, dims=(2, 3))

                # reshape
                curr_score = torch.relu(y).mean(dim=1, keepdim=True)  # (bt, 1, h, w)
                y = rearrange(y, "(b t) d h w -> b d (t h w)", t=t, h=h, w=w)
                
            y = torch.cat((y_cls, y), dim=-1)
            
            y = y.permute(0, 2, 1)  # (b l d)
            z = z.permute(0, 2, 1)  # (b l d)
            y = y * F.silu(z)
            
            out = self.out_proj(y)
        
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            curr_score = None
            
        return out, curr_score

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state
    

class Mamba_quadtreev2_hierarchy(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
        # ========================
        token_seq=[8,14,14],
        stage_num=0,
        depth_num=0,
        block_depth=[0],
        # ========================
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True
            
            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_norm_b = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        
        # Quadtree
        self.token_seq = token_seq
        self.stage_num = stage_num
        self.depth_num = depth_num
        self.block_index = (sum(block_depth[0:stage_num]) + depth_num)if stage_num>=1 else depth_num
        self.quad_flag = False
        self.shift_flag = False
        if self.stage_num == 0 or self.stage_num == 1:
            self.quad_flag = True
            if self.depth_num % 2 == 1:
                self.shift_flag = True
                
            # Hibert curve gen.
            H, W = self.token_seq[1], self.token_seq[2]
            quad_size = int(2)
            quad_number = quad_size * 8
            if H % quad_number != 0 or W % quad_number != 0:
                # print("h % quad_number != 0 or w % quad_number != 0")
                H, W = math.ceil(H / quad_number) * quad_number, math.ceil(W / quad_number) * quad_number
            
            H //= 2
            W //= 2
            p = int(np.log2(H))  # 힐베르트 곡선의 단계 (order)
            n = 2  # 2차원

            # 힐베르트 곡선 객체 생성
            hilbert_curve = HilbertCurve(p, n)

            # 힐베르트 곡선의 전체 좌표 계산
            coords = []
            for y in range(H):
                for x in range(W):
                    coords.append((x, y))

            # 각 좌표에 대한 힐베르트 인덱스 계산
            hilbert_indices = []
            
            for coord in coords:
                x, y = coord
                # 힐베르트 곡선의 크기에 맞게 좌표 조정
                hilbert_index = hilbert_curve.distance_from_point([x, y])
                hilbert_indices.append(hilbert_index)

            # 힐베르트 인덱스에 따라 정렬
            hilbert_indices = np.array(hilbert_indices)
            self.hibert_sorted_indices = sorted_indices = np.argsort(hilbert_indices)
            # 역순서 인덱스 계산
            self.hibert_inverse_indices = np.argsort(sorted_indices)
            
            

    def forward(self, hidden_states, prev_score, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        
        if self.bimamba:
            x, z = xz.chunk(2, dim=1)
            x_b = x.flip([-1])
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )
            
            if causal_conv1d_fn is None:
                x_b = self.act(self.conv1d_b(x_b)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x_b = causal_conv1d_fn(
                    x_b,
                    rearrange(self.conv1d_b.weight, "d 1 w -> d w"),
                    self.conv1d_b.bias,
                    self.activation,
                )
            
            if self.quad_flag:
                # reshape
                b, d, l = x.shape
                t, h, w = self.token_seq
                x_cls = x[:, :, 0:1]
                x = x[:, :, 1:]
                x = x.reshape(b, d, t, h, w)
                x = rearrange(x, "b d t h w -> (b t) d h w", t=t)
                
                x_b = x_b.flip([-1]) 
                x_b_cls = x_b[:, :, 0:1]
                x_b = x_b[:, :, 1:]
                x_b = x_b.reshape(b, d, t, h, w)
                x_b = rearrange(x_b, "b d t h w -> (b t) d h w", t=t)
                
                b, d, h, w = x.shape

                quad_size = int(2)
                quad_number = quad_size * 8
                # 이전 stage에서 받아온 score를 활용
                if prev_score is None:
                    score = torch.relu(x+x_b).mean(dim=1, keepdim=True)  # (b, 1, h, w)
                else:
                    score = prev_score  # (b, 1, h, w)
                if self.shift_flag:
                    shift_size, reverse_size = shift_size_generate(self.block_index, h)

                    x = torch.roll(x, shifts=shift_size, dims=(2, 3))
                    x_b = torch.roll(x_b, shifts=shift_size, dims=(2, 3))
                
                if h % quad_number != 0 or w % quad_number != 0:
                    # print("h % quad_number != 0 or w % quad_number != 0")
                    newH, newW = math.ceil(h / quad_number) * quad_number, math.ceil(w / quad_number) * quad_number
                    diff_H, diff_W = newH - h, newW - w
                    # Apply padding evenly on all four sides (top, bottom, left, and right)            
                    pad_top = diff_H // 2
                    pad_bottom = diff_H - pad_top
                    pad_left = diff_W // 2
                    pad_right = diff_W - pad_left

                    x = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
                    x_b = F.pad(x_b, (pad_left, pad_right, pad_top, pad_bottom))
                    # score = F.pad(score, (pad_left, pad_right, pad_top, pad_bottom))

                    b, d, h, w = x.shape
                    diff_flag = True
                else:
                    diff_flag = False

                ### quad_one_stage
                x_rs = x.reshape(b, d, -1)
                x_b_rs = x_b.reshape(b, d, -1)
                score_window = F.adaptive_avg_pool2d(score[:, 0:1, :, :], (2, 2)) # b, 1, 2, 2
                # print(score_window)
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=2.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # smoothness
                # hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=1.0, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]
                hard_keep_decision = F.gumbel_softmax(score_window.view(b, 1, -1), dim=-1, tau=0.3, hard=True).unsqueeze(-1).unsqueeze(-1)  # [b, 1, 4, 1, 1]  # sharpness
                # print(hard_keep_decision)
                
                # (forward)
                hard_keep_decision_mask = window_expansion(hard_keep_decision, H=int(h), W=int(w))  # [b, 1, l]
                x_masked_select = x_rs * hard_keep_decision_mask
                x_masked_nonselect = x_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_masked_select_localscan = apply_hilbert_curve_2d_quad(x_masked_select, self.hibert_sorted_indices, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_masked_nonselect_localscan = local_scan_quad(x_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_quad_window = x_masked_nonselect_localscan + x_masked_select_localscan  # B, C, L
                # Non-ROI & ROI Region Split
                x_quad_window_split = NONROI_ROI_split(x_quad_window, hard_keep_decision, H=int(h), W=int(w))
                
                # (backward)
                x_b_masked_select = x_b_rs * hard_keep_decision_mask
                x_b_masked_nonselect = x_b_rs * (1.0 - hard_keep_decision_mask)
                # local scan quad region
                x_b_masked_select_localscan = apply_hilbert_curve_2d_quad(x_b_masked_select, self.hibert_sorted_indices, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_masked_nonselect_localscan = local_scan_quad(x_b_masked_nonselect, H=int(h), W=int(w))  # BCHW -> B, C, L
                x_b_quad_window = x_b_masked_nonselect_localscan + x_b_masked_select_localscan  # B, C, L
                # Non-ROI & ROI Region Split
                x_b_quad_window_split = NONROI_ROI_split(x_b_quad_window, hard_keep_decision, H=int(h), W=int(w))
                
                # reshape
                x = rearrange(x_quad_window_split, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x = torch.cat((x_cls, x), dim=-1)
                x_b = rearrange(x_b_quad_window_split, "(b t) d (h w) -> b d (t h w)", t=t, h=h, w=w)
                x_b = torch.cat((x_b_cls, x_b), dim=-1)
                x_b = x_b.flip([-1])
                
                seqlen = 1 + (t * h * w)
                

            # bi-directional scan (forward)
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            
            # bi-directional scan (backward)
            A_b = -torch.exp(self.A_b_log.float())  # (d_inner, d_state)
            
            x_dbl_b = self.x_proj_b(rearrange(x_b, "b d l -> (b l) d"))  # (bl d)
            dt_b, B_b, C_b = torch.split(x_dbl_b, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt_b = self.dt_proj_b.weight @ dt_b.t()
            dt_b = rearrange(dt_b, "d (b l) -> b d l", l=seqlen)
            B_b = rearrange(B_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C_b = rearrange(C_b, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y_b = selective_scan_fn(
                x_b,
                dt_b,
                A_b,
                B_b,
                C_b,
                self.D_b.float(),
                z=None,
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y_b = y_b.flip([-1])
            
            y = self.out_norm(y.permute(0, 2, 1))  # (b l d)
            y_b = self.out_norm_b(y_b.permute(0, 2, 1))  # (b l d)
            y += y_b
            y = y.permute(0, 2, 1)  # (b d l)
            
            y_cls = y[:, :, 0:1]
            y = y[:, :, 1:]
            curr_score = None
            if self.quad_flag:
                # reshape
                y = rearrange(y, "b d (t h w) -> (b t) d (h w)", t=t, h=h, w=w)
                
                # Non-ROI & ROI Region Merge
                y_hard_keep_decision = hard_keep_decision.clone()
                y = NONROI_ROI_merge(y, y_hard_keep_decision, H=int(h), W=int(w))
                
                # for quad
                y_select = reverse_hilbert_curve_2d_quad(y, self.hibert_inverse_indices, H=int(h), W=int(w))
                y_nonselect = local_reverse_quad(y, H=int(h), W=int(w))
                y_hard_keep_decision_mask = hard_keep_decision_mask.clone()
                y_masked_select = y_select * y_hard_keep_decision_mask
                y_masked_nonselect = y_nonselect * (1.0 - y_hard_keep_decision_mask)

                y = y_masked_select + y_masked_nonselect  # B, C, L

                if diff_flag:
                    y = y.reshape(b, d, h, -1)
                    # Remove padding evenly on all four sides (top, bottom, left, and right)            
                    y = y[:, :, pad_top:-pad_bottom, pad_left:-pad_right].contiguous()
                    h, w = h - (pad_top+pad_bottom), w - (pad_left+pad_right)
                    
                else:
                    y = y.view(b, d, h, -1)

                if self.shift_flag:
                    y = torch.roll(y, shifts=reverse_size, dims=(2, 3))

                # reshape
                curr_score = torch.relu(y).mean(dim=1, keepdim=True)  # (bt, 1, h, w)
                y = rearrange(y, "(b t) d h w -> b d (t h w)", t=t, h=h, w=w)
                
            y = torch.cat((y_cls, y), dim=-1)
            
            y = y.permute(0, 2, 1)  # (b l d)
            z = z.permute(0, 2, 1)  # (b l d)
            y = y * F.silu(z)
            
            out = self.out_proj(y)
        
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=None,
            )
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
            curr_score = None
            
        return out, curr_score

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba=True,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba = bimamba

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        # NOTE: why plus 1?
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        # bidirectional
        # forked from https://github.com/hustvl/Vim
        if self.bimamba:
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True 

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
            )
            self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

            self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
            self.D_b._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def forward(self, hidden_states, inference_params=None, T=1):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        # NOTE: same as in_proj(hidden_states) but memory-efficient with the following operations
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and inference_params is None:  # Doesn't support outputting the states
            if self.bimamba:
                A_b = -torch.exp(self.A_b_log.float())
                out = mamba_inner_fn_no_out_proj(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
                out_b = mamba_inner_fn_no_out_proj(
                    xz.flip([-1]),
                    self.conv1d_b.weight,
                    self.conv1d_b.bias,
                    self.x_proj_b.weight,
                    self.dt_proj_b.weight,
                    A_b,
                    None,
                    None,
                    self.D_b.float(),
                    delta_bias=self.dt_proj_b.bias.float(),
                    delta_softplus=True,
                )
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias)
            else:
                out = mamba_inner_fn(
                    xz,
                    self.conv1d.weight,
                    self.conv1d.bias,
                    self.x_proj.weight,
                    self.dt_proj.weight,
                    self.out_proj.weight,
                    self.out_proj.bias,
                    A,
                    None,  # input-dependent B
                    None,  # input-dependent C
                    self.D.float(),
                    delta_bias=self.dt_proj.bias.float(),
                    delta_softplus=True,
                )
        else:
            x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                conv_state.copy_(x[:, :, -self.d_conv :])  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x,
                    rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    self.conv1d.bias,
                    self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)
            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)
        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state


class Block(nn.Module):
    def __init__(
        self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False
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
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(
        self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None
    ):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states, inference_params=inference_params)
        return hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)

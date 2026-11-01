# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


# Inspired by https://github.com/DepthAnything/Depth-Anything-V2


import os
from typing import List, Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from .head_act import activate_head
from .utils import create_uv_grid, position_grid_to_embed


class BGHead(nn.Module):
    """
    DPT  Head for dense prediction tasks.

    This implementation follows the architecture described in "Vision Transformers for Dense Prediction"
    (https://arxiv.org/abs/2103.13413). The DPT head processes features from a vision transformer
    backbone and produces dense predictions by fusing multi-scale features.

    Args:
        dim_in (int): Input dimension (channels).
        patch_size (int, optional): Patch size. Default is 14.
        output_dim (int, optional): Number of output channels. Default is 4.
        activation (str, optional): Activation type. Default is "inv_log".
        conf_activation (str, optional): Confidence activation type. Default is "expp1".
        features (int, optional): Feature channels for intermediate representations. Default is 256.
        out_channels (List[int], optional): Output channels for each intermediate layer.
        intermediate_layer_idx (List[int], optional): Indices of layers from aggregated tokens used for DPT.
        pos_embed (bool, optional): Whether to use positional embedding. Default is True.
        feature_only (bool, optional): If True, return features only without the last several layers and activation head. Default is False.
        down_ratio (int, optional): Downscaling factor for the output resolution. Default is 1.
    """

    def __init__(
        self,
        dim_in: int,
        patch_size: int = 14, 
        output_dim: int = 4,
        activation: str = "inv_log",
        conf_activation: str = "expp1",
        features: int = 256,
        out_channels: List[int] = [256, 512, 1024, 1024],
        intermediate_layer_idx: List[int] = [4, 11, 17, 23],
        pos_embed: bool = True,
        feature_only: bool = False,
        down_ratio: int = 1,
        return_conf: bool = True,
    ) -> None:
        super(BGHead, self).__init__()
        self.patch_size = patch_size
        self.activation = activation
        self.conf_activation = conf_activation
        self.pos_embed = pos_embed
        self.feature_only = feature_only
        self.down_ratio = down_ratio
        self.intermediate_layer_idx = intermediate_layer_idx
        self.return_conf = return_conf

        self.norm = nn.LayerNorm(dim_in)

        # Projection layers for each output channel from tokens.
        self.projects = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=dim_in,
                    out_channels=oc,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                )
                for oc in out_channels
            ]
        )

        # Resize layers for upsampling feature maps.
        self.resize_layers = nn.ModuleList(
            [
                nn.Identity(),
                nn.Identity(),
                # nn.ConvTranspose2d(
                #     in_channels=out_channels[0], out_channels=out_channels[0], kernel_size=4, stride=4, padding=0
                # ),
                # nn.ConvTranspose2d(
                #     in_channels=out_channels[1], out_channels=out_channels[1], kernel_size=2, stride=2, padding=0
                # ),
                nn.Identity(),
                nn.Identity(),
                # nn.Conv2d(
                #     in_channels=out_channels[3], out_channels=out_channels[3], kernel_size=3, stride=2, padding=1
                # ),
            ]
        )

        self.scratch = _make_scratch(
            out_channels,
            features,
            expand=False,
        )

        # Attach additional modules to scratch.
        self.scratch.stem_transpose = None
        self.scratch.refinenet1 = _make_fusion_block(features)
        self.scratch.refinenet2 = _make_fusion_block(features)
        self.scratch.refinenet3 = _make_fusion_block(features)
        self.scratch.refinenet4 = _make_fusion_block(features, has_residual=False)

        head_features_1 = features
        head_features_2 = 32
        
        if feature_only:
            self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1, kernel_size=3, stride=1, padding=1)
        else:
            self.scratch.output_conv1 = nn.Conv2d(
                head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1
            )
            conv2_in_channels = head_features_1 // 2

            self.scratch.output_conv2 = nn.Sequential(
                nn.Conv2d(conv2_in_channels, head_features_2, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            )

        self._pos_cache = {}  # {(device, dtype, C, ph, pw, W, H, ratio): tensor}

        # Initialize final BGHead layer to zeros or identity
        if not feature_only:
            assert output_dim % 13 == 0, f"output_dim={output_dim} must be a multiple of 13"
            self.n_bins = output_dim // 13
            # self.init_bghead_zero()
            # self.init_bghead_identity()

    def forward(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_chunk_size: int = 8,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the DPT head, supports processing by chunking frames.
        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W], in range [0, 1].
            patch_start_idx (int): Starting index for patch tokens in the token sequence.
                Used to separate patch tokens from other tokens (e.g., camera or register tokens).
            frames_chunk_size (int, optional): Number of frames to process in each chunk.
                If None or larger than S, all frames are processed at once. Default: 8.

        Returns:
            Tensor or Tuple[Tensor, Tensor]:
                - If feature_only=True: Feature maps with shape [B, S, C, H, W]
                - Otherwise: Tuple of (predictions, confidence) both with shape [B, S, 1, H, W]
        """
        B, S, _, H, W = images.shape

        # If frames_chunk_size is not specified or greater than S, process all frames at once
        if frames_chunk_size is None or frames_chunk_size >= S:
            return self._forward_impl(aggregated_tokens_list, images, patch_start_idx)

        # Otherwise, process frames in chunks to manage memory usage
        assert frames_chunk_size > 0

        # Process frames in batches
        all_preds = []
        all_conf = []
        
        for frames_start_idx in range(0, S, frames_chunk_size):
            frames_end_idx = min(frames_start_idx + frames_chunk_size, S)

            # Process batch of frames
            if self.feature_only:
                chunk_output = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_output)
            else:
                chunk_preds, chunk_conf = self._forward_impl(
                    aggregated_tokens_list, images, patch_start_idx, frames_start_idx, frames_end_idx
                )
                all_preds.append(chunk_preds)
                all_conf.append(chunk_conf)
        
        # Concatenate results along the sequence dimension
        if self.feature_only:
            return torch.cat(all_preds, dim=1)
        else:
            return torch.cat(all_preds, dim=1), torch.cat(all_conf, dim=1)

    def _forward_impl(
        self,
        aggregated_tokens_list: List[torch.Tensor],
        images: torch.Tensor,
        patch_start_idx: int,
        frames_start_idx: int = None,
        frames_end_idx: int = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Implementation of the forward pass through the DPT head.

        This method processes a specific chunk of frames from the sequence.

        Args:
            aggregated_tokens_list (List[Tensor]): List of token tensors from different transformer layers.
            images (Tensor): Input images with shape [B, S, 3, H, W].
            patch_start_idx (int): Starting index for patch tokens.
            frames_start_idx (int, optional): Starting index for frames to process.
            frames_end_idx (int, optional): Ending index for frames to process.

        Returns:
            Tensor or Tuple[Tensor, Tensor]: Feature maps or (predictions, confidence).
        """
        if frames_start_idx is not None and frames_end_idx is not None:
            images = images[:, frames_start_idx:frames_end_idx]

        B, S, _, H, W = images.shape

        patch_h, patch_w = H // self.patch_size, W // self.patch_size

        out = []
        dpt_idx = 0
        
        for layer_idx in self.intermediate_layer_idx:
            # x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            if len(aggregated_tokens_list) > 10:
                x = aggregated_tokens_list[layer_idx][:, :, patch_start_idx:]
            else:
                list_idx = self.intermediate_layer_idx.index(layer_idx)
                x = aggregated_tokens_list[list_idx][:, :, patch_start_idx:]

            # Select frames if processing a chunk
            if frames_start_idx is not None and frames_end_idx is not None:
                x = x[:, frames_start_idx:frames_end_idx].contiguous()

            x = x.view(B * S, -1, x.shape[-1])
            x = self.norm(x)
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))

            x = self.projects[dpt_idx](x)
            if self.pos_embed:
                x = self._apply_pos_embed(x, W, H)

            x = self.resize_layers[dpt_idx](x)
            
            out.append(x)
            dpt_idx += 1

        # Fuse features from multiple layers.
        out = self.scratch_forward(out)
        # Interpolate fused output to match target image resolution.
        out = custom_interpolate(
            out,
            (16,16),
            #(int(patch_h * self.patch_size / self.down_ratio), int(patch_w * self.patch_size / self.down_ratio)),
            mode="bilinear",
            align_corners=True,
        )

        if self.pos_embed:
            out = self._apply_pos_embed(out, W, H)

        if self.feature_only:
            return out.view(B, S, *out.shape[1:])

        out = self.scratch.output_conv2(out)
        preds, conf = activate_head(out, activation=self.activation, conf_activation=self.conf_activation, conf_dim=self.n_bins)

        preds = preds.view(B, S, *preds.shape[1:])
        conf = conf.view(B, S, *conf.shape[1:])
        if self.return_conf:
            return preds, conf
        else:
            return preds, None
    
    @torch.no_grad()
    def _get_pos_embed_cached(
        self, *, C: int, ph: int, pw: int, W: int, H: int, dtype: torch.dtype, device: torch.device, ratio: float
    ) -> torch.Tensor:
        """
        Returns a cached [1, C, ph, pw] positional embedding tensor on (device, dtype),
        computed using your create_uv_grid → position_grid_to_embed pipeline, scaled by `ratio`.
        """
        # Keyed by everything that affects the content & layout
        key = (device, dtype, C, ph, pw, W, H, float(ratio))
        pe = self._pos_cache.get(key)
        if pe is not None:
            return pe

        # Build once, store detached (no grad) in the exact dtype/device requested
        uv = create_uv_grid(pw, ph, aspect_ratio=W / H, dtype=dtype, device=device)  # [ph, pw, 2]
        pe = position_grid_to_embed(uv, C) * ratio                                    # [ph, pw, C]
        pe = pe.permute(2, 0, 1).unsqueeze(0).contiguous()                            # [1, C, ph, pw]
        self._pos_cache[key] = pe
        return pe
    
    def _apply_pos_embed_original(self, x, W, H, ratio=0.1):
        ph, pw = x.shape[-2], x.shape[-1]
        pe = create_uv_grid(pw, ph, aspect_ratio=W/H, dtype=x.dtype, device=x.device)   # [ph,pw,2]
        pe = position_grid_to_embed(pe, x.shape[1]) * ratio                              # [ph,pw,C]
        pe = pe.permute(2,0,1)[None].expand(x.shape[0], -1, -1, -1).contiguous()        # [N,C,ph,pw]
        return x + pe

    def _apply_pos_embed(self, x: torch.Tensor, W: int, H: int, ratio: float = 0.1) -> torch.Tensor:
        """
        Apply positional embedding to tensor x.
        """
        # patch_w = x.shape[-1]
        # patch_h = x.shape[-2]
        # pos_embed = create_uv_grid(patch_w, patch_h, aspect_ratio=W / H, dtype=x.dtype, device=x.device)
        # pos_embed = position_grid_to_embed(pos_embed, x.shape[1])
        # pos_embed = pos_embed * ratio
        # pos_embed = pos_embed.permute(2, 0, 1)[None].expand(x.shape[0], -1, -1, -1)
        # return x + pos_embed
        pe = self._get_pos_embed_cached(
            C=x.shape[1], ph=x.shape[-2], pw=x.shape[-1],
            W=W, H=H, dtype=x.dtype, device=x.device, ratio=ratio)
        return x + pe.expand(x.shape[0], -1, -1, -1)

    def scratch_forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the fusion blocks.

        Args:
            features (List[Tensor]): List of feature maps from different layers.

        Returns:
            Tensor: Fused feature map.
        """
        layer_1, layer_2, layer_3, layer_4 = features

        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        out = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        del layer_4_rn, layer_4

        out = self.scratch.refinenet3(out, layer_3_rn, size=layer_2_rn.shape[2:])
        del layer_3_rn, layer_3

        out = self.scratch.refinenet2(out, layer_2_rn, size=layer_1_rn.shape[2:])
        del layer_2_rn, layer_2

        # out = self.scratch.refinenet1(out, layer_1_rn)
        out = self.scratch.refinenet1(out, layer_1_rn, size=layer_1_rn.shape[2:])
        del layer_1_rn, layer_1

        out = self.scratch.output_conv1(out)
        return out
    
    @torch.no_grad()
    def init_bghead_zero(self):
        last = self.scratch.output_conv2[-1]
        assert isinstance(last, nn.Conv2d)

        out_ch = last.out_channels
        assert out_ch % 13 == 0, f"out_channels={out_ch} must be a multiple of 13"
        n_bins = out_ch // 13
        self.n_bins = n_bins

        last.weight.zero_()
        last.bias.zero_()
    
    @torch.no_grad()
    def init_bghead_identity(
        self,
        tanh_target: float = 0.99,      # only used if self.activation == "tanh"
        conf_bias_sigmoid: float = -2.0 # if self.conf_activation == "sigmoid" → ~0.12 alpha at start
        ):
        """
        Identity-init the final conv so post-activation params form an identity 3x4 per bin.
        Assumes output_dim = 13 * n_bins = (12 params + 1 conf) * n_bins.
        Supports:
        self.activation in {"linear","inv_log","tanh"}
        self.conf_activation in {"sigmoid","expp1","expp0"}
        """
        last = self.scratch.output_conv2[-1]
        assert isinstance(last, nn.Conv2d) # final layer must be Conv2d

        out_ch = last.out_channels
        assert out_ch % 13 == 0, f"out_channels={out_ch} must be a multiple of 13"
        n_bins = out_ch // 13
        self.n_bins = n_bins

        # Target post-activation identity vector per bin (row-major 3x4):
        # [1,0,0,0, 0,1,0,0, 0,0,1,0]
        id_row = [1,0,0,0,  0,1,0,0,  0,0,1,0]
        target_affine = id_row * n_bins  # length 12*n_bins

        act = (self.activation or "linear").lower()
        conf_act = (self.conf_activation or "sigmoid").lower()

        # Map target (post-activation) → pre-activation bias for params (length 12*n_bins)
        if act == "linear":
            pre_affine = target_affine
        elif act == "inv_log":
            # inverse of inverse_log_transform: pre = sign(t) * log(|t| + 1)
            pre_affine = []
            for t in target_affine:
                if t == 0:
                    pre_affine.append(0.0)
                else:
                    pre_affine.append(float(torch.log(torch.tensor(abs(t) + 1.0))))
        elif act == "tanh":
            # Can’t hit 1 exactly; choose a close target and use atanh(t)
            assert 0.0 < tanh_target < 1.0, "tanh_target must be in (0,1)"
            M = float(torch.atanh(torch.tensor(tanh_target)))
            pre_affine = [M if t == 1 else 0.0 for t in target_affine]
        else:
            raise ValueError(f"Unsupported activation '{self.activation}' for identity init")

        # Confidence bias: 1 per bin (length n_bins)
        if conf_act == "sigmoid":
            conf_bias = [float(conf_bias_sigmoid)] * n_bins   # small alpha at start
        elif conf_act in ("expp1", "expp0"):
            conf_bias = [0.0] * n_bins                        # keep magnitudes tame
        else:
            raise ValueError(f"Unsupported conf_activation '{self.conf_activation}'")

        with torch.no_grad():
            last.weight.zero_()  # gate off feature dependence at init
            bias = torch.tensor(pre_affine + conf_bias, dtype=last.bias.dtype, device=last.bias.device)
            last.bias.copy_(bias)


################################################################################
# Modules
################################################################################


def _make_fusion_block(features: int, size: int = None, has_residual: bool = True, groups: int = 1) -> nn.Module:
    return FeatureFusionBlock(
        features,
        nn.ReLU(inplace=True),
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=size,
        has_residual=has_residual,
        groups=groups,
    )


def _make_scratch(in_shape: List[int], out_shape: int, groups: int = 1, expand: bool = False) -> nn.Module:
    scratch = nn.Module()
    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        if len(in_shape) >= 4:
            out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )
    return scratch


class ResidualConvUnit(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn, groups=1):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn
        self.groups = groups
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups)

        self.norm1 = None
        self.norm2 = None

        self.activation = activation
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.norm1 is not None:
            out = self.norm1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.norm2 is not None:
            out = self.norm2(out)

        return self.skip_add.add(out, x)


class FeatureFusionBlock(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        size=None,
        has_residual=True,
        groups=1,
    ):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners
        self.groups = groups
        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=self.groups
        )

        if has_residual:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.has_residual = has_residual
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn, groups=self.groups)

        self.skip_add = nn.quantized.FloatFunctional()
        self.size = size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if self.has_residual:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = custom_interpolate(output, **modifier, mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)

        return output


def custom_interpolate(
    x: torch.Tensor,
    size: Tuple[int, int] = None,
    scale_factor: float = None,
    mode: str = "bilinear",
    align_corners: bool = True,
) -> torch.Tensor:
    """
    Custom interpolate to avoid INT_MAX issues in nn.functional.interpolate.
    """
    if size is None:
        size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
    
    INT_MAX = 1610612736

    input_elements = size[0] * size[1] * x.shape[0] * x.shape[1]

    if input_elements > INT_MAX:
        chunks = torch.chunk(x, chunks=(input_elements // INT_MAX) + 1, dim=0)
        interpolated_chunks = [
            nn.functional.interpolate(chunk, size=size, mode=mode, align_corners=align_corners) for chunk in chunks
        ]
        x = torch.cat(interpolated_chunks, dim=0)
        return x.contiguous()
    else:
        return nn.functional.interpolate(x, size=size, mode=mode, align_corners=align_corners)

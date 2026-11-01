import copy

# VGGT parts
import os
import sys
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Literal, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float
from src.dataset.shims.normalize_shim import apply_normalize_shim
from src.dataset.types import BatchedExample, DataShim

from src.model.encoder.vggt.utils.geometry import (
    batchify_unproject_depth_map_to_point_map,
    unproject_depth_map_to_point_map,
)
from src.model.encoder.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from torch import nn, Tensor
from torch_scatter import scatter_add, scatter_max

from ..types import Gaussians
BackboneCfg = Any  # Legacy configuration typing; the Omega inference path does not construct CroCo.

from .common.gaussian_adapter import (
    GaussianAdapter,
    GaussianAdapterCfg,
    UnifiedGaussianAdapter,
)
from .encoder import Encoder, EncoderOutput
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from src.model.encoder.vggt.heads.bg_head import BGHead

try:
    from vggt_omega.models import VGGTOmega
    from src.model.encoder.heads.omega_gs_head import VGGTOmega_Dense_GS_Head
except Exception:
    VGGTOmega = None

inf = float("inf")
dtype_ = (torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16) if torch.cuda.is_available() else torch.float32

@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class GSHeadParams:
    dec_depth: int = 23
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 2048
    dec_embed_dim: int = 2048
    feature_dim: int = 256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True
    predict_static: bool = False
    predict_uncertainty: bool = False

@dataclass
class BGHeadParams:
    patch_size: tuple[int, int] = (14, 14)
    enc_embed_dim: int = 512 #2048
    dec_embed_dim: int = 512 #2048
    feature_dim: int = 64 #256
    depth_mode = ("exp", -inf, inf)
    conf_mode = True


@dataclass
class EncoderAnySplatCfg:
    name: Literal["anysplat"]
    anchor_feat_dim: int
    voxel_size: float
    n_offsets: int
    d_feature: int
    add_view: bool
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    pred_pose: bool = True
    gt_pose_to_pts: bool = False
    gs_prune: bool = False
    gs_prune_mode: str = "batchwise_pad"
    opacity_threshold: float = 0.001
    gs_keep_ratio: float = 1.0
    pred_head_type: Literal["depth", "point"] = "point"
    freeze_backbone: bool = False
    freeze_backbone_until_iter: int = 0
    freeze_module: Literal[
        "all",
        "global",
        "frame",
        "patch_embed",
        "patch_embed+frame",
        "patch_embed+global",
        "global+frame",
        "None",
    ] = "None"
    distill: bool = False
    offload_distill_to_cpu: bool = False
    render_conf: bool = False
    opacity_conf: bool = False
    conf_threshold: float = 0.1
    intermediate_layer_idx: Optional[List[int]] = None
    voxelize: bool = False
    bilateral_grid: bool = False
    predict_static: bool = False
    predict_uncertainty: bool = False
    fusion_mode: Literal["anysplat", "static", "multiscale"] = "anysplat"
    vggt_variant: Literal["vggt", "vggt_omega"] = "vggt"
    vggt_omega_checkpoint: str = ""
    gaussian_refiner: bool = False
    gaussian_attention_refiner: bool = False


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat

def softcap_logvar(logvar, lo=-9.0, hi=1.0, softness=1.0):
    x = torch.clamp(logvar, min=lo)
    # keep x when below hi; softly bend when above hi
    return x - softness * F.softplus((x - hi) / softness) + softness * F.softplus(torch.zeros((), device=x.device))

def clamp_logvar(logvar, lo=-9.0, hi=1.0):
    return logvar.clamp(lo, hi)

def logvar_to_var(logvar, eps=1e-6):
    return torch.exp(clamp_logvar(logvar)).clamp_min(eps)

def var_to_logvar(var, eps=1e-6):
    return torch.log(var.clamp_min(eps))

def _topk_per_group(score, group, k):
    """
    score: [N]
    group: [N] int64, values in [0, G)
    returns keep mask [N]
    """
    assert score.ndim == 1
    assert group.ndim == 1
    assert score.shape[0] == group.shape[0]

    N = score.shape[0]
    if N == 0:
        return torch.zeros_like(score, dtype=torch.bool)

    group = group.long()
    num_groups = int(group.max().item()) + 1

    keep = torch.zeros(N, device=score.device, dtype=torch.bool)
    score_work = score.clone()

    for _ in range(k):
        best_score, best_idx = scatter_max(
            score_work,
            group,
            dim=0,
            dim_size=num_groups,
        )

        valid = (
            (best_idx >= 0)
            & (best_idx < N)
            & torch.isfinite(best_score)
        )

        if not valid.any():
            break

        chosen = best_idx[valid].long()
        keep[chosen] = True
        score_work[chosen] = -torch.inf

    return keep


class GaussianResidualRefiner(nn.Module):
    def __init__(self, feat_dim, hidden_dim=256, max_xyz_offset=0.002):
        super().__init__()
        print("[GaussianResidualRefiner] feat_dim", feat_dim, "hidden_dim", hidden_dim, "max_xyz_offset", max_xyz_offset)
        self.feat_dim = feat_dim
        self.max_xyz_offset = max_xyz_offset

        self.net = nn.Sequential(
            nn.LayerNorm(feat_dim),
            nn.Linear(feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.delta_feat = nn.Linear(hidden_dim, feat_dim)
        self.delta_xyz = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.delta_feat.weight)
        nn.init.zeros_(self.delta_feat.bias)
        nn.init.zeros_(self.delta_xyz.weight)
        nn.init.zeros_(self.delta_xyz.bias)

    def forward(self, pts, feats):
        h = self.net(feats)

        d_feat = self.delta_feat(h)
        d_xyz = torch.tanh(self.delta_xyz(h)) * self.max_xyz_offset

        return pts + d_xyz, feats + d_feat


class KNNGeometricAttentionRefiner(nn.Module):
    def __init__(
        self,
        feat_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        k: int = 16,
        max_xyz_offset: float = 0.0002,
        refine_density: bool = False,
    ):
        super().__init__()
        print("[KNNGeometricAttentionRefiner] feat_dim", feat_dim, "hidden_dim", hidden_dim, "max_xyz_offset", max_xyz_offset)

        from knn_cuda import KNN

        assert hidden_dim % num_heads == 0

        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.k = k
        self.max_xyz_offset = max_xyz_offset
        self.refine_density = refine_density

        self.knn = KNN(k=k, transpose_mode=True)

        self.q_proj = nn.Linear(feat_dim, hidden_dim)
        self.k_proj = nn.Linear(feat_dim + 3, hidden_dim)
        self.v_proj = nn.Linear(feat_dim + 3, hidden_dim)

        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.delta_feat = nn.Linear(hidden_dim, feat_dim)
        self.delta_xyz = nn.Linear(hidden_dim, 3)

        nn.init.zeros_(self.delta_feat.weight)
        nn.init.zeros_(self.delta_feat.bias)
        nn.init.zeros_(self.delta_xyz.weight)
        nn.init.zeros_(self.delta_xyz.bias)

    @torch.no_grad()
    def _knn_indices(self, pts):
        """
        pts: [N, 3]
        returns idx: [N, K]
        """
        ref = pts.unsqueeze(0).contiguous()    # [1, N, 3]
        query = pts.unsqueeze(0).contiguous()  # [1, N, 3]

        dist, idx = self.knn(ref, query)       # idx: [1, N, K]
        return idx[0].long()

    def forward(self, pts, feats):
        """
        pts:   [N, 3]
        feats: [N, C]
        """
        assert pts.ndim == 2 and pts.shape[-1] == 3
        assert feats.ndim == 2 and feats.shape[-1] == self.feat_dim

        N = pts.shape[0]
        if N <= 1:
            return pts, feats

        idx = self._knn_indices(pts)  # [N, K]

        neigh_pts = pts[idx]          # [N, K, 3]
        neigh_feats = feats[idx]      # [N, K, C]

        rel = neigh_pts - pts[:, None, :]
        rel = rel / (rel.norm(dim=-1, keepdim=True).mean().detach() + 1e-6)

        q = self.q_proj(feats)                         # [N, H]
        kv_in = torch.cat([neigh_feats, rel], dim=-1)  # [N, K, C+3]
        k = self.k_proj(kv_in)                         # [N, K, H]
        v = self.v_proj(kv_in)                         # [N, K, H]

        head_dim = self.hidden_dim // self.num_heads

        q = q.view(N, self.num_heads, head_dim)
        k = k.view(N, self.k, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(N, self.k, self.num_heads, head_dim).transpose(1, 2)

        attn_logits = (q[:, :, None, :] * k).sum(-1) / (head_dim ** 0.5)
        attn = torch.softmax(attn_logits, dim=-1)

        h = (attn[..., None] * v).sum(dim=2)
        h = h.reshape(N, self.hidden_dim)
        h = self.out(h)

        d_feat = self.delta_feat(h)
        d_xyz = torch.tanh(self.delta_xyz(h)) * self.max_xyz_offset

        out_feats = feats + d_feat
        if not self.refine_density:
            out_feats[:, 0] = feats[:, 0]

        return pts + d_xyz, out_feats


# class LocalGeometricAttentionRefiner(nn.Module):
#     def __init__(
#         self,
#         feat_dim: int,
#         hidden_dim: int = 256,
#         num_heads: int = 4,
#         max_xyz_offset: float = 0.0005,
#         cell_size: float = 0.006,
#         max_points_per_cell: int = 256,
#         refine_density: bool = False,
#     ):
#         super().__init__()
#         print("[LocalGeometricAttentionRefiner] feat_dim", feat_dim, "hidden_dim", hidden_dim, "max_xyz_offset", max_xyz_offset)
#         self.feat_dim = feat_dim
#         self.hidden_dim = hidden_dim
#         self.max_xyz_offset = max_xyz_offset
#         self.cell_size = cell_size
#         self.max_points_per_cell = max_points_per_cell
#         self.refine_density = refine_density

#         # feat + xyz relative to local cell center + density scalar
#         self.in_proj = nn.Sequential(
#             nn.LayerNorm(feat_dim + 4),
#             nn.Linear(feat_dim + 4, hidden_dim),
#             nn.GELU(),
#         )

#         self.attn = nn.MultiheadAttention(
#             embed_dim=hidden_dim,
#             num_heads=num_heads,
#             batch_first=True,
#         )

#         self.ffn = nn.Sequential(
#             nn.LayerNorm(hidden_dim),
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.GELU(),
#             nn.Linear(hidden_dim * 4, hidden_dim),
#         )

#         self.delta_feat = nn.Linear(hidden_dim, feat_dim)
#         self.delta_xyz = nn.Linear(hidden_dim, 3)

#         # Zero-init = starts as exact no-op.
#         nn.init.zeros_(self.delta_feat.weight)
#         nn.init.zeros_(self.delta_feat.bias)
#         nn.init.zeros_(self.delta_xyz.weight)
#         nn.init.zeros_(self.delta_xyz.bias)

#     def forward(self, pts, feats):
#         """
#         pts:   [N, 3]
#         feats: [N, C] where C = raw_gs_dim
#         returns refined_pts, refined_feats
#         """
#         assert pts.ndim == 2 and pts.shape[-1] == 3
#         assert feats.ndim == 2 and feats.shape[-1] == self.feat_dim

#         N = pts.shape[0]
#         if N == 0:
#             return pts, feats

#         device = pts.device
#         dtype = pts.dtype

#         # Group Gaussians into coarse spatial cells.
#         cell_idx = torch.floor(pts / self.cell_size).long()
#         unique_cells, inverse = torch.unique(
#             cell_idx,
#             dim=0,
#             return_inverse=True,
#         )

#         refined_pts = pts.clone()
#         refined_feats = feats.clone()

#         # Process each local cell independently.
#         # This is intentionally simple and safe; optimize later if it helps.
#         for cell_id in range(unique_cells.shape[0]):
#             idx = torch.nonzero(inverse == cell_id, as_tuple=False).squeeze(-1)

#             if idx.numel() <= 1:
#                 continue

#             # Cap huge cells to avoid quadratic attention blowup.
#             if idx.numel() > self.max_points_per_cell:
#                 # Prefer high-opacity / high-density candidates.
#                 density_score = feats[idx, 0]
#                 keep_local = torch.topk(
#                     density_score,
#                     k=self.max_points_per_cell,
#                     largest=True,
#                 ).indices
#                 idx = idx[keep_local]

#             p = pts[idx]       # [M, 3]
#             f = feats[idx]     # [M, C]

#             center = p.mean(dim=0, keepdim=True)
#             rel = (p - center) / max(self.cell_size, 1e-8)

#             density = f[:, :1]
#             x = torch.cat([f, rel, density], dim=-1)

#             h = self.in_proj(x).unsqueeze(0)  # [1, M, hidden]

#             h_attn, _ = self.attn(h, h, h, need_weights=False)
#             h = h + h_attn
#             h = h + self.ffn(h)
#             h = h.squeeze(0)

#             d_feat = self.delta_feat(h)
#             d_xyz = torch.tanh(self.delta_xyz(h)) * self.max_xyz_offset

#             f_new = f + d_feat

#             if not self.refine_density:
#                 f_new[:, 0] = f[:, 0]

#             refined_pts[idx] = p + d_xyz
#             refined_feats[idx] = f_new

#         return refined_pts, refined_feats


class EncoderAnySplat(Encoder[EncoderAnySplatCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderAnySplatCfg) -> None:
        super().__init__(cfg)

        self.vggt_variant = cfg.vggt_variant

        if self.vggt_variant == "vggt":
            raise ValueError("This public inference-only package supports vggt_omega checkpoints only.")

        elif self.vggt_variant == "vggt_omega":
            if VGGTOmega is None:
                raise ImportError(
                    "vggt_omega is not installed. Clone facebookresearch/vggt-omega "
                    "and run `pip install -e .`."
                )

            if self.cfg.pred_head_type != "depth":
                raise ValueError("vggt_omega currently supports pred_head_type='depth' only in this adapter.")

            # A full WildAnySplat checkpoint contains all Omega backbone weights.
            # The optional Omega initializer is only needed for partial checkpoints.
            model_full = VGGTOmega().eval()
            if cfg.vggt_omega_checkpoint:
                print(f"Initializing VGGT-Omega from {cfg.vggt_omega_checkpoint}", flush=True)
                state = torch.load(cfg.vggt_omega_checkpoint, map_location="cpu", weights_only=True)
                model_full.load_state_dict(state, strict=True)

            self.aggregator = model_full.aggregator.to(dtype_)
            self.camera_head = model_full.camera_head
            self.depth_head = model_full.dense_head
            self.point_head = None
            self.vggt_patch_size = 16

        else:
            raise ValueError(f"Unknown vggt_variant: {self.vggt_variant}")

        self.freeze_backbone = cfg.freeze_backbone
        self.freeze_backbone_until_iter = getattr(cfg, "freeze_backbone_until_iter", 0)
        self.backbone_unfrozen = False
        self.distill = cfg.distill
        self.pred_pose = cfg.pred_pose

        if self.distill:
            self.distill_aggregator = copy.deepcopy(self.aggregator)
            self.distill_camera_head = copy.deepcopy(self.camera_head)
            self.distill_depth_head = copy.deepcopy(self.depth_head)
            for module in [
                self.distill_aggregator,
                self.distill_camera_head,
                self.distill_depth_head,
            ]:
                for param in module.parameters():
                    param.requires_grad = False
                    if self.cfg.offload_distill_to_cpu:
                        param.data = param.data.cpu()

        # freeze at the beginning if needed
        if self.freeze_backbone or self.freeze_backbone_until_iter > 0:
            self.set_backbone_trainable(False)
        else:
            self.apply_partial_freeze()

        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity
        self.voxel_size = cfg.voxel_size
        self.gs_params_head_type = cfg.gs_params_head_type
        # fake backbone for head parameters
        head_params = GSHeadParams()
        self.predict_static = cfg.predict_static
        self.predict_uncertainty = cfg.predict_uncertainty
        extra_dim = int(self.predict_static) + int(self.predict_uncertainty)
        if self.vggt_variant == "vggt":
            self.gaussian_param_head = VGGT_DPT_GS_Head(
                dim_in=2048,
                patch_size=(self.vggt_patch_size, self.vggt_patch_size),
                output_dim=self.raw_gs_dim + 1  + extra_dim,  # +conf +optional static +optional var
                gaussian_d_in=self.gaussian_adapter.d_in,
                activation="norm_exp",
                conf_activation="expp1",
                features=head_params.feature_dim,
                predict_static=self.predict_static,
                predict_uncertainty=self.predict_uncertainty,
                use_rgb_init_path=False,   # False keeps the old behavior
            )
        else:
            self.gaussian_param_head = VGGTOmega_Dense_GS_Head(
                dim_in=2048,
                patch_size=self.vggt_patch_size,
                output_dim=self.raw_gs_dim + 1  + extra_dim,  # +conf +optional static +optional var
                gaussian_d_in=self.gaussian_adapter.d_in,
                activation="norm_exp",
                conf_activation="expp1",
                features=head_params.feature_dim,
                predict_static=self.predict_static,
                predict_uncertainty=self.predict_uncertainty,
            )

        if self.cfg.bilateral_grid:
            print("[EncoderAnySplat] init BGHead")
            bg_head_params = BGHeadParams()
            self.bg_param_head = BGHead(
                dim_in=2048,
                # patch_size=bg_head_params.patch_size[0],
                patch_size=self.vggt_patch_size,
                output_dim=8*12+8*1,
                activation="linear",
                conf_activation="expp1",
                features=bg_head_params.feature_dim,
                out_channels=[32, 64, 128, 128],
                return_conf=True,
            )

        self.register_buffer(
            "voxel_offsets",
            torch.tensor([[0,0,0],[1,0,0],[0,1,0],[1,1,0],[0,0,1],[1,0,1],[0,1,1],[1,1,1]], dtype=torch.int32),
            persistent=False,
        )

        if self.cfg.gaussian_refiner and self.cfg.gaussian_attention_refiner:
            raise ValueError("Use either gaussian_refiner or gaussian_attention_refiner, not both.")

        if getattr(self.cfg, "gaussian_refiner", False):
            print("[EncoderAnySplat] init GaussianResidualRefiner")
            self.gaussian_refiner = GaussianResidualRefiner(
                feat_dim=self.raw_gs_dim,
                hidden_dim=256,
                max_xyz_offset=0.25 * self.cfg.voxel_size,
            )

        if getattr(self.cfg, "gaussian_attention_refiner", False):
            print("[EncoderAnySplat] init KNNGeometricAttentionRefiner")
            self.gaussian_attention_refiner = KNNGeometricAttentionRefiner(
                feat_dim=self.raw_gs_dim,
                hidden_dim=128,
                num_heads=4,
                k=16,
                max_xyz_offset=0.1 * self.cfg.voxel_size,
                refine_density=False,
            )
            # print("[EncoderAnySplat] init LocalGeometricAttentionRefiner")
            # self.gaussian_attention_refiner = LocalGeometricAttentionRefiner(
            #     feat_dim=self.raw_gs_dim,
            #     hidden_dim=256,
            #     num_heads=4,
            #     max_xyz_offset=0.25 * self.cfg.voxel_size,
            #     cell_size=3.0 * self.cfg.voxel_size,
            #     max_points_per_cell=256,
            #     refine_density=False,
            # )


    def _run_aggregator(self, images, visualization_dump=None):
        if self.vggt_variant == "vggt":
            if visualization_dump is not None:
                return self.aggregator(
                    images,
                    intermediate_layer_idx=self.cfg.intermediate_layer_idx,
                    visualization_dump=visualization_dump,
                )
            return self.aggregator(
                images,
                intermediate_layer_idx=self.cfg.intermediate_layer_idx,
            )

        if self.vggt_variant == "vggt_omega":
            if visualization_dump is not None:
                print("[vggt_omega] visualization_dump is ignored", flush=True)
            return self.aggregator(images)

        raise ValueError(f"Unknown vggt_variant: {self.vggt_variant}")


    def _run_camera_head(self, aggregated_tokens_list, patch_start_idx):
        if self.vggt_variant == "vggt":
            pred_pose_enc_list = self.camera_head(aggregated_tokens_list)
            return pred_pose_enc_list, pred_pose_enc_list[-1]

        pose_enc = self.camera_head(
            aggregated_tokens_list,
            patch_token_start=patch_start_idx,
        )
        return [pose_enc], pose_enc


    def _run_depth_head(self, aggregated_tokens_list, image, patch_start_idx):
        if self.vggt_variant == "vggt":
            return self.depth_head(
                aggregated_tokens_list,
                images=image,
                patch_start_idx=patch_start_idx,
            )

        return self.depth_head(
            aggregated_tokens_list,
            images=image,
            patch_token_start=patch_start_idx,
        )

    def get_backbone_modules(self):
        if self.cfg.pred_head_type == "depth":
            return [self.aggregator, self.camera_head, self.depth_head]
        else:
            return [self.aggregator, self.camera_head, self.point_head]

    def set_backbone_trainable(self, trainable: bool):
        for module in self.get_backbone_modules():
            for param in module.parameters():
                param.requires_grad = trainable

    def apply_partial_freeze(self):
        freeze_module = self.cfg.freeze_module
        if freeze_module == "None":
            return
        elif freeze_module == "all":
            for param in self.aggregator.parameters():
                param.requires_grad = False
        else:
            module_pairs = {
                "patch_embed+frame": ["patch_embed", "frame"],
                "patch_embed+global": ["patch_embed", "global"],
                "global+frame": ["global", "frame"],
            }

            if freeze_module in module_pairs:
                for name, param in self.aggregator.named_parameters():
                    if any(m in name for m in module_pairs[freeze_module]):
                        param.requires_grad = False
            else:
                for name, param in self.named_parameters():
                    param.requires_grad = (
                        freeze_module not in name and "distill" not in name
                    )

    def update_freeze_status(self, global_step: int):
        if self.freeze_backbone:
            return

        if self.freeze_backbone_until_iter > 0:
            if global_step < self.freeze_backbone_until_iter:
                self.set_backbone_trainable(False)
            elif not self.backbone_unfrozen:
                print(f"Unfreezing backbone at iter {global_step}", flush=True)
                self.set_backbone_trainable(True)
                self.backbone_unfrozen = True
                # optional: re-apply partial freeze after unfreezing if needed
                # self.`apply_partial_freeze`()

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def normalize_pts3d(self, pts3ds, valid_masks, original_extrinsics=None):
        # normalize pts_all
        B = pts3ds.shape[0]
        pts3d_norms = []
        scale_factors = []
        for bs in range(B):
            pts3d, valid_mask = pts3ds[bs], valid_masks[bs]
            if original_extrinsics is not None:
                camera_c2w = original_extrinsics[bs]
                first_camera_w2c = (
                    camera_c2w[0].inverse().unsqueeze(0).repeat(pts3d.shape[0], 1, 1)
                )

                pts3d_homo = torch.cat(
                    [pts3d, torch.ones_like(pts3d[:, :, :, :1])], dim=-1
                )
                transformed_pts3d = torch.bmm(
                    first_camera_w2c, pts3d_homo.flatten(1, 2).transpose(1, 2)
                ).transpose(1, 2)[..., :3]
                scene_scale = torch.norm(
                    transformed_pts3d.flatten(0, 1)[valid_mask.flatten(0, 2).bool()],
                    dim=-1,
                ).mean()
            else:
                transformed_pts3d = pts3d[valid_mask]
                dis = transformed_pts3d.norm(dim=-1)
                scene_scale = dis.mean().clip(min=1e-8)
            # pts3d_norm[bs] = pts3d[bs] / scene_scale
            pts3d_norms.append(pts3d / scene_scale)
            scale_factors.append(scene_scale)
        return torch.stack(pts3d_norms, dim=0), torch.stack(scale_factors, dim=0)

    def align_pts_all_with_pts3d(
        self, pts_all, pts3d, valid_mask, original_extrinsics=None
    ):
        # align pts_all with pts3d
        B = pts_all.shape[0]

        # follow vggt's normalization implementation
        pts3d_norm, scale_factor = self.normalize_pts3d(
            pts3d, valid_mask, original_extrinsics
        )  # check if this is correct
        pts_all = pts_all * scale_factor.view(B, 1, 1, 1, 1)

        return pts_all

    def pad_tensor_list(self, tensor_list, pad_shape, value=0.0):
        padded = []
        for t in tensor_list:
            pad_len = pad_shape[0] - t.shape[0]
            if pad_len > 0:
                padding = torch.full(
                    (pad_len, *t.shape[1:]), value, device=t.device, dtype=t.dtype
                )
                t = torch.cat([t, padding], dim=0)
            padded.append(t)
        return torch.stack(padded)
    
    def _gs_prune_batchwise_pad(
        self,
        neural_pts, depths, neural_feats, opacity,
        opacity_threshold: float,
        keep_ratio: float,
        pad_pts=-1e4,
        pad_feats=-1e10,
        pad_opacity=0.0,
        pad_depth=0.0,
    ):
        """
        neural_pts:   [B, N, 3]
        depths:       [B, N, 1]
        neural_feats: [B, N, raw_gs_dim]
        opacity:      [B, N]
        returns: padded tensors and valid mask [B, Kmax]
        """
        B, N, _ = neural_pts.shape
        device = neural_pts.device

        usage = opacity > opacity_threshold  # [B,N]

        # keep-ratio fallback PER SAMPLE (matches spirit of legacy)
        if keep_ratio < 1.0:
            target_keep = max(int(N * keep_ratio), 1)
            kept_counts = usage.sum(dim=1)
            for b in range(B):
                if kept_counts[b] > target_keep:
                    idx_sort = opacity[b].argsort(descending=True)
                    keep_idx = idx_sort[:target_keep]
                    new_mask = torch.zeros_like(usage[b])
                    new_mask[keep_idx] = True
                    usage[b] = new_mask

        pts_list, depth_list, feat_list, op_list = [], [], [], []
        for b in range(B):
            m = usage[b]
            if m.sum() == 0:
                m[opacity[b].argmax()] = True  # ensure at least one
            pts_list.append(neural_pts[b, m])       # [Kb,3]
            depth_list.append(depths[b, m])         # [Kb,1]
            feat_list.append(neural_feats[b, m])    # [Kb,D]
            op_list.append(opacity[b, m])           # [Kb]

        maxK = max(x.shape[0] for x in pts_list)
        idx = torch.arange(maxK, device=device)[None, :]
        counts = torch.tensor([x.shape[0] for x in pts_list], device=device)
        valid = idx < counts[:, None]  # [B,maxK]

        def pad2(xs, pad_val):
            out = []
            for x in xs:
                k = x.shape[0]
                if k < maxK:
                    pad = torch.full((maxK - k, x.shape[1]), pad_val, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=0)
                out.append(x)
            return torch.stack(out, dim=0)

        def pad1(xs, pad_val):
            out = []
            for x in xs:
                k = x.shape[0]
                if k < maxK:
                    pad = torch.full((maxK - k,), pad_val, device=x.device, dtype=x.dtype)
                    x = torch.cat([x, pad], dim=0)
                out.append(x)
            return torch.stack(out, dim=0)

        pr_pts = pad2(pts_list, pad_pts).view(B, maxK, 3)
        pr_depths = pad2([d.view(-1, 1) for d in depth_list], pad_depth).view(B, maxK, 1)
        pr_feats = pad2(feat_list, pad_feats).view(B, maxK, self.raw_gs_dim)
        pr_opacity = pad1(op_list, pad_opacity).view(B, maxK)

        return pr_pts.contiguous(), pr_depths.contiguous(), pr_feats.contiguous(), pr_opacity.contiguous(), valid


    def voxelizaton_with_multiscale_topk_fusion(
        self,
        img_feat,
        pts3d,
        voxel_size,
        conf=None,
        voxel_scales=(0.5, 1.0, 2.0),
        topk_per_voxel=2,
        final_dedup=True,
    ):
        # img_feat: [B*V, C, H, W]
        # pts3d:    [B*V, 3, H, W]
        # conf:     [B*V, H, W]

        device = pts3d.device
        dtype = pts3d.dtype

        pts_flat = pts3d.permute(0, 2, 3, 1).reshape(-1, 3)
        feat_flat = img_feat.permute(0, 2, 3, 1).reshape(-1, img_feat.shape[1])

        if conf is None:
            score_flat = torch.zeros(pts_flat.shape[0], device=device, dtype=dtype)
        else:
            score_flat = conf.reshape(-1).to(device=device, dtype=dtype)

        all_pts, all_feats, all_scores = [], [], []

        for scale in voxel_scales:
            vs = voxel_size * scale

            voxel_indices = (pts_flat / vs).round().int()
            _, inverse_indices = torch.unique(
                voxel_indices,
                dim=0,
                return_inverse=True,
            )
            inverse_indices = inverse_indices.long()

            keep = _topk_per_group(
                score=score_flat,
                group=inverse_indices,
                k=topk_per_voxel,
            )

            all_pts.append(pts_flat[keep])
            all_feats.append(feat_flat[keep])
            all_scores.append(score_flat[keep])

        voxel_pts = torch.cat(all_pts, dim=0)
        voxel_feats = torch.cat(all_feats, dim=0)
        voxel_scores = torch.cat(all_scores, dim=0)

        if final_dedup:
            # Remove duplicates selected by several scales.
            final_vs = voxel_size * min(voxel_scales)
            final_indices = (voxel_pts / final_vs).round().int()
            _, final_inverse = torch.unique(
                final_indices,
                dim=0,
                return_inverse=True,
            )
            final_inverse = final_inverse.long()

            keep = _topk_per_group(
                score=voxel_scores,
                group=final_inverse,
                k=1,
            )

            voxel_pts = voxel_pts[keep]
            voxel_feats = voxel_feats[keep]

        return voxel_pts, voxel_feats


    def voxelizaton_with_fusion_anysplat(self, img_feat, pts3d, voxel_size, conf=None):
        # img_feat: B*V, C, H, W
        # pts3d: B*V, 3, H, W
        V, C, H, W = img_feat.shape
        pts3d_flatten = pts3d.permute(0, 2, 3, 1).flatten(0, 2)

        voxel_indices = (pts3d_flatten / voxel_size).round().int()  # [B*V*N, 3]
        unique_voxels, inverse_indices, counts = torch.unique(voxel_indices, dim=0, return_inverse=True, return_counts=True)

        # Flatten confidence scores and features
        conf_flat = conf.flatten()  # [B*V*N]
        anchor_feats_flat = img_feat.permute(0, 2, 3, 1).flatten(0, 2)  # [B*V*N, ...]

        # Compute softmax weights per voxel
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)  # [num_unique_voxels]
        weights = (conf_exp / (voxel_weights[inverse_indices] + 1e-6)).unsqueeze(-1)  # [B*V*N, 1]

        # Compute weighted average of positions and features
        weighted_pts = pts3d_flatten * weights
        weighted_feats = anchor_feats_flat.squeeze(1) * weights

        # Aggregate per voxel
        voxel_pts = scatter_add(weighted_pts, inverse_indices, dim=0)  # [num_unique_voxels, 3]
        voxel_feats = scatter_add(weighted_feats, inverse_indices, dim=0)  # [num_unique_voxels, feat_dim]

        return voxel_pts, voxel_feats


    def voxelizaton_with_fusion(
        self,
        img_feat,
        pts3d,
        voxel_size,
        conf=None,
        static_prob=None,
        var=None,
        valid_mask=None,          # [V,H,W] or [V,1,H,W]
        temp=1.0,                 # <1 => sharper
        sigma_mul=0.5,
        robust=False,
        return_losses=False,
    ):
        # img_feat: [V, C, H, W]
        # pts3d:    [V, 3, H, W]
        V, C, H, W = img_feat.shape

        pts3d_flat = pts3d.permute(0, 2, 3, 1).reshape(-1, 3)       # [N,3]
        img_feat_flat = img_feat.permute(0, 2, 3, 1).reshape(-1, C) # [N,C]
        N = pts3d_flat.shape[0]

        voxel_indices = (pts3d_flat / voxel_size).round().int()
        unique_voxels, inverse_indices, counts = torch.unique(
            voxel_indices,
            dim=0,
            return_inverse=True,
            return_counts=True,
        )

        # Confidence logits.
        if conf is None:
            conf_flat = torch.zeros(
                N,
                device=pts3d_flat.device,
                dtype=pts3d_flat.dtype,
            )
        else:
            cc = conf
            if cc.ndim == 4:
                cc = cc.squeeze(1)
            conf_flat = cc.reshape(-1).to(pts3d_flat.dtype)

        assert conf_flat.numel() == N, (
            f"conf size {conf_flat.shape} does not match points {N}"
        )

        conf_flat = conf_flat / max(temp, 1e-6)

        # Stable per-voxel softmax.
        conf_voxel_max, _ = scatter_max(conf_flat, inverse_indices, dim=0)
        conf_exp = torch.exp(conf_flat - conf_voxel_max[inverse_indices])

        # Static gate: 1 = static/keep, 0 = transient/suppress.
        if static_prob is not None:
            sp = static_prob
            if sp.ndim == 4:
                sp = sp.squeeze(1)

            gate_flat = sp.reshape(-1).to(conf_exp.dtype).clamp(0.0, 1.0)

            assert gate_flat.numel() == N, (
                f"static gate size {gate_flat.shape} does not match points {N}"
            )

            conf_exp = conf_exp * gate_flat.clamp_min(1e-4)

        # Optional reliability gate.
        if valid_mask is not None:
            vm = valid_mask
            if vm.ndim == 4:
                vm = vm.squeeze(1)

            vm_flat = vm.reshape(-1)

            assert vm_flat.numel() == N, (
                f"valid_mask size {vm_flat.shape} does not match points {N}"
            )

            if vm_flat.dtype == torch.bool:
                vm_flat = vm_flat.to(conf_exp.dtype)
            else:
                vm_flat = vm_flat.to(conf_exp.dtype).clamp(0.0, 1.0)

            conf_exp = conf_exp * vm_flat

        # Normalize within each voxel.
        voxel_weights = scatter_add(conf_exp, inverse_indices, dim=0)
        denom = voxel_weights[inverse_indices].clamp_min(1e-12)
        weights = (conf_exp / denom).clamp_min(0.0)  # [N]

        # Optional robust geometric reweighting.
        if robust:
            wsum = scatter_add(weights, inverse_indices, dim=0).clamp_min(1e-12)
            mu = scatter_add(
                pts3d_flat * weights[:, None],
                inverse_indices,
                dim=0,
            ) / wsum[:, None]

            sigma = sigma_mul * voxel_size
            r2 = ((pts3d_flat - mu[inverse_indices]) ** 2).sum(-1)
            inlier = torch.exp(-r2 / (2 * sigma * sigma + 1e-12))

            weights2 = weights * inlier
            denom2 = scatter_add(weights2, inverse_indices, dim=0).clamp_min(1e-12)
            weights = weights2 / denom2[inverse_indices]

        # Aggregate per voxel.
        voxel_pts = scatter_add(
            pts3d_flat * weights[:, None],
            inverse_indices,
            dim=0,
        )

        voxel_feats = scatter_add(
            img_feat_flat * weights[:, None],
            inverse_indices,
            dim=0,
        )

        voxel_static = None
        if static_prob is not None:
            voxel_static = scatter_add(
                gate_flat * weights,
                inverse_indices,
                dim=0,
            )
            voxel_static = voxel_static.clamp(0.0, 1.0)

        # Optional variance fusion.
        voxel_var = None
        if var is not None:
            vv = var
            if vv.ndim == 4:
                vv = vv.squeeze(1)

            var_flat = vv.reshape(-1).to(weights.dtype)

            assert var_flat.numel() == N, (
                f"var size {var_flat.shape} does not match points {N}"
            )

            voxel_var = scatter_add(
                var_flat * weights,
                inverse_indices,
                dim=0,
            )

        # Optional auxiliary losses.
        loss_entropy = None
        loss_voxvar = None

        if return_losses:
            ent_point = -(weights * torch.log(weights + 1e-8))
            ent_vox = scatter_add(ent_point, inverse_indices, dim=0)

            wsumf = scatter_add(weights, inverse_indices, dim=0).clamp_min(1e-12)
            muf = scatter_add(
                pts3d_flat * weights[:, None],
                inverse_indices,
                dim=0,
            ) / wsumf[:, None]

            r2f = ((pts3d_flat - muf[inverse_indices]) ** 2).sum(-1)
            voxvar = scatter_add(r2f * weights, inverse_indices, dim=0)

            # Avoid single-point voxels dominating diagnostics.
            count_per_voxel = scatter_add(
                torch.ones_like(weights),
                inverse_indices,
                dim=0,
            )
            multi = count_per_voxel > 1

            if multi.any():
                loss_entropy = ent_vox[multi].mean()
                loss_voxvar = voxvar[multi].mean()
            else:
                loss_entropy = ent_vox.mean() * 0.0
                loss_voxvar = voxvar.mean() * 0.0

        return voxel_pts, voxel_feats, voxel_var, voxel_static, loss_entropy, loss_voxvar

    def _choose_voxel_corner(self, p, voxel_size, mode="stochastic", sharpen_p=4.0):
        # p: [N,3]
        v = p / voxel_size
        v0 = torch.floor(v).to(torch.int32)                 # [N,3]
        frac = (v - v0.to(v.dtype)).clamp(0, 1)             # [N,3]
        fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]

        w000 = (1-fx)*(1-fy)*(1-fz)
        w100 = (fx)*(1-fy)*(1-fz)
        w010 = (1-fx)*(fy)*(1-fz)
        w110 = (fx)*(fy)*(1-fz)
        w001 = (1-fx)*(1-fy)*(fz)
        w101 = (fx)*(1-fy)*(fz)
        w011 = (1-fx)*(fy)*(fz)
        w111 = (fx)*(fy)*(fz)
        w8 = torch.stack([w000,w100,w010,w110,w001,w101,w011,w111], dim=1)  # [N,8]

        if sharpen_p is not None and sharpen_p != 1.0:
            w8 = w8.clamp_min(1e-12).pow(float(sharpen_p))
            w8 = w8 / (w8.sum(dim=1, keepdim=True) + 1e-12)

        if mode == "argmax":
            choice = w8.argmax(dim=1)                       # [N]
        elif mode == "stochastic":
            choice = torch.multinomial(w8, num_samples=1).squeeze(1)  # [N]
            # faster version:
            # u = torch.rand_like(w8)
            # g = -torch.log(-torch.log(u + 1e-12) + 1e-12)
            # choice = (torch.log(w8 + 1e-12) + g).argmax(dim=1)
        else:
            raise ValueError(f"Unknown corner choice mode: {mode}")

        vox = v0 + self.voxel_offsets[choice]                          # [N,3]
        return vox

    # def voxelizaton_with_fusion_tmp(
    #     self,
    #     img_feat, pts3d, voxel_size,
    #     conf=None, static_prob=None, var=None,
    #     valid_mask=None,
    #     temp=0.7,
    #     sigma_mul=0.5,
    #     robust=True,
    #     return_losses=True,
    #     assignment="round",       # "round" | "stochastic8" | "argmax8" | "soft8"
    #     splat_sharpen_p=4.0,       # >1 sharpens (less blur). try 2..8
    #     clamp_conf=True,
    #     spatial_gate=False,
    #     spatial_sigma_mul=0.5,   # in voxel units, 0.3~0.8 typical
    # ):
    #     """
    #     img_feat: [V, C, H, W]
    #     pts3d:    [V, 3, H, W]
    #     conf:     [V, H, W] or [V,1,H,W]
    #     """
    #     V, C, H, W = img_feat.shape
    #     p = pts3d.permute(0, 2, 3, 1).reshape(-1, 3)        # [N,3]
    #     f = img_feat.permute(0, 2, 3, 1).reshape(-1, C)     # [N,C]
    #     N = p.shape[0]

    #     # ---- confidence logits -> exp weights (1D)
    #     if conf is None:
    #         c = torch.zeros((N,), device=p.device, dtype=p.dtype)
    #     else:
    #         c = conf
    #         if c.ndim == 4:
    #             c = c.squeeze(1)
    #         c = c.reshape(-1).to(p.dtype)

    #     c = c / max(temp, 1e-6)
    #     if clamp_conf:
    #         c = c.clamp(-10, 10)

    #     # ---- Gates (1D)
    #     gate = None
    #     if static_prob is not None:
    #         g = static_prob
    #         if g.ndim == 4:
    #             g = g.squeeze(1)
    #         gate = g.reshape(-1).to(p.dtype).clamp(0.0, 1.0)

    #     if valid_mask is not None:
    #         vm = valid_mask
    #         if vm.ndim == 4:
    #             vm = vm.squeeze(1)
    #         vm = vm.reshape(-1)
    #         if vm.dtype == torch.bool:
    #             vm = vm.to(p.dtype)
    #         else:
    #             vm = vm.to(p.dtype).clamp(0.0, 1.0)
    #         gate = vm if gate is None else (gate * vm)

    #     if gate is None:
    #         gate = torch.ones((N,), device=p.device, dtype=p.dtype)

    #     # ============================================================
    #     # Voxel assignment (NO expansion for stochastic8/argmax8)
    #     # ============================================================
    #     if assignment in ["round", "floor", "stochastic8", "argmax8"]:
    #         if assignment == "round":
    #             vox = (p / voxel_size).round().to(torch.int32)  # [N,3]
    #             center = vox.to(p.dtype) * voxel_size
    #         elif assignment == "floor":
    #             vox = torch.floor(p / voxel_size).to(torch.int32)  # [N,3]
    #             center = (vox.to(p.dtype) + 0.5) * voxel_size
    #         elif assignment == "stochastic8":
    #             vox = self._choose_voxel_corner(p, voxel_size, mode="stochastic", sharpen_p=splat_sharpen_p)  # [N,3]
    #             center = vox.to(p.dtype) * voxel_size
    #         elif assignment == "argmax8":
    #             vox = self._choose_voxel_corner(p, voxel_size, mode="argmax", sharpen_p=splat_sharpen_p)      # [N,3]
    #             center = vox.to(p.dtype) * voxel_size

    #         if spatial_gate:
    #             sigma = spatial_sigma_mul * voxel_size
    #             r2 = ((p - center) ** 2).sum(-1)
    #             g_sp = torch.exp(-r2 / (2 * sigma * sigma + 1e-12))
    #         else:
    #             g_sp = 1.0
                
    #         unique_vox, inv, counts = torch.unique(vox, dim=0, return_inverse=True, return_counts=True)

    #         # stable per-voxel softmax on c
    #         cmax, _ = scatter_max(c, inv, dim=0)
    #         expc = torch.exp(c - cmax[inv]) * gate * g_sp

    #         denom = scatter_add(expc, inv, dim=0).clamp_min(1e-12)  # [num_vox]
    #         w = (expc / denom[inv]).clamp_min(0.0)                  # [N]

    #     elif assignment == "soft8":
    #         # WARNING: this is the N*8 expansion version; will likely OOM at your scale.
    #         # ============================================================
    #         # SOFT splatting to 8 neighbor voxels with sharpening
    #         # ============================================================
    #         v = p / voxel_size                                # [N,3]
    #         v0 = torch.floor(v).to(torch.int32)               # base corner [N,3]
    #         frac = (v - v0.to(v.dtype)).clamp(0, 1)           # [N,3]

    #         fx, fy, fz = frac[:, 0], frac[:, 1], frac[:, 2]
    #         w000 = (1 - fx) * (1 - fy) * (1 - fz)
    #         w100 = (fx)     * (1 - fy) * (1 - fz)
    #         w010 = (1 - fx) * (fy)     * (1 - fz)
    #         w110 = (fx)     * (fy)     * (1 - fz)
    #         w001 = (1 - fx) * (1 - fy) * (fz)
    #         w101 = (fx)     * (1 - fy) * (fz)
    #         w011 = (1 - fx) * (fy)     * (fz)
    #         w111 = (fx)     * (fy)     * (fz)
    #         w8 = torch.stack([w000, w100, w010, w110, w001, w101, w011, w111], dim=1)  # [N,8]

    #         # Sharpen to reduce blur: push mass toward nearest voxel corner
    #         if splat_sharpen_p is not None and splat_sharpen_p != 1.0:
    #             w8 = (w8.clamp_min(1e-12) ** float(splat_sharpen_p))
    #             w8 = w8 / (w8.sum(dim=1, keepdim=True) + 1e-12)

    #         vox8 = v0[:, None, :] + self.voxel_offsets[None, :, :]       # [N,8,3]
    #         vox8 = vox8.reshape(-1, 3)                         # [N*8,3]

    #         # Expand per-point data to match 8 contributions
    #         inv_point = torch.arange(N, device=p.device).repeat_interleave(8)  # [N*8]
    #         gate8 = gate[inv_point]                                           # [N*8]
    #         c8 = c[inv_point]                                                 # [N*8]
    #         splat_w = w8.reshape(-1)                                          # [N*8]

    #         # Unique voxel ids for expanded contributions
    #         unique_vox, inv, counts = torch.unique(vox8, dim=0, return_inverse=True, return_counts=True)

    #         # stable per-voxel softmax over c8 (still per voxel)
    #         cmax, _ = scatter_max(c8, inv, dim=0)
    #         expc = torch.exp(c8 - cmax[inv])                                   # [N*8]

    #         # combine: confidence numerator * gates * splat weight
    #         expc = expc * gate8 * splat_w                                      # [N*8]

    #         denom = scatter_add(expc, inv, dim=0).clamp_min(1e-12)
    #         w = (expc / denom[inv]).clamp_min(0.0)                             # [N*8]

    #         # For aggregation, we need expanded p/f/var to [N*8,...]
    #         p = p[inv_point]                                                   # [N*8,3]
    #         f = f[inv_point]                                                   # [N*8,C]
    #         if var is not None:
    #             vv = var
    #             if vv.ndim == 4:
    #                 vv = vv.squeeze(1)
    #             var_flat = vv.reshape(-1).to(p.dtype)
    #             var_flat = var_flat[inv_point]                                  # [N*8] 

    #     else:
    #         raise ValueError(f"Unknown assignment mode: {assignment}")


    #     # ---- Robust within-voxel reweighting (works for both modes)
    #     if robust:
    #         wsum = scatter_add(w, inv, dim=0).clamp_min(1e-12)                 # [num_vox]
    #         mu = scatter_add(p * w[:, None], inv, dim=0) / wsum[:, None]       # [num_vox,3]

    #         sigma = sigma_mul * voxel_size
    #         r2 = ((p - mu[inv]) ** 2).sum(-1)                                  # [M] (M=N or N*8)
    #         inlier = torch.exp(-r2 / (2 * (sigma ** 2) + 1e-12))               # [M]
    #         w2 = w * inlier

    #         denom2 = scatter_add(w2, inv, dim=0).clamp_min(1e-12)
    #         w = w2 / denom2[inv]                                               # [M]

    #     # ---- Aggregate per voxel
    #     voxel_pts = scatter_add(p * w[:, None], inv, dim=0)                    # [num_vox,3]
    #     voxel_feats = scatter_add(f * w[:, None], inv, dim=0)                  # [num_vox,C]

    #     voxel_var = None
    #     if var is not None:
    #         if assignment == "soft8":
    #             # var_flat already expanded to [N*8]
    #             voxel_var = scatter_add(var_flat * w, inv, dim=0)
    #         else:
    #             vv = var
    #             if vv.ndim == 4:
    #                 vv = vv.squeeze(1)
    #             var_flat = vv.reshape(-1).to(voxel_pts.dtype)                  # [N]
    #             voxel_var = scatter_add(var_flat * w, inv, dim=0)              # [num_vox]

    #     # ---- Optional training-time losses
    #     loss_entropy = None
    #     loss_voxvar = None
    #     if return_losses:
    #         ent_point = -(w * torch.log(w + 1e-8))                             # [M]
    #         ent_vox = scatter_add(ent_point, inv, dim=0)                       # [num_vox]

    #         wsumf = scatter_add(w, inv, dim=0).clamp_min(1e-12)
    #         muf = scatter_add(p * w[:, None], inv, dim=0) / wsumf[:, None]
    #         r2f = ((p - muf[inv]) ** 2).sum(-1)
    #         # voxvar = scatter_add(r2f * w, inv, dim=0)
    #         norm = (voxel_size ** 2)
    #         voxvar = scatter_add((r2f / (norm + 1e-12)) * w, inv, dim=0)

    #         counts = scatter_add(torch.ones_like(w), inv, dim=0)  # or use counts from torch.unique
    #         mask = counts > 1
    #         if mask.any():
    #             loss_entropy = ent_vox[mask].mean()
    #             loss_voxvar  = voxvar[mask].mean()
    #         else:
    #             loss_entropy = ent_vox.mean() * 0.0
    #             loss_voxvar  = voxvar.mean() * 0.0

    #     return voxel_pts, voxel_feats, voxel_var, loss_entropy, loss_voxvar

    def forward(
        self,
        image: torch.Tensor,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        target_image: Optional[torch.Tensor] = None,
    ) -> Gaussians:
        
        device = image.device
        b, v, _, h, w = image.shape
        print(
            "[EncoderAnySplat.forward] image shape:",
            image.shape,
            "target_image:",
            None if target_image is None else target_image.shape,
            flush=True,
        )
        
        distill_infos = {}
        if self.distill:

            if target_image is not None:
                distill_image = target_image.detach()
            else:
                distill_image = image.detach()
            # print("[forward] distill_image", distill_image.shape, flush=True)

            if self.cfg.offload_distill_to_cpu:
                for module in [
                    self.distill_aggregator,
                    self.distill_camera_head,
                    self.distill_depth_head,
                ]:
                    for param in module.parameters():
                        param.data = param.data.to(device, non_blocking=True)

            with torch.no_grad():
                # Process with bfloat16 precision
                with torch.amp.autocast("cuda", enabled=True, dtype=dtype_):
                    if self.vggt_variant == "vggt":
                        distill_aggregated_tokens_list, distill_patch_start_idx = (
                            self.distill_aggregator(
                                distill_image.to(dtype_),
                                intermediate_layer_idx=self.cfg.intermediate_layer_idx,
                            )
                        )
                    else:
                        distill_aggregated_tokens_list, distill_patch_start_idx = (
                            self.distill_aggregator(distill_image.to(dtype_))
                        )

                # Process with default precision
                with torch.amp.autocast("cuda", enabled=False):
                    # Get camera pose information w2c
                    if self.vggt_variant == "vggt":
                        distill_pred_pose_enc_list = self.distill_camera_head(
                            distill_aggregated_tokens_list
                        )
                        last_distill_pred_pose_enc = distill_pred_pose_enc_list[-1]
                    else:
                        last_distill_pred_pose_enc = self.distill_camera_head(
                            distill_aggregated_tokens_list,
                            patch_token_start=distill_patch_start_idx,
                        )
                        distill_pred_pose_enc_list = [last_distill_pred_pose_enc]

                    distill_extrinsic, distill_intrinsic = pose_encoding_to_extri_intri(
                        last_distill_pred_pose_enc, image.shape[-2:]
                    )

                    # Get depth information
                    if self.vggt_variant == "vggt":
                        distill_depth_map, distill_depth_conf = self.distill_depth_head(
                            distill_aggregated_tokens_list,
                            images=distill_image,
                            patch_start_idx=distill_patch_start_idx,
                        )
                    else:
                        distill_depth_map, distill_depth_conf = self.distill_depth_head(
                            distill_aggregated_tokens_list,
                            images=distill_image,
                            patch_token_start=distill_patch_start_idx,
                        )


                    # Convert depth to 3D points
                    distill_pts_all = batchify_unproject_depth_map_to_point_map(
                        distill_depth_map, distill_extrinsic, distill_intrinsic
                    )
                # Store results
                # for p_idx in range(len(distill_pred_pose_enc_list)):
                #     distill_pred_pose_enc_list[p_idx] = distill_pred_pose_enc_list[p_idx][:,:v]
                distill_infos["pred_pose_enc_list"] = distill_pred_pose_enc_list
                distill_infos["pts_all"] = distill_pts_all #[:,:v]
                distill_infos["depth_map"] = distill_depth_map #[:,:v]

                conf_threshold = torch.quantile(distill_depth_conf.flatten(2, 3), 0.3, dim=-1, keepdim=True)  # Get threshold for each view
                conf_mask = distill_depth_conf > conf_threshold.unsqueeze(-1)
                distill_infos["depth_conf"] = distill_depth_conf #[:,:v]
                distill_infos["conf_mask"] = conf_mask #[:,:v]

                if target_image is not None:
                    extrinsic_target = distill_extrinsic[:,v:]
                    intrinsic_target = distill_intrinsic[:,v:]
                    extrinsic_padding = (torch.tensor([0, 0, 0, 1], device=device, dtype=distill_extrinsic.dtype).view(1, 1, 1, 4).repeat(b, distill_extrinsic.shape[1], 1, 1))
                    distill_infos["extrinsic"] = torch.cat([distill_extrinsic, extrinsic_padding], dim=2).inverse() #w2c -> c2w
                    distill_infos["intrinsic"] = torch.stack([distill_intrinsic[:, :, 0] / w, distill_intrinsic[:, :, 1] / h, distill_intrinsic[:, :, 2]], dim=2)
                    # print("[forward] extrinsic_target", extrinsic_target.shape, "intrinsic_target", intrinsic_target.shape)

                if self.cfg.offload_distill_to_cpu:
                    for module in [
                        self.distill_aggregator,
                        self.distill_camera_head,
                        self.distill_depth_head,
                    ]:
                        for param in module.parameters():
                            param.data = param.data.cpu()
                # Clean up to save memory
                del distill_aggregated_tokens_list, distill_patch_start_idx
                del distill_pred_pose_enc_list, last_distill_pred_pose_enc
                del distill_extrinsic, distill_intrinsic
                del distill_depth_map, distill_depth_conf
                # torch.cuda.empty_cache()

        with torch.amp.autocast("cuda", enabled=True, dtype=dtype_):
            # image_has_nan = torch.isnan(image).any()
            # print("image has nan:", image_has_nan, image.shape, image.min(), image.max(), image.dtype)
            agg_out = self._run_aggregator(image.to(dtype_), visualization_dump=visualization_dump)
            if visualization_dump is not None and self.vggt_variant == "vggt":
                aggregated_tokens_list, patch_start_idx, visualization_dump = agg_out
                for token in aggregated_tokens_list:
                    token_has_nan = torch.isnan(token).any()
                    if token_has_nan:
                        print("token has nan:", token_has_nan, token.shape)
            else:
                aggregated_tokens_list, patch_start_idx = agg_out

        with torch.amp.autocast("cuda", enabled=False):
            pred_pose_enc_list, last_pred_pose_enc = self._run_camera_head(
                aggregated_tokens_list,
                patch_start_idx,
            )
            extrinsic, intrinsic = pose_encoding_to_extri_intri(
                last_pred_pose_enc, image.shape[-2:]
            )  # only for debug

            if self.cfg.pred_head_type == "point":
                pts_all, pts_conf = self.point_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
            elif self.cfg.pred_head_type == "depth":
                depth_map, depth_conf = self._run_depth_head(
                    aggregated_tokens_list,
                    image,
                    patch_start_idx,
                )
                pts_all = batchify_unproject_depth_map_to_point_map(
                    depth_map, extrinsic, intrinsic
                )
            else:
                raise ValueError(f"Invalid pred_head_type: {self.cfg.pred_head_type}")

            if self.cfg.render_conf:
                conf_valid = torch.quantile(
                    depth_conf.flatten(0, 1), self.cfg.conf_threshold
                )
                conf_valid_mask = depth_conf > conf_valid
            else:
                conf_valid_mask = torch.ones_like(depth_conf, dtype=torch.bool)

            # print("self.cfg.bilateral_grid", self.cfg.bilateral_grid, "image", image.shape) #torch.Size([1, 4, 3, 350, 448])
            if self.cfg.bilateral_grid:
                bg, bg_conf = self.bg_param_head(
                    aggregated_tokens_list,
                    images=image,
                    patch_start_idx=patch_start_idx,
                )
                # print("bg raw", bg.shape, "bg_conf raw", bg_conf.shape) # bg raw torch.Size([B, S, 16, 16, 96]) bg_conf raw torch.Size([B, S, 16, 16, 8])
                bg = bg.view(*bg.shape[:4], 8, 12).permute(0, 1, 5, 4, 2, 3)
                bg_conf = bg_conf.view(*bg_conf.shape[:4], 8, 1).permute(0, 1, 5, 4, 2, 3)
                # print("bg", bg.shape, "bg_conf", bg_conf.shape) #bg torch.Size([B, S, 12, 8, 16, 16]) bg_conf torch.Size([B, S, 1, 8, 16, 16])
                # print("bg:", bg[0,0,:,0,0,0], flush=True)
            else:
                bg = None
                bg_conf = None

        if self.vggt_variant == "vggt":
            out = self.gaussian_param_head(
                aggregated_tokens_list,
                pts_all.flatten(0, 1).permute(0, 3, 1, 2),
                image,
                patch_start_idx=patch_start_idx,
                image_size=(h, w),
            )
        else:
            out = self.gaussian_param_head(
                aggregated_tokens_list,
                image,
                patch_token_start=patch_start_idx,
            )

        del aggregated_tokens_list, patch_start_idx
        # torch.cuda.empty_cache()

        pts_flat = pts_all.flatten(2, 3)
        scene_scale = pts_flat.norm(dim=-1).mean().clip(min=1e-8)

        if self.predict_static and self.predict_uncertainty:
            anchor_feats = out[:, :, :self.raw_gs_dim]
            conf = out[:, :, self.raw_gs_dim]

            # Compact copies so returned scalar maps do not pin full `out` storage.
            static_logit = out[:, :, self.raw_gs_dim + 1].contiguous()
            logvar = out[:, :, self.raw_gs_dim + 2].contiguous()

            static_prob = (
                static_logit.sigmoid()
                if self.cfg.fusion_mode == "static"
                else None
            )
            var = logvar_to_var(logvar)

        elif self.predict_static:
            anchor_feats = out[:, :, :self.raw_gs_dim]
            conf = out[:, :, self.raw_gs_dim]

            # Critical: separate compact storage from the full DPT output.
            static_logit = out[:, :, self.raw_gs_dim + 1].contiguous()

            static_prob = (
                static_logit.sigmoid()
                if self.cfg.fusion_mode == "static"
                else None
            )
            var = None

        elif self.predict_uncertainty:
            anchor_feats = out[:, :, :self.raw_gs_dim]
            conf = out[:, :, self.raw_gs_dim]

            logvar = out[:, :, self.raw_gs_dim + 1].contiguous()
            static_logit = None
            static_prob = None
            var = logvar_to_var(logvar)

        else:
            anchor_feats = out[:, :, :self.raw_gs_dim]
            conf = out[:, :, self.raw_gs_dim]
            static_logit = None
            static_prob = None
            var = None

        if self.predict_static:
            print("static_logit:", static_logit.min().item(), static_logit.mean().item(), static_logit.max().item(), flush=True)
            print("static_prob :", static_prob.min().item(), static_prob.mean().item(), static_prob.max().item(), flush=True)

        neural_feats_list, neural_pts_list, neural_var_list, neural_static_list = [], [], [], []
        loss_entropy = torch.zeros((), device=device)
        loss_voxvar  = torch.zeros((), device=device)
        loss_entropy_i = 0.0
        loss_voxvar_i = 0.0
        return_losses = False

        if self.cfg.voxelize:
            for b_i in range(b):
                pts_b = pts_all[b_i].permute(0, 3, 1, 2).contiguous()

                if self.cfg.fusion_mode == "anysplat":
                    neural_pts, neural_feats = self.voxelizaton_with_fusion_anysplat(
                        anchor_feats[b_i],
                        pts_b,
                        self.voxel_size,
                        conf=conf[b_i],
                    )
                    neural_var = None
                    neural_static = None
                    loss_entropy_i = torch.zeros((), device=device)
                    loss_voxvar_i = torch.zeros((), device=device)

                elif self.cfg.fusion_mode == "multiscale":
                    neural_pts, neural_feats = self.voxelizaton_with_multiscale_topk_fusion(
                        anchor_feats[b_i],
                        pts_b,
                        self.voxel_size,
                        conf=conf[b_i],
                    )
                    neural_var = None
                    neural_static = None
                    loss_entropy_i = torch.zeros((), device=device)
                    loss_voxvar_i = torch.zeros((), device=device)


                elif self.cfg.fusion_mode == "static":
                    neural_pts, neural_feats, neural_var, neural_static, loss_entropy_i, loss_voxvar_i = (
                        self.voxelizaton_with_fusion(
                            anchor_feats[b_i],
                            pts_b,
                            self.voxel_size,
                            conf=conf[b_i],
                            static_prob=(static_prob[b_i] if self.predict_static else None),
                            var=(var[b_i] if self.predict_uncertainty else None),
                            temp=1.0,
                            robust=False,
                            return_losses=False,
                        )
                    )

                else:
                    raise ValueError(f"Unknown fusion_mode: {self.cfg.fusion_mode}")

                # Save originals for debugging.
                used_refiner = False
                orig_pts = neural_pts
                orig_feats = neural_feats

                if getattr(self.cfg, "gaussian_refiner", False):
                    neural_pts, neural_feats = self.gaussian_refiner(neural_pts, neural_feats)
                    used_refiner = True

                if getattr(self.cfg, "gaussian_attention_refiner", False):
                    neural_pts, neural_feats = self.gaussian_attention_refiner(neural_pts, neural_feats)
                    used_refiner = True

                if used_refiner:
                    dxyz_norm = (neural_pts - orig_pts).norm(dim=-1).mean()
                    dfeat_norm = (neural_feats - orig_feats).norm(dim=-1).mean()

                neural_feats_list.append(neural_feats)
                neural_pts_list.append(neural_pts)

                if self.cfg.fusion_mode == "static" and self.predict_static:
                    if neural_static is None:
                        neural_static = torch.ones(
                            (neural_pts.shape[0],),
                            device=neural_pts.device,
                            dtype=neural_pts.dtype,
                        )
                    neural_static_list.append(neural_static)

                if self.predict_uncertainty:
                    if neural_var is None:
                        neural_var = torch.full(
                            (neural_pts.shape[0],),
                            1e6,
                            device=neural_pts.device,
                            dtype=neural_pts.dtype,
                        )
                    neural_var_list.append(neural_var)

                if return_losses:
                    loss_entropy = loss_entropy + loss_entropy_i
                    loss_voxvar = loss_voxvar + loss_voxvar_i
                    
        else:
            for b_i in range(b):
                mask_b_i = conf_valid_mask[b_i]
                neural_feats_list.append(
                    anchor_feats[b_i].permute(0, 2, 3, 1)[mask_b_i]
                )
                neural_pts_list.append(pts_all[b_i][mask_b_i])
                if self.predict_uncertainty:
                    neural_var_list.append(var[b_i][mask_b_i])  # (N,)

        max_voxels = max(f.shape[0] for f in neural_feats_list)
        neural_feats = self.pad_tensor_list(neural_feats_list, (max_voxels,), value=-1e10)
        neural_pts = self.pad_tensor_list(neural_pts_list, (max_voxels,), -1e4)  # -1 == invalid voxel
        if self.cfg.fusion_mode == "static" and self.predict_static:
            neural_static = self.pad_tensor_list(neural_static_list, (max_voxels,), value=1.0)
        else:
            neural_static = None
        if self.predict_uncertainty:
            neural_var = self.pad_tensor_list(neural_var_list, (max_voxels,), value=1e6)
        else:
            neural_var = None

        depths = neural_pts[..., -1].unsqueeze(-1)
        densities = neural_feats[..., 0].sigmoid()

        assert len(densities.shape) == 2, "the shape of densities should be (B, N)"
        assert neural_pts.shape[1] > 1, "the number of voxels should be greater than 1"

        opacity = self.map_pdf_to_opacity(densities, global_step).squeeze(-1)
        if neural_static is not None:
            opacity = opacity * neural_static.detach().clamp(0.0, 1.0)
        # # -------------------------------
        # # Mask padded (fake) Gaussians
        # # -------------------------------
        # num_gauss = torch.tensor([t.shape[0] for t in neural_pts_list], device=device)
        # idx = torch.arange(max_voxels, device=device)[None, :]
        # valid = idx < num_gauss[:, None]  # (B, max_voxels)
        if self.cfg.opacity_conf:
            shift = torch.quantile(depth_conf, self.cfg.conf_threshold)
            opacity = opacity * torch.sigmoid(depth_conf - shift)[conf_valid_mask].unsqueeze(0)  # little bit hacky
        # opacity = opacity * valid.float()

        # GS Prune, but only works when bs = 1
        # if want to support bs > 1, need to random prune gaussians based on the rank of opacity like LongLRM
        # Note: we not prune gaussians here, but we will try it in the future
        if self.cfg.gs_prune:
            opacity_threshold = self.cfg.opacity_threshold

            if self.cfg.gs_prune_mode == "legacy_bs1":
                print("[gs_prune] pruning Gaussians with legacy_bs1", flush=True)
                # ORIGINAL AnySplat behavior
                if b == 1:
                    gaussian_usage = opacity > opacity_threshold  # (B,N)

                    if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
                        num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
                        idx_sort = opacity.argsort(dim=1, descending=True)
                        keep_idx = idx_sort[:, :num_keep]
                        gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
                        gaussian_usage.scatter_(1, keep_idx, True)

                    neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
                    depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
                    neural_feats = neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous()
                    opacity = opacity[gaussian_usage].view(b, -1).contiguous()

                # if b>1, do nothing (matches legacy limitation)

            elif self.cfg.gs_prune_mode == "batchwise_pad":
                print("[gs_prune] pruning Gaussians with batchwise_pad", flush=True)
                neural_pts, depths, neural_feats, opacity, valid_prune = self._gs_prune_batchwise_pad(
                    neural_pts, depths, neural_feats, opacity,
                    opacity_threshold=opacity_threshold,
                    keep_ratio=self.cfg.gs_keep_ratio,
                    pad_pts=-1e4,
                    pad_feats=-1e10,
                    pad_opacity=0.0,
                    pad_depth=0.0,
                )
                # ensure padded entries don't contribute
                opacity = opacity * valid_prune.float()

            else:
                raise ValueError(f"Unknown gs_prune_mode: {self.cfg.gs_prune_mode}")

        # if self.cfg.gs_prune and b == 1:
        #     opacity_threshold = self.cfg.opacity_threshold
        #     gaussian_usage = opacity > opacity_threshold  # (B, N)
        #     num_before = opacity.shape[1]

        #     print(
        #         f"based on opacity threshold {opacity_threshold}, pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
        #     )

        #     if (gaussian_usage.sum() / gaussian_usage.numel()) > self.cfg.gs_keep_ratio:
        #         # rank by opacity
        #         num_keep = int(gaussian_usage.shape[1] * self.cfg.gs_keep_ratio)
        #         idx_sort = opacity.argsort(dim=1, descending=True)
        #         keep_idx = idx_sort[:, :num_keep]
        #         gaussian_usage = torch.zeros_like(gaussian_usage, dtype=torch.bool)
        #         gaussian_usage.scatter_(1, keep_idx, True)

        #     neural_pts = neural_pts[gaussian_usage].view(b, -1, 3).contiguous()
        #     depths = depths[gaussian_usage].view(b, -1, 1).contiguous()
        #     neural_feats = (neural_feats[gaussian_usage].view(b, -1, self.raw_gs_dim).contiguous())
        #     opacity = opacity[gaussian_usage].view(b, -1).contiguous()
        #     if self.predict_uncertainty:
        #         neural_var = neural_var[gaussian_usage].view(b, -1).contiguous()

        #     print(
        #         f"finally pruned {gaussian_usage.shape[1] - neural_pts.shape[1]} gaussians out of {gaussian_usage.shape[1]}"
        #     )

        gaussians = self.gaussian_adapter.forward(
            neural_pts,
            depths,
            opacity,
            neural_feats[..., 1:].squeeze(2),
        )
        del anchor_feats
        del conf
        del out
        if self.predict_uncertainty:
            gaussians.vars = neural_var  # (B, N)

        if visualization_dump is not None:
            if 'depth' in visualization_dump.keys(): 
                visualization_dump["depth"] = rearrange(
                    pts_all[..., -1].flatten(2, 3).unsqueeze(-1).unsqueeze(-1),
                    "b v (h w) srf s -> b v h w srf s",
                    h=h,
                    w=w,
                )


        infos = {}
        infos["scene_scale"] = scene_scale
        infos["voxelize_ratio"] = densities.shape[1] / (h * w * v)
        infos["loss_entropy"] = loss_entropy / b
        infos["loss_voxvar"] = loss_voxvar / b
        if used_refiner:
            infos["refiner_dxyz"] = dxyz_norm.detach()
            infos["refiner_dfeat"] = dfeat_norm.detach()


        # print(
        #     f"scene scale: {scene_scale:.3f}, pixel-wise num: {h*w*v}, after voxelize: {neural_pts.shape[1]}, voxelize ratio: {infos['voxelize_ratio']:.3f}"
        # )
        # print(
        #     f"Gaussians attributes: \n"
        #     f"opacities: mean: {gaussians.opacities.mean()}, min: {gaussians.opacities.min()}, max: {gaussians.opacities.max()} \n"
        #     f"scales: mean: {gaussians.scales.mean()}, min: {gaussians.scales.min()}, max: {gaussians.scales.max()}"
        # )

        # Add target views back
        if target_image is not None:
            # print("Adding target views back, extrinsic", extrinsic.shape, "extrinsic_target", extrinsic_target.shape, "intrinsic", intrinsic.shape, "intrinsic_target", intrinsic_target.shape, flush=True)
            extrinsic = torch.cat((extrinsic, extrinsic_target), dim=1)
            intrinsic = torch.cat((intrinsic, intrinsic_target), dim=1)
            # print("Added target views back, extrinsic", extrinsic.shape, "intrinsic", intrinsic.shape, flush=True)

        # print("B:", b, "V:", v, "H:", h, "W:", w)
        extrinsic_padding = (torch.tensor([0, 0, 0, 1], device=device, dtype=extrinsic.dtype).view(1, 1, 1, 4).repeat(b, extrinsic.shape[1], 1, 1))
        extrinsic = torch.cat([extrinsic, extrinsic_padding], dim=2).inverse() #w2c -> c2w
        intrinsic = intrinsic.clone()  # Create a new tensor
        intrinsic = torch.stack([intrinsic[:, :, 0] / w, intrinsic[:, :, 1] / h, intrinsic[:, :, 2]], dim=2)

        if self.predict_uncertainty:
            print("gaussians.means", gaussians.means.shape, gaussians.means.min(), gaussians.means.max(),
                "gaussians.opacities", gaussians.opacities.shape, gaussians.opacities.min(), gaussians.opacities.max(),
                "gaussians.vars", gaussians.vars.shape, gaussians.vars.min(), gaussians.vars.max())
            
        encoder_output = EncoderOutput(
            gaussians=gaussians,
            pred_pose_enc_list=pred_pose_enc_list, #w2c
            pred_context_pose=dict(extrinsic=extrinsic, intrinsic=intrinsic), #c2w
            depth_dict=dict(depth=depth_map, depth_conf=depth_conf, conf_valid_mask=conf_valid_mask),
            infos=infos,
            distill_infos=distill_infos,
            bg_dict=dict(bg=bg, bg_conf=bg_conf),
            static_dict=dict(static_logit=static_logit) if self.predict_static else None,
        )

        if visualization_dump is not None:
            return encoder_output, visualization_dump
        else:
            return encoder_output

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim

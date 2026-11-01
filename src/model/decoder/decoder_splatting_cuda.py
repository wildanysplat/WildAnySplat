from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
import torchvision

from ..types import Gaussians
# from .cuda_splatting import DepthRenderingMode, render_cuda
from .decoder import Decoder, DecoderOutput
from math import sqrt 
from gsplat import rasterization, rasterization_2dgs


DepthRenderingMode = Literal["depth", "disparity", "relative_disparity", "log"]

def to_gib(value: int) -> float:
    return value / 1024**3

@dataclass
class DecoderSplattingCUDACfg:
    name: Literal["splatting_cuda"]
    background_color: list[float]
    make_scale_invariant: bool


class DecoderSplattingCUDA(Decoder[DecoderSplattingCUDACfg]):
    background_color: Float[Tensor, "3"]
    
    def __init__(
        self,
        cfg: DecoderSplattingCUDACfg,
    ) -> None:
        super().__init__(cfg)
        self.make_scale_invariant = cfg.make_scale_invariant
        self.register_buffer(
            "background_color",
            torch.tensor(cfg.background_color, dtype=torch.float32),
            persistent=False,
        )

    def rendering_fn(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"], #c2w
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        B, V, _, _  = intrinsics.shape
        H, W = image_shape
        rendered_imgs, rendered_depths, rendered_alphas, = [], [], []
        rendered_vars = []
        rendered_normals, rendered_surf_normals = [], []
        xyzs, opacitys, rotations, scales, features = gaussians.means, gaussians.opacities, gaussians.rotations, gaussians.scales, gaussians.harmonics.permute(0, 1, 3, 2).contiguous()
        covariances = gaussians.covariances
        use_2dgs = covariances is None
        for i in range(B):
            xyz_i = xyzs[i].float()
            feature_i = features[i].float()
            covar_i = None if covariances is None else covariances[i].float()
            scale_i = scales[i].float()
            rotation_i = rotations[i].float()
            opacity_i = opacitys[i].squeeze().float()
            test_w2c_i = extrinsics[i].float().inverse() # w2c (V, 4, 4)
            test_intr_i_normalized = intrinsics[i].float()

            vars_i = None
            if getattr(gaussians, "vars", None) is not None:
                vars_i = gaussians.vars[i].float()  # (N,)

            # Denormalize the intrinsics into standred format
            test_intr_i = test_intr_i_normalized.clone()
            test_intr_i[:, 0] = test_intr_i_normalized[:, 0] * W
            test_intr_i[:, 1] = test_intr_i_normalized[:, 1] * H
            sh_degree = (int(sqrt(feature_i.shape[-2])) - 1)
            # K = feature_i.shape[-2]
            # print(f"[rendering_fn] checking SH coeff count {K} is same as {(sh_degree + 1) ** 2}", flush=True)

            rendering_list = []
            rendering_depth_list = []
            rendering_alpha_list = []
            rendering_var_list = []
            rendering_normal_list = []
            rendering_surf_normal_list = []
            for j in range(V):

                # print(
                #     f"[before rasterization] "
                #     f"batch={i} view={j}/{V} "
                #     f"N={xyz_i.shape[0]} "
                #     f"resolution={H}x{W} "
                #     f"allocated={to_gib(torch.cuda.memory_allocated()):.2f} GiB "
                #     f"reserved={to_gib(torch.cuda.memory_reserved()):.2f} GiB "
                #     f"max_allocated={to_gib(torch.cuda.max_memory_allocated()):.2f} GiB",
                #     flush=True,
                # )

                # 2DGS path
                if use_2dgs:
                    # print("[rendering_fn] using 2DGS")
                    rendering, alpha, render_normals, surf_normals, distort, median_depth, meta = rasterization_2dgs(
                        xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                        test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H, sh_degree=sh_degree,
                        render_mode="RGB+D", packed=False, #False,
                        near_plane=1e-10,
                        far_plane=1e10,
                        radius_clip=0.1,
                        backgrounds=torch.cat([self.background_color, torch.zeros(1, device=xyz_i.device, dtype=torch.float32)], dim=0).unsqueeze(0)  # (1, 4)
                        # depth_mode="expected",   # or "median"
                        )
                    if render_normals is not None:
                        if render_normals.dim() == 3:          # (H, W, 3)
                            render_normals = render_normals.unsqueeze(0)
                        rendering_normal_list.append(render_normals.permute(0, 3, 1, 2))
                    if surf_normals is not None:
                        if surf_normals.dim() == 3:            # (H, W, 3)
                            surf_normals = surf_normals.unsqueeze(0)
                        rendering_surf_normal_list.append(surf_normals.permute(0, 3, 1, 2))

                # AnySplat
                else:
                    # print("[rendering_fn] using 3DGS")
                    rendering, alpha, meta = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                                                    test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H, sh_degree=sh_degree, 
                                                    # near_plane=near[i].mean(), far_plane=far[i].mean(),
                                                    render_mode="RGB+D", packed=False, #False,
                                                    near_plane=1e-10,
                                                    backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                                                    radius_clip=0.1,
                                                    covars=covar_i,
                                                    rasterize_mode='classic') # (V, H, W, 3)


                # rendering, alpha, meta = rasterization(xyz_i, rotation_i, scale_i, opacity_i, feature_i,
                #                                 test_w2c_i[j:j+1], test_intr_i[j:j+1], W, H, sh_degree=sh_degree, 
                #                                 # near_plane=near[i].mean(), far_plane=far[i].mean(),
                #                                 render_mode="RGB+D", packed=False,
                #                                 near_plane=1e-10,
                #                                 backgrounds=self.background_color.unsqueeze(0).repeat(1, 1),
                #                                 radius_clip=0.1,
                #                                 covars=covar_i,
                #                                 vars=vars_i,
                #                                 rasterize_mode='classic') # (V, H, W, 3) 
                
                rendering_img, rendering_depth = torch.split(rendering, [3, 1], dim=-1)
                rendering_img = rendering_img.clamp(0.0, 1.0)
                rendering_list.append(rendering_img.permute(0, 3, 1, 2))
                rendering_depth_list.append(rendering_depth)
                rendering_alpha_list.append(alpha)

                var_img = meta.get("render_vars", None)
                if var_img is not None:
                    print("rendered var_img", var_img.shape, var_img.min(), var_img.max())
                    rendering_var_list.append(var_img)

            rendered_depths.append(torch.cat(rendering_depth_list, dim=0).squeeze())
            rendered_imgs.append(torch.cat(rendering_list, dim=0))
            rendered_alphas.append(torch.cat(rendering_alpha_list, dim=0).squeeze())

            # var
            if len(rendering_var_list) > 0:
                rendered_vars.append(torch.cat(rendering_var_list, dim=0).squeeze())
            else:
                rendered_vars.append(None)

            # render normals
            if len(rendering_normal_list) > 0:
                rendered_normals.append(torch.cat(rendering_normal_list, dim=0))
                
            else:
                rendered_normals.append(None)

            # surf normals
            if len(rendering_surf_normal_list) > 0:
                rendered_surf_normals.append(torch.cat(rendering_surf_normal_list, dim=0))
            else:
                rendered_surf_normals.append(None)
                
        return DecoderOutput(
            torch.stack(rendered_imgs),
            torch.stack(rendered_depths),
            torch.stack(rendered_alphas),
            conf=None,
            lod_rendering=None,
            vars=None if any(v is None for v in rendered_vars) else torch.stack(rendered_vars),
            color_aug=None,
            render_normals=None if any(n is None for n in rendered_normals) else torch.stack(rendered_normals),
            surf_normals=None if any(n is None for n in rendered_surf_normals) else torch.stack(rendered_surf_normals),
            )

    def forward(
        self,
        gaussians: Gaussians,
        extrinsics: Float[Tensor, "batch view 4 4"],
        intrinsics: Float[Tensor, "batch view 3 3"],
        near: Float[Tensor, "batch view"],
        far: Float[Tensor, "batch view"],
        image_shape: tuple[int, int],
        depth_mode: DepthRenderingMode | None = None,
        cam_rot_delta: Float[Tensor, "batch view 3"] | None = None,
        cam_trans_delta: Float[Tensor, "batch view 3"] | None = None,
    ) -> DecoderOutput:
        
        return self.rendering_fn(gaussians, extrinsics, intrinsics, near, far, image_shape, depth_mode, cam_rot_delta, cam_trans_delta)


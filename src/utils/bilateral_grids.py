import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F


class BilateralGrid(nn.Module):
    def __init__(self):
        super(BilateralGrid, self).__init__()
        # self.rgb2gray_weight = torch.Tensor([[0.299, 0.587, 0.114]])
        self.register_buffer("rgb2gray_weight", torch.tensor([[0.299, 0.587, 0.114]], dtype=torch.float32), persistent=False)
        self.rgb2gray = lambda rgb: (rgb @ self.rgb2gray_weight.T) * 2. - 1.
        self.s_max = 1.0

        I12 = torch.tensor(
            [1,0,0,0,
             0,1,0,0,
             0,0,1,0],
            dtype=torch.float32
        ).view(1, 12, 1, 1)
        self.register_buffer("I12", I12, persistent=False)

    def edge_aware_guidance(self, rgb_map):
        gray = (rgb_map @ torch.Tensor([0.299, 0.587, 0.114]).to(rgb_map.device).view(1,3,1,1))
        sobel_x = F.conv2d(gray, weight=torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]]).to(rgb_map.device), padding=1)
        sobel_y = F.conv2d(gray, weight=torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]]).to(rgb_map.device), padding=1)
        edge = torch.sqrt(sobel_x ** 2 + sobel_y ** 2)
        edge_norm = edge / edge.max()
        return edge_norm * 2. - 1.

    def slice(self, bilateral_grid, rgb_map):
        # device = bilateral_grid.get_device()
        device = bilateral_grid.device
        dtype = bilateral_grid.dtype

        N, _, H, W = rgb_map.shape
        # hg, wg = torch.meshgrid([torch.arange(0, H), torch.arange(0, W)])  # [0,511] HxW
        # if device >= 0:
        #     hg = hg.to(device)
        #     wg = wg.to(device)
        hg, wg = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij",
        )
        # hg = hg.float().repeat(N, 1, 1).unsqueeze(3) / (H - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        # wg = wg.float().repeat(N, 1, 1).unsqueeze(3) / (W - 1) * 2 - 1  # norm to [-1,1] NxHxWx1
        hg = hg[None, :, :, None].expand(N, -1, -1, -1) / (H - 1) * 2 - 1
        wg = wg[None, :, :, None].expand(N, -1, -1, -1) / (W - 1) * 2 - 1

        # rgb_map = rgb_map.permute(0, 2, 3, 1).contiguous()
        # zg = (rgb_map @ self.rgb2gray_weight.T.to(device)) * 2. - 1.
        rgb_map = rgb_map.to(device=device, dtype=dtype).permute(0, 2, 3, 1).contiguous()
        weight = self.rgb2gray_weight.to(device=rgb_map.device, dtype=rgb_map.dtype)
        zg = (rgb_map @ weight.T) * 2. - 1.

        guidemap_guide = torch.cat([wg, hg, zg], dim=3).unsqueeze(1)  # Nx1xHxWx3

        coeff = F.grid_sample(bilateral_grid, guidemap_guide, mode='bilinear', align_corners=True, padding_mode='border')
        return coeff.squeeze(2)

    def apply_coeffs(self, coeff, full_res_input):
        if coeff.shape[1] == 1:
            output = torch.sum(coeff * full_res_input, dim=1, keepdim=True)

        else:
            full_res_input = full_res_input.permute(0, 2, 3, 1)
            coeff = coeff.reshape(coeff.shape[0], 3, 4, coeff.shape[2], coeff.shape[3])
            coeff = coeff.permute(0, 3, 4, 1, 2)
            output = torch.matmul(coeff[..., :3], full_res_input.unsqueeze(-1)).squeeze(-1) + coeff[..., 3]
            output = output.permute(0, 3, 1, 2)
        return output
    
    def apply_bilateral_grids(self, bg_dict, image, s: float, residual: bool = True):
        """
        Apply bilateral grids
        s: residual scale (e.g., 0.05..0.5)
        residual=True: use A = I + s*tanh(ΔA). If False, behave like your original code.
        """
        bg, bg_conf = bg_dict["bg"], bg_dict["bg_conf"]
        b, v, c, h, w = image.shape
        image_conf = None

        if bg_conf is not None:
            image_conf = self.slice(
                bg_conf.reshape(b * v, 1, 8, 16, 16),
                image.reshape(b * v, c, h, w)
            ).reshape(b, v, 1, h, w)

        if bg is not None:
            if residual:
                print("Applying residual bg", bg.shape, " to image", image.shape, image.min(), image.max(), "with s", s)
            else:
                print("Applying bg", bg.shape, " to image", image.shape, image.min(), image.max(), "with s", s)
            slice_coeffs = self.slice(
                bg.reshape(b * v, 12, 8, 16, 16),
                image.reshape(b * v, c, h, w)
            )

            if residual:
                # ΔA(x) -> A(x) = I + s*tanh(ΔA(x))
                # (tanh bounds; s sets maximum deviation)
                delta = torch.tanh(slice_coeffs) * float(s)
                I12 = self.I12.to(device=slice_coeffs.device, dtype=slice_coeffs.dtype)
                coeffs = I12 + delta
            else:
                coeffs = slice_coeffs  # original behavior (absolute)

            image = (self.apply_coeffs(
                coeffs,
                image.reshape(b * v, c, h, w)
            )).reshape(b, v, c, h, w).clamp(0,1)
            print("New image", image.shape, image.min(), image.max())

        return image, image_conf
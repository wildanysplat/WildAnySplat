"""Gaussian-splat PLY and coloured point-cloud PLY exports.

The WildAnySplat model predicts quaternion rotations in XYZW order and SH
coefficients in [N, RGB, (degree+1)**2] order. Standard 3DGS PLY expects
WXYZ quaternions, channel-major SH coefficients, log-scales and logit opacity.

Most Gaussian Splat editors (including SuperSplat) support SH degree <=3;
WildAnySplat internally predicts degree 4. Degree 3 is the default for PLY
interchange. The model's renderer still uses its full SH degree.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

SH_C0 = 0.28209479177387814


def construct_list_of_attributes(num_rest: int) -> list[str]:
    return (["x", "y", "z", "nx", "ny", "nz"]
            + [f"f_dc_{i}" for i in range(3)]
            + [f"f_rest_{i}" for i in range(num_rest)]
            + ["opacity"]
            + [f"scale_{i}" for i in range(3)]
            + [f"rot_{i}" for i in range(4)])


def _ply_header(count: int, fields: list[tuple[str, str]]) -> bytes:
    # Only binary little-endian PLY; property definitions in exact record order.
    lines = ["ply", "format binary_little_endian 1.0",
             "comment WildAnySplat Gaussian Splat export",
             f"element vertex {count}"]
    lines.extend(f"property {kind} {name}" for name, kind in fields)
    lines.append("end_header")
    return ("\n".join(lines) + "\n").encode("ascii")


def sh_dc_to_rgb(dc: np.ndarray) -> np.ndarray:
    """SH DC-to-RGB mapping used by standard 3D Gaussian Splat viewers."""
    return np.clip(0.5 + SH_C0 * dc, 0.0, 1.0)


@torch.no_grad()
def export_ply(
    means: Tensor,
    scales: Tensor,
    rotations: Tensor,
    harmonics: Tensor,
    opacities: Tensor,
    path: Path,
    shift_and_scale: bool = False,
    save_sh_dc_only: bool = False,
    sh_degree: int = 3,
    min_opacity: float = 0.0,
    rgb_pointcloud_path: Path | None = None,
    chunk_size: int = 65536,
) -> dict[str, Any]:
    """Export a standard Gaussian Splat PLY and optional MeshLab RGB PLY.

    Args:
        scales: *linear* Gaussian standard deviations, [N,3].
        rotations: quaternion in XYZW order, [N,4], matching the model's
            ``build_covariance`` implementation. Converted to WXYZ for PLY.
        harmonics: [N,3,K] with K a square (25 for WildAnySplat degree 4).
        opacities: *linear* alpha in [0,1], [N].
        sh_degree: SH degree written to PLY (3 default for SuperSplat).
            The original decoder still renders with degree 4 independently.
        rgb_pointcloud_path: optional ordinary RGB-vertex PLY for MeshLab.

    Writes in small CPU chunks, without constructing millions of Python tuples.
    Invalid (NaN/Inf/nonpositive scale/zero quaternion) entries are dropped.
    """
    path = Path(path)
    if rgb_pointcloud_path is not None:
        rgb_pointcloud_path = Path(rgb_pointcloud_path)
        if rgb_pointcloud_path.resolve() == path.resolve():
            raise ValueError("RGB point-cloud path must differ from Gaussian PLY path")
    if not 0.0 <= min_opacity < 1.0:
        raise ValueError("min_opacity must be in [0,1)")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if means.ndim != 2 or means.shape[1] != 3:
        raise ValueError("means must have shape [N,3]")
    n = means.shape[0]
    if (scales.shape != (n, 3) or rotations.shape != (n, 4)
            or opacities.reshape(-1).shape != (n,) or
            harmonics.ndim != 3 or harmonics.shape[:2] != (n, 3)):
        raise ValueError("Unexpected Gaussian attribute shape")
    sh_count = harmonics.shape[2]
    model_degree = math.isqrt(sh_count) - 1
    if (model_degree + 1) ** 2 != sh_count:
        raise ValueError(f"SH coefficient count {sh_count} is not a perfect square")
    degree = 0 if save_sh_dc_only else int(sh_degree)
    if not (0 <= degree <= model_degree):
        raise ValueError(f"PLY SH degree must be between 0 and model degree {model_degree}")

    selected_sh = (degree + 1) ** 2
    alpha = opacities.reshape(-1)
    valid = (torch.isfinite(means).all(-1)
             & torch.isfinite(scales).all(-1)
             & (scales > 0).all(-1)
             & torch.isfinite(rotations).all(-1)
             & (torch.linalg.vector_norm(rotations.float(), dim=-1) > 1e-8)
             & torch.isfinite(alpha)
             & (alpha >= min_opacity)
             & torch.isfinite(harmonics[:, :, :selected_sh]).all(dim=(1, 2)))
    selected = torch.nonzero(valid, as_tuple=True)[0]
    count = selected.numel()
    if count == 0:
        raise ValueError("No valid Gaussians remain for PLY export")

    # No coordinate transform by default: predicted cameras and the PLY must
    # remain in the same coordinate system. The old shift_and_scale option is
    # intentionally disallowed rather than silently breaking that alignment.
    if shift_and_scale:
        raise ValueError("shift_and_scale is not supported: it changes the camera coordinate frame")

    n_rest = 3 * (selected_sh - 1)
    names = construct_list_of_attributes(n_rest)
    cols = len(names)
    point_dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                            ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    path.parent.mkdir(parents=True, exist_ok=True)
    if rgb_pointcloud_path is not None:
        rgb_pointcloud_path.parent.mkdir(parents=True, exist_ok=True)

    # Both file formats use a fixed schema and a predetermined vertex count.
    from contextlib import nullcontext
    point_file = open(rgb_pointcloud_path, "wb") if rgb_pointcloud_path else nullcontext()
    with open(path, "wb") as gaussian_file, point_file as point_out:
        gaussian_file.write(_ply_header(count, [(name, "float") for name in names]))
        if rgb_pointcloud_path is not None:
            point_out.write(_ply_header(count, [
                ("x", "float"), ("y", "float"), ("z", "float"),
                ("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]))
        for start in range(0, count, chunk_size):
            ids = selected[start:start + chunk_size]
            xyz = means[ids].detach().float().cpu().numpy()
            sc = scales[ids].detach().float().cpu().numpy()
            q = rotations[ids].detach().float().cpu().numpy()  # XYZW
            sh = harmonics[ids, :, :selected_sh].detach().float().cpu().numpy()
            op = alpha[ids].detach().float().cpu().numpy()
            q /= np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-8)
            q_wxyz = q[:, [3, 0, 1, 2]]
            op = np.log(np.clip(op, 1e-6, 1 - 1e-6) / (1 - np.clip(op, 1e-6, 1 - 1e-6)))

            values = np.empty((len(ids), cols), dtype="<f4")
            j = 0
            values[:, j:j+3] = xyz; j += 3
            values[:, j:j+3] = 0; j += 3
            values[:, j:j+3] = sh[:, :, 0]; j += 3
            if n_rest:
                values[:, j:j+n_rest] = sh[:, :, 1:].reshape(len(ids), -1)
                j += n_rest
            values[:, j] = op; j += 1
            values[:, j:j+3] = np.log(sc); j += 3
            values[:, j:j+4] = q_wxyz; j += 4
            assert j == cols
            values.tofile(gaussian_file)

            if rgb_pointcloud_path is not None:
                pts = np.empty(len(ids), dtype=point_dtype)
                for axis, name in enumerate(("x", "y", "z")):
                    pts[name] = xyz[:, axis]
                rgb = np.rint(sh_dc_to_rgb(sh[:, :, 0]) * 255).astype(np.uint8)
                for axis, name in enumerate(("red", "green", "blue")):
                    pts[name] = rgb[:, axis]
                pts.tofile(point_out)

    return {
        "input_gaussians": int(n),
        "exported_gaussians": int(count),
        "dropped_invalid_or_below_opacity": int(n - count),
        "model_sh_degree": int(model_degree),
        "ply_sh_degree": int(degree),
        "properties": len(names),
        "gaussian_ply": str(path),
        "rgb_pointcloud_ply": str(rgb_pointcloud_path) if rgb_pointcloud_path else None,
    }

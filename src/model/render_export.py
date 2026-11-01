"""Render WildAnySplat reconstructions and NVS videos using its own decoder.

The NVS trajectory follows src/misc/image_io.py's original
``save_interpolated_video`` approach: intermediate camera poses between
neighbouring predicted context cameras. Rotations use proper SO(3) SLERP and
frames are rendered one at a time to avoid OOM on long sequences.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterator

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp


def interpolate_camera_trajectory(
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    between: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate camera-to-world matrices and normalized intrinsics.

    Includes the original first and last input cameras exactly once, and
    ``between`` new views between each pair. Input order is the supplied
    image order (or video timestamp order). No camera reordering is attempted.
    """
    c2w = np.asarray(extrinsics, dtype=np.float64)
    kn = np.asarray(intrinsics, dtype=np.float64)
    if c2w.ndim != 3 or c2w.shape[1:] != (4, 4):
        raise ValueError("extrinsics must be [views,4,4]")
    if kn.shape != (len(c2w), 3, 3) or len(c2w) < 2:
        raise ValueError("intrinsics must be [views,3,3] for at least two views")
    if between < 0:
        raise ValueError("between must be >= 0")
    if not np.isfinite(c2w).all() or not np.isfinite(kn).all():
        raise ValueError("Camera trajectory contains nonfinite values")
    output_c2w, output_k = [], []
    for i in range(len(c2w) - 1):
        r = Rotation.from_matrix(c2w[i:i+2, :3, :3])
        slerp = Slerp([0., 1.], r)
        for j in range(between + 1):
            t = j / (between + 1)
            p = np.eye(4, dtype=np.float32)
            p[:3, :3] = slerp([t]).as_matrix()[0]
            p[:3, 3] = ((1 - t) * c2w[i, :3, 3] + t * c2w[i+1, :3, 3])
            output_c2w.append(p)
            output_k.append(((1-t)*kn[i] + t*kn[i+1]).astype(np.float32))
    output_c2w.append(c2w[-1].astype(np.float32))
    output_k.append(kn[-1].astype(np.float32))
    return np.stack(output_c2w, axis=0), np.stack(output_k, axis=0)


@torch.inference_mode()
def render_one(model, gaussians, c2w, kn, size: int = 448):
    """Return RGB (uint8 HWC) and raw depth (float32 HW) from the model decoder."""
    device = gaussians.means.device
    pose = torch.as_tensor(c2w, device=device, dtype=torch.float32).reshape(1, 1, 4, 4)
    intr = torch.as_tensor(kn, device=device, dtype=torch.float32).reshape(1, 1, 3, 3)
    near = torch.full((1, 1), 0.1, device=device, dtype=torch.float32)
    far = torch.full((1, 1), 100.0, device=device, dtype=torch.float32)
    result = model.decoder(gaussians, pose, intr, near, far, (size, size), "depth")

    # ``DecoderSplattingCUDA`` returns color [B,V,3,H,W]. Its depth uses
    # squeeze() internally, so for V=1 it can be [B,H,W] or [H,W].
    rgb = result.color[0, 0].detach().float().clamp(0.0, 1.0)
    if tuple(rgb.shape) != (3, size, size):
        raise RuntimeError(f"Unexpected rendered color shape: {tuple(rgb.shape)}")
    rgb8 = (rgb.permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    depth = result.depth.detach().float().cpu().numpy().reshape(-1, size, size)[0]
    return rgb8, depth


def estimate_depth_range(depths: list[np.ndarray]) -> tuple[float, float]:
    """Stable scene-wise limits for depth-video colouring."""
    # Subsample pixels to keep this cheap with many input views.
    valid_chunks = []
    for d in depths:
        vals = np.asarray(d)[::4, ::4].ravel()
        vals = vals[np.isfinite(vals) & (vals > 0) & (vals < 1e8)]
        if len(vals):
            valid_chunks.append(vals)
    if not valid_chunks:
        return 0.1, 10.0
    vals = np.concatenate(valid_chunks)
    low, high = np.quantile(vals, [0.01, 0.99]).tolist()
    if high <= low + 1e-6:
        high = low + 1.0
    return float(low), float(high)


def depth_to_rgb(depth: np.ndarray, low: float, high: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0) & (depth < 1e8)
    normalized = np.clip((depth - low) / max(1e-6, high - low), 0, 1)
    gray = np.rint(np.where(valid, normalized, 0) * 255).astype(np.uint8)
    rgb = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    rgb[~valid] = 0
    return rgb


def _video_writer(path: Path, fps: int):
    """H264/yuv420p for browser players; needs imageio-ffmpeg."""
    return imageio.get_writer(
        str(path), format="FFMPEG", mode="I", fps=fps,
        codec="libx264", pixelformat="yuv420p", macro_block_size=1,
        output_params=["-movflags", "+faststart"],
        ffmpeg_log_level="error",
    )


def render_input_view_images(
    model, gaussians, poses: dict, images: torch.Tensor,
    outdir: Path, size: int = 448,
) -> tuple[list[np.ndarray], list[Path]]:
    """Save rendered PNGs and side-by-side input/reconstruction PNGs."""
    input_folder = outdir / "rendered_input_views"
    comparison_folder = outdir / "input_vs_render"
    input_folder.mkdir(parents=True, exist_ok=True)
    comparison_folder.mkdir(parents=True, exist_ok=True)
    c2ws = poses["extrinsic"][0].detach().cpu().float().numpy()
    ks = poses["intrinsic"][0].detach().cpu().float().numpy()
    if len(images[0]) != len(c2ws):
        raise ValueError("Input image count does not match predicted cameras")
    depths, written = [], []
    for i, (c2w, kn) in enumerate(zip(c2ws, ks)):
        rgb8, d = render_one(model, gaussians, c2w, kn, size)
        out_path = input_folder / f"view_{i:04d}.png"
        Image.fromarray(rgb8).save(out_path)
        src = (images[0, i].detach().float().clamp(0, 1)
               .permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
        if src.shape[:2] != rgb8.shape[:2]:
            src = np.asarray(Image.fromarray(src).resize((size, size), Image.Resampling.BICUBIC))
        Image.fromarray(np.concatenate((src, rgb8), axis=1)).save(
            comparison_folder / f"view_{i:04d}_input_vs_render.png")
        depths.append(d)
        written.append(out_path)
    return depths, written


def render_nvs_video(
    model, gaussians, poses: dict, outdir: Path,
    between: int = 10, fps: int = 24, size: int = 448,
    depth_video: bool = True, reference_depths: list[np.ndarray] | None = None,
    save_frames: bool = False,
) -> dict:
    if fps <= 0:
        raise ValueError("fps must be > 0")
    if size <= 0 or size % 2:
        raise ValueError("render size must be positive and even for H.264")
    outdir.mkdir(parents=True, exist_ok=True)
    c2ws = poses["extrinsic"][0].detach().cpu().float().numpy()
    kn = poses["intrinsic"][0].detach().cpu().float().numpy()
    path, path_k = interpolate_camera_trajectory(c2ws, kn, between)
    np.savez_compressed(outdir / "nvs_cameras.npz", c2w=path,
                        K_normalized=path_k, size=np.int32(size), fps=np.int32(fps))
    video_path = outdir / "rgb.mp4"
    depth_path = outdir / "depth.mp4"
    frame_dir = outdir / "nvs_frames"
    if save_frames:
        frame_dir.mkdir(parents=True, exist_ok=True)
    depth_bounds = estimate_depth_range(reference_depths) if reference_depths else None

    # If input renders were disabled, use first trajectory frame for a stable
    # depth-colour range. This avoids per-frame flickering colour maps.
    first_rgb, first_depth = render_one(model, gaussians, path[0], path_k[0], size)
    if depth_bounds is None:
        depth_bounds = estimate_depth_range([first_depth])
    from contextlib import ExitStack
    with ExitStack() as stack:
        rgb_writer = stack.enter_context(_video_writer(video_path, fps))
        depth_writer = stack.enter_context(_video_writer(depth_path, fps)) if depth_video else None
        for i, (c2w, k) in enumerate(zip(path, path_k)):
            rgb8, depth = (first_rgb, first_depth) if i == 0 else render_one(
                model, gaussians, c2w, k, size)
            rgb_writer.append_data(rgb8)
            if depth_writer:
                depth_writer.append_data(depth_to_rgb(depth, *depth_bounds))
            if save_frames:
                Image.fromarray(rgb8).save(frame_dir / f"frame_{i:04d}.png")
            if (i + 1) % 20 == 0 or i + 1 == len(path):
                print(f"NVS rendering: {i+1}/{len(path)} frames", flush=True)

    return {
        "rgb": "rgb.mp4", "depth": "depth.mp4" if depth_video else None,
        "camera_path": "nvs_cameras.npz", "frames": len(path), "fps": fps,
        "interpolated_views_per_pair": between, "depth_range": list(depth_bounds),
        "trajectory": "interpolation between adjacent predicted input cameras",
    }

#!/usr/bin/env python3
"""WildAnySplat inference-only CLI: image/video -> 3DGS, cameras, renders and NVS MP4.

This uses the trained WildAnySplat model, never the unmodified AnySplat weights.
No training datasets, Lightning, Hydra or logger code is imported.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
HERE = Path(__file__).resolve().parent


def numeric_sort_key(path: Path) -> tuple:
    return tuple(int(part) if part.isdigit() else part.casefold()
                 for part in re.split(r"(\d+)", path.name))


def uniform_indices(count: int, limit: int) -> list[int]:
    """Spread at most limit viewpoints across the entire image/video sequence."""
    if count < 2:
        raise ValueError("A reconstruction requires at least 2 input frames/images.")
    if limit < 2:
        raise ValueError("--num-views must be >= 2")
    return np.linspace(0, count - 1, min(count, limit), dtype=int).tolist()


def load_image_paths(folder: Path, num_views: int) -> list[Path]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    paths = sorted((p for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
                   key=numeric_sort_key)
    if len(paths) < 2:
        raise ValueError(f"Need at least 2 images in {folder}; found {len(paths)}")
    return [paths[i] for i in uniform_indices(len(paths), num_views)]


def extract_video_frames(video: Path, output: Path, num_views: int) -> list[Path]:
    """Read selected frames via OpenCV. Avoid dumping an entire video to disk."""
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV cannot open video: {video}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count < 2:
        cap.release()
        raise RuntimeError("Could not determine >=2 video frames. Check video codec and container.")
    output.mkdir(parents=True, exist_ok=True)
    selected = uniform_indices(frame_count, num_views)
    paths = []
    try:
        for i, frame_id in enumerate(selected):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_id))
            ok, bgr = cap.read()
            if not ok or bgr is None:
                raise RuntimeError(f"Failed to decode video frame {frame_id} from {video}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            path = output / f"frame_{i:04d}.png"
            Image.fromarray(rgb).save(path)
            paths.append(path)
    finally:
        cap.release()
    return paths


def preprocess_image(path: Path, size: int = 448):
    """Match the 448-square aspect-preserving centre crop in src.utils.image."""
    import torch

    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    if width > height:
        new_height = size
        new_width = int(width * size / height)
    else:
        new_width = size
        new_height = int(height * size / width)
    image = image.resize((new_width, new_height), resample=Image.Resampling.BICUBIC)
    left, top = (new_width - size) // 2, (new_height - size) // 2
    image = image.crop((left, top, left + size, top + size))
    rgb = np.asarray(image, dtype=np.uint8).copy()
    # Equivalent to torchvision.transforms.ToTensor() * 2 - 1; model.inference
    # receives [0,1] as in the user's wildanysplat/inference.py.
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div_(255.0)
    return tensor


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--images", type=Path, help="Folder of RGB images")
    source.add_argument("--video", type=Path, help="Input video (.mp4, .mov, etc.)")
    weights = p.add_mutually_exclusive_group(required=True)
    weights.add_argument("--checkpoint", type=Path, help="Local model.safetensors produced by scripts/prepare_checkpoint.py")
    weights.add_argument("--hf-repo", help="Hugging Face model repo, e.g. your-org/WildAnySplat")
    p.add_argument("--config", type=Path, default=None, help="Optional matching local inference_config.yaml; otherwise uses the Hub config or packaged template")
    p.add_argument("--num-views", type=int, default=12, help="Maximum number of evenly sampled views")
    p.add_argument("--output", type=Path, default=Path("outputs/demo"), help="Directory for reconstruction files")
    p.add_argument("--dc-only", action="store_true", help="Use SH degree 0 (DC-only) for the exported Gaussian PLY")
    p.add_argument("--ply-sh-degree", type=int, choices=range(5), default=3,
                   help="PLY SH degree: 3 is broadly SuperSplat-compatible; internal rendering retains SH degree 4")
    p.add_argument("--min-opacity", type=float, default=1.0 / 255.0,
                   help="Drop Gaussians whose sigmoid opacity is below this threshold before exporting PLY files")
    p.add_argument("--rgb-pointcloud", action=argparse.BooleanOptionalAction, default=True,
                   help="Also write points_rgb.ply for MeshLab with ordinary RGB vertex colours")
    p.add_argument("--render-input-views", action=argparse.BooleanOptionalAction, default=True,
                   help="Render predicted input cameras, and save input-vs-render PNGs")
    p.add_argument("--nvs", action=argparse.BooleanOptionalAction, default=True,
                   help="Render interpolated NVS trajectory and save rgb.mp4")
    p.add_argument("--nvs-between", type=int, default=10,
                   help="Interpolated camera views per adjacent input pair (as in original eval code)")
    p.add_argument("--nvs-fps", type=int, default=24, help="Frames per second for NVS MP4s")
    p.add_argument("--nvs-depth", action=argparse.BooleanOptionalAction, default=True,
                   help="Also write colourised depth.mp4 for the NVS trajectory")
    p.add_argument("--save-nvs-frames", action="store_true", help="Save every NVS frame as PNG")
    p.add_argument("--render-size", type=int, default=448,
                   help="Square RGB render resolution (even for H.264, default 448)")
    return p.parse_args(argv)


def resolve_weights_and_config(args):
    if args.hf_repo:
        from huggingface_hub import hf_hub_download
        checkpoint = Path(hf_hub_download(repo_id=args.hf_repo, filename="model.safetensors"))
        config = args.config or Path(hf_hub_download(repo_id=args.hf_repo, filename="inference_config.yaml"))
    else:
        checkpoint = args.checkpoint
        config = args.config or HERE / "configs/inference_config.yaml"
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    if not config.is_file():
        raise FileNotFoundError(f"Missing inference configuration: {config}")
    return checkpoint, config


def initialize_model(checkpoint: Path, config: Path):
    import torch
    from omegaconf import OmegaConf
    from safetensors.torch import load_file
    from src.model.model.anysplat import AnySplat

    if not torch.cuda.is_available():
        raise RuntimeError("This release requires a CUDA GPU (torch_scatter + gsplat + VGGT-Omega).")
    cfg = OmegaConf.load(config)
    if cfg.encoder.vggt_variant != "vggt_omega":
        raise ValueError("This minimal runtime supports vggt_omega only.")
    cfg.encoder.distill = False
    cfg.encoder.vggt_omega_checkpoint = ""  # full trained checkpoint includes all backbone weights
    print("Constructing model architecture (this may use substantial host RAM)...", flush=True)
    model = AnySplat(cfg.encoder, cfg.decoder)
    weights = load_file(str(checkpoint), device="cpu")
    # Do not silently load only a fraction of the trained model.
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The published checkpoint does not match inference_config.yaml. "
            "Check bilateral_grid, predict_static, fusion_mode, and the exact "
            "training architecture. A full model checkpoint is required.\n" + str(exc)
        ) from exc
    del weights
    model = model.to(device="cuda").eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def export_cameras(poses: dict, outdir: Path, paths: list[Path]):
    import torch
    # EncoderAnySplat returns 'extrinsic' as camera-to-world and 'intrinsic'
    # normalised to width/height in its 448x448 inference coordinate frame.
    c2w = poses["extrinsic"].detach().cpu().float().numpy()[0]
    kn = poses["intrinsic"].detach().cpu().float().numpy()[0]
    if c2w.shape[0] != len(paths) or kn.shape[0] != len(paths):
        raise RuntimeError("Predicted camera count does not match sampled input views.")
    kp = kn.copy()
    kp[:, 0, :] *= 448
    kp[:, 1, :] *= 448
    np.savez_compressed(outdir / "cameras.npz", c2w=c2w, K_normalized=kn,
                        K_pixels_448=kp, image_files=np.array([p.name for p in paths]))
    (outdir / "cameras.json").write_text(json.dumps({
        "coordinate_system": "model output, camera-to-world OpenCV-style",
        "image_resolution": [448, 448],
        "intrinsics": "normalized in cameras.npz; K_pixels_448 provided for convenience",
        "files": [p.name for p in paths],
    }, indent=2) + "\n")


def run(args: argparse.Namespace) -> None:
    import torch
    from src.model.ply_export import export_ply

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for WildAnySplat inference.")
    outdir = args.output.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    if args.images:
        paths = load_image_paths(args.images, args.num_views)
    else:
        paths = extract_video_frames(args.video, outdir / "sampled_video_frames", args.num_views)
    print(f"Selected {len(paths)} input views:")
    for path in paths:
        print("  ", path)
    images = torch.stack([preprocess_image(p) for p in paths], dim=0).unsqueeze(0).to("cuda")
    ckpt, cfg = resolve_weights_and_config(args)
    model = initialize_model(ckpt, cfg)
    print(f"Inferring {len(paths)} views, shape {tuple(images.shape)}", flush=True)
    with torch.inference_mode():
        # Current AnySplat.inference returns exactly three values in the user's source.
        gaussians, pred_context_pose, bg_dict = model.inference(images)
    # Export only the standard Gaussian PLY (which also works in SuperSplat).
    # Filter near-transparent/invalid Gaussians before writing; leave the
    # original full Gaussian set intact for input-view and NVS rendering.
    means = gaussians.means[0]
    scales = gaussians.scales[0]
    rotations = gaussians.rotations[0]
    harmonics = gaussians.harmonics[0]
    alpha = gaussians.opacities[0]
    valid = (
        torch.isfinite(means).all(dim=1)
        & torch.isfinite(scales).all(dim=1)
        & torch.isfinite(rotations).all(dim=1)
        & torch.isfinite(harmonics).flatten(1).all(dim=1)
        & torch.isfinite(alpha)
        & (alpha >= float(args.min_opacity))
    )
    kept = int(valid.sum().item())
    dropped = int(valid.numel() - kept)
    if kept == 0:
        raise RuntimeError(f"All Gaussians were removed by --min-opacity={args.min_opacity}")

    ply = outdir / "gaussians.ply"
    points_ply = outdir / "points_rgb.ply" if args.rgb_pointcloud else None
    export_info = export_ply(
        means[valid], scales[valid], rotations[valid],
        harmonics[valid], alpha[valid], ply,
        save_sh_dc_only=args.dc_only, sh_degree=args.ply_sh_degree,
        min_opacity=0.0, rgb_pointcloud_path=points_ply)
    print(f"Saved {export_info['exported_gaussians']:,} splats to {ply} "
          f"(SH degree {export_info['ply_sh_degree']}; model degree "
          f"{export_info['model_sh_degree']}; dropped {dropped:,} low-opacity/invalid)", flush=True)
    if points_ply:
        print(f"MeshLab RGB point cloud: {points_ply}", flush=True)

    export_cameras(pred_context_pose, outdir, paths)

    appearance = {}
    if isinstance(bg_dict, dict):
        for key in ("bg", "bg_conf"):
            value = bg_dict.get(key)
            if isinstance(value, torch.Tensor):
                appearance[key] = value.detach().cpu().float().numpy()
    if appearance:
        np.savez_compressed(outdir / "appearance_grids.npz", **appearance)

    from src.model.render_export import render_input_view_images, render_nvs_video
    reference_depths = None
    outputs = ["gaussians.ply", "cameras.npz", "cameras.json"]
    if points_ply:
        outputs.append("points_rgb.ply")
    if appearance:
        outputs.append("appearance_grids.npz")
    if args.render_input_views:
        reference_depths, input_render_paths = render_input_view_images(
            model, gaussians, pred_context_pose, images, outdir, size=args.render_size)
        outputs += ["rendered_input_views/", "input_vs_render/"]
        print(f"Rendered {len(input_render_paths)} input-view images", flush=True)
    nvs_info = None
    if args.nvs:
        nvs_info = render_nvs_video(
            model, gaussians, pred_context_pose, outdir,
            between=args.nvs_between, fps=args.nvs_fps,
            size=args.render_size, depth_video=args.nvs_depth,
            reference_depths=reference_depths,
            save_frames=args.save_nvs_frames)
        outputs += ["rgb.mp4", "nvs_cameras.npz"]
        if args.nvs_depth:
            outputs.append("depth.mp4")
        if args.save_nvs_frames:
            outputs.append("nvs_frames/")

    summary = {
        "checkpoint": str(ckpt), "config": str(cfg),
        "input": str(args.images or args.video), "num_views": len(paths),
        "num_gaussians": int(gaussians.means.shape[1]),
        "gaussian_ply": export_info,
        "nvs": nvs_info,
        "outputs": outputs,
        "note": (
            "gaussians.ply is the standard Gaussian PLY export, with low-opacity "
            "splats removed first. points_rgb.ply contains RGB vertex colours "
            "for MeshLab. Rendered videos use the original full-degree Gaussian model. "
            "This checkpoint has no bilateral-grid prediction head."
        ),
    }
    (outdir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("Output files:", ", ".join(outputs), flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        raise SystemExit(f"WildAnySplat inference error: {exc}") from exc


if __name__ == "__main__":
    main()

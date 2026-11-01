#!/usr/bin/env python3
"""Export ONLY inference state_dict tensors from a trusted Lightning checkpoint.

Typical input: trainer checkpoint with state_dict keys 'model.encoder....'.
Output: model.safetensors suitable for local inference and the Hugging Face Hub.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import shutil
from pathlib import Path


def prepare_state_dict(blob: object) -> dict:
    import torch
    if not isinstance(blob, dict):
        raise ValueError("Expected a checkpoint dictionary or raw state_dict")
    for key in ("state_dict", "model_state_dict"):
        if key in blob and isinstance(blob[key], dict):
            blob = blob[key]
            break
    if not blob or not all(isinstance(k, str) for k in blob):
        raise ValueError("Could not identify weight state_dict")
    if any(k.startswith("model.") for k in blob):
        blob = {k.removeprefix("model."): v for k, v in blob.items() if k.startswith("model.")}
    # Exclude teacher copies and unused training-only components.
    result = {
        k: v.cpu().contiguous()
        for k, v in blob.items()
        if torch.is_tensor(v) and (
            k.startswith("encoder.") or k.startswith("decoder.") or k.startswith("bg.")
        ) and not (
            k.startswith("encoder.distill_")
        )
    }
    if not result or not any(k.startswith("encoder.aggregator.") for k in result):
        raise ValueError("Expected FULL WildAnySplat encoder/aggregator weights, not a partial/adapter-only checkpoint")
    if not any(k.startswith("encoder.gaussian_param_head.") for k in result):
        raise ValueError("Missing trained Gaussian prediction head; wrong checkpoint")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path, help="Trusted Lightning .ckpt or raw PyTorch .pt")
    p.add_argument("--output", type=Path, default=Path("model.safetensors"))
    p.add_argument("--config", type=Path, help="Inference YAML matching THIS checkpoint; copied next to weights")
    p.add_argument("--allow-unsafe-pickle", action="store_true", help="Only for YOUR OWN trusted checkpoint if weights_only=True cannot load it")
    args = p.parse_args()
    import torch
    from safetensors.torch import save_file
    from safetensors import safe_open
    if args.output.exists():
        p.error(f"Refusing to overwrite existing file: {args.output}")
    if args.config:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(args.config)
        if (cfg.encoder.vggt_variant != "vggt_omega"
                or cfg.encoder.bilateral_grid is not False
                or cfg.encoder.predict_static is not True
                or cfg.encoder.fusion_mode != "static"):
            p.error("Expected final 200k inference YAML: VGGT-Omega, bilateral_grid=false, predict_static=true, fusion_mode=static")
    try:
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    except Exception as exc:
        if not args.allow_unsafe_pickle:
            raise SystemExit(
                "Secure weights_only=True checkpoint load failed. If this is your own trusted "
                "Lightning checkpoint, rerun with --allow-unsafe-pickle. Original error: " + str(exc)
            ) from exc
        print("WARNING: Unpickling a trusted checkpoint (arbitrary Python code may execute)")
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False, mmap=True)
    tensors = prepare_state_dict(checkpoint)
    if args.config and any(k.startswith("encoder.bg_param_head.") for k in tensors):
        raise SystemExit("Checkpoint contains a bilateral-grid prediction head but provided config sets bilateral_grid=false. Verify the real training configuration before exporting.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(args.output), metadata={"format": "pt", "model": "WildAnySplat"})
    with safe_open(str(args.output), framework="pt", device="cpu") as f:
        written_keys = set(f.keys())
        if written_keys != set(tensors):
            raise RuntimeError("Saved tensor keys differ from intended exported keys")
    sha = hashlib.sha256()
    with args.output.open("rb") as fp:
        for chunk in iter(lambda: fp.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    if args.config:
        if args.config.resolve() != (args.output.parent / "inference_config.yaml").resolve():
            shutil.copyfile(args.config, args.output.parent / "inference_config.yaml")
    manifest = {
        "source_checkpoint_name": args.checkpoint.name, "checkpoint_size_bytes": args.output.stat().st_size,
        "tensor_count": len(tensors), "sha256": sha.hexdigest(),
        "inference_architecture": "VGGT-Omega; predict_static=true; fusion_mode=static; bilateral_grid=false",
        "architecture_note": "Run a strict CUDA model load with this checkpoint before publication."
    }
    (args.output.parent / "checkpoint_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {len(tensors)} inference tensors to {args.output} ({args.output.stat().st_size / 1e9:.2f} GB)")
    print("SHA256:", sha.hexdigest())


if __name__ == "__main__":
    main()

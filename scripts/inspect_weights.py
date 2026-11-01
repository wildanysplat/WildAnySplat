#!/usr/bin/env python3
"""Inspect a safetensors export, optionally verifying every tensor against a Lightning checkpoint.

Run the quick check first. The optional full comparison loads the checkpoint
using memory mapping on CPU; leave sufficient free system RAM and disk cache.
Only use --allow-unsafe-pickle for your own trusted training checkpoint.
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path


def component(key: str) -> str:
    for prefix in (
        "model.", "encoder.distill_", "encoder.bg_param_head.",
        "encoder.aggregator.", "encoder.gaussian_param_head.",
        "encoder.camera_head.", "encoder.depth_head.",
        "encoder.gaussian_adapter.", "decoder.", "bg.",
    ):
        if key.startswith(prefix):
            return prefix.removesuffix(".")
    return key.split(".", 1)[0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--weights", type=Path, required=True, help="Existing or exported model.safetensors")
    ap.add_argument("--checkpoint", type=Path, default=None, help="Optional final .ckpt for complete tensor-by-tensor equality check")
    ap.add_argument("--allow-unsafe-pickle", action="store_true", help="For your own trusted Lightning checkpoint only")
    args = ap.parse_args()

    from safetensors import safe_open
    if not args.weights.is_file():
        ap.error(f"Missing file: {args.weights}")

    with safe_open(str(args.weights), framework="pt", device="cpu") as st:
        keys = set(st.keys())
        groups = Counter(component(k) for k in keys)
        print(f"File: {args.weights}")
        print(f"Size: {args.weights.stat().st_size / (1024 ** 3):.2f} GiB")
        print(f"Tensor count: {len(keys)}")
        for name, count in groups.most_common():
            print(f"  {name:35s} {count:5d}")
        errors = []
        if any(k.startswith("model.") for k in keys):
            errors.append("Unexpected 'model.' training wrapper prefix: demo requires encoder.* / decoder.* / bg.*")
        if not any(k.startswith("encoder.aggregator.") for k in keys):
            errors.append("Missing trained Omega aggregator")
        if not any(k.startswith("encoder.gaussian_param_head.") for k in keys):
            errors.append("Missing Gaussian prediction head")
        if any(k.startswith("encoder.distill_") for k in keys):
            errors.append("Contains teacher copies (encoder.distill_*); do not publish unfiltered")
        if any(k.startswith("encoder.bg_param_head.") for k in keys):
            errors.append("Contains bilateral-grid prediction head, but FINAL YAML has bilateral_grid=false")
        if errors:
            print("\nCHECK FAILED:")
            for err in errors:
                print("  -", err)
        else:
            print("\nQuick structural checks passed (does NOT establish model step or exact tensor values).")

    if args.checkpoint:
        import torch
        from prepare_checkpoint import prepare_state_dict
        print("\nComparing with final Lightning checkpoint on CPU; this may use substantial memory...")
        kw = dict(map_location="cpu", weights_only=not args.allow_unsafe_pickle, mmap=True)
        if args.allow_unsafe_pickle:
            print("WARNING: PyTorch pickle loading is unsafe for untrusted files.")
        ckpt = torch.load(args.checkpoint, **kw)
        expected = prepare_state_dict(ckpt)
        extra, missing = keys - set(expected), set(expected) - keys
        print("Unexpected keys:", len(extra), "Missing keys:", len(missing))
        if extra:
            print("  Example unexpected:", sorted(extra)[:5])
        if missing:
            print("  Example missing:", sorted(missing)[:5])
        mismatched = []
        with safe_open(str(args.weights), framework="pt", device="cpu") as st:
            shared = sorted(keys & set(expected))
            for idx, key in enumerate(shared, 1):
                tensor = st.get_tensor(key)
                if tensor.dtype != expected[key].dtype or not torch.equal(tensor, expected[key]):
                    mismatched.append(key)
                if idx % 250 == 0:
                    print(f"  Compared {idx}/{len(shared)} tensors", flush=True)
        print(f"Tensor values different: {len(mismatched)} / {len(shared)}")
        if mismatched:
            print("  Example differences:", mismatched[:8])
        if not missing and not extra and not mismatched and not errors:
            print("PASS: Existing safetensors exactly matches the filtered final checkpoint.")
        else:
            raise SystemExit("FAIL: The existing safetensors is not a clean, identical release of this final checkpoint.")
    elif errors:
        raise SystemExit("FAIL: File is not a compatible clean inference-only safetensors export.")


if __name__ == "__main__":
    main()

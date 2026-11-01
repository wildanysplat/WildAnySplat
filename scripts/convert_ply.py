#!/usr/bin/env python3
"""Convert an existing degree-4 WildAnySplat Gaussian PLY without re-inference.

Usage:
  python scripts/convert_ply.py --input outputs/scene/gaussians.ply

The original exported PLY stores *logit* opacity, *log* scale and WXYZ
quaternions already. Never run these encodings a second time when converting.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.model.ply_export import _ply_header, construct_list_of_attributes, sh_dc_to_rgb


def read_binary_vertex_ply(path: Path) -> np.memmap:
    """Memory-map standard uncompressed little-endian, vertex-only PLY.

    This intentionally rejects mesh faces/compressed PLY: this conversion is
    only for the old Gaussian-splat PLY exported by WildAnySplat.
    """
    types = {"float": "<f4", "float32": "<f4", "double": "<f8",
             "float64": "<f8", "uchar": "u1", "uint8": "u1"}
    props = []
    count = None
    in_vertex = False
    fmt = None
    with open(path, "rb") as fh:
        if fh.readline().strip() != b"ply":
            raise ValueError("Not a PLY file")
        for _ in range(500):
            row = fh.readline()
            if not row:
                raise ValueError("Incomplete PLY header")
            line = row.decode("ascii").strip()
            parts = line.split()
            if not parts or parts[0] in ("comment", "obj_info"):
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                if parts[1] != "vertex":
                    raise ValueError("PLY converter supports vertex-only Gaussian splat PLY")
                in_vertex = True
                count = int(parts[2])
            elif parts[0] == "property":
                if len(parts) != 3 or not in_vertex or parts[1] not in types:
                    raise ValueError(f"Unsupported PLY property: {line}")
                props.append((parts[2], types[parts[1]]))
            elif parts[0] == "end_header":
                offset = fh.tell()
                break
        else:
            raise ValueError("PLY header is too large")
    if fmt != "binary_little_endian" or count is None or not props:
        raise ValueError("Only vertex-only binary_little_endian PLY is supported")
    dtype = np.dtype(props)
    expected = offset + dtype.itemsize * count
    if path.stat().st_size < expected:
        raise ValueError("Truncated PLY: vertex bytes are missing")
    return np.memmap(path, dtype=dtype, offset=offset, mode="r", shape=(count,))


def convert(input_path: Path, output_path: Path, points_path: Path | None = None,
            degree: int = 3, chunk_size: int = 65536) -> dict:
    input_path, output_path = Path(input_path), Path(output_path)
    points_path = Path(points_path) if points_path else None
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output must differ from input; original PLY is preserved")
    if points_path and points_path.resolve() in (input_path.resolve(), output_path.resolve()):
        raise ValueError("RGB point cloud must have a distinct filename")
    vertices = read_binary_vertex_ply(input_path)
    fields = set(vertices.dtype.names)
    mandatory = set(construct_list_of_attributes(0))
    if not mandatory.issubset(fields):
        raise ValueError(f"Input is missing Gaussian fields: {sorted(mandatory-fields)}")
    rest_names = sorted((x for x in fields if x.startswith("f_rest_")),
                        key=lambda name: int(name.split("_")[-1]))
    if rest_names and rest_names != [f"f_rest_{i}" for i in range(len(rest_names))]:
        raise ValueError("Non-contiguous SH f_rest property indices")
    if len(rest_names) % 3:
        raise ValueError("SH rest count must be divisible by three")
    k = len(rest_names) // 3 + 1
    model_degree = math.isqrt(k) - 1
    if (model_degree + 1)**2 != k:
        raise ValueError(f"Unsupported SH coefficient count: {k}")
    if not (0 <= degree <= model_degree):
        raise ValueError(f"Requested degree {degree} exceeds file degree {model_degree}")
    in_rest_per_channel = (model_degree+1)**2 - 1
    out_rest_per_channel = (degree+1)**2 - 1
    names = construct_list_of_attributes(3*out_rest_per_channel)
    count = len(vertices)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if points_path:
        points_path.parent.mkdir(parents=True, exist_ok=True)
    from contextlib import nullcontext
    point_file = open(points_path, "wb") if points_path else nullcontext()
    with open(output_path, "wb") as splat, point_file as point:
        splat.write(_ply_header(count, [(name, "float") for name in names]))
        if point is not None:
            point.write(_ply_header(count, [("x", "float"), ("y", "float"), ("z", "float"),
                                            ("red", "uchar"), ("green", "uchar"), ("blue", "uchar")]))
        point_dtype = np.dtype([("x","<f4"),("y","<f4"),("z","<f4"),
                                ("red","u1"),("green","u1"),("blue","u1")])
        for start in range(0, count, chunk_size):
            rows = vertices[start:start+chunk_size]
            arr = np.empty((len(rows),len(names)),dtype="<f4")
            for j, name in enumerate(names):
                # SH rest coefficients are channel-major in standard 3DGS PLY:
                # [all red coefficients] [all green] [all blue]. When trimming
                # degree 4 (24 rest/channel) to degree 3 (15 rest/channel),
                # they MUST be repacked, not simply truncated to the first 45.
                if name.startswith("f_rest_"):
                    output_index = int(name.split("_")[-1])
                    channel, coeff = divmod(output_index, out_rest_per_channel)
                    source_name = f"f_rest_{channel*in_rest_per_channel+coeff}"
                else:
                    source_name = name
                arr[:,j] = rows[source_name]
            # A malformed PLY is not made healthy by discarding or rewriting
            # its NaNs invisibly. Warn clearly before importing into SuperSplat.
            if not np.isfinite(arr).all():
                raise ValueError(f"Nonfinite values in PLY rows {start}:{start+len(rows)}")
            arr.tofile(splat)
            if point is not None:
                p = np.empty(len(rows),dtype=point_dtype)
                for axis in ("x","y","z"):
                    p[axis] = rows[axis]
                dc = np.stack([rows[f"f_dc_{i}"] for i in range(3)],axis=-1)
                rgb = np.rint(255*sh_dc_to_rgb(dc)).astype(np.uint8)
                for idx,axis in enumerate(("red","green","blue")):
                    p[axis] = rgb[:,idx]
                p.tofile(point)
    return {"input_gaussians":count,"input_sh_degree":model_degree,
            "export_sh_degree":degree,"output":str(output_path),
            "rgb_pointcloud":str(points_path) if points_path else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input",required=True,type=Path)
    ap.add_argument("--output",type=Path,default=None)
    ap.add_argument("--points",type=Path,default=None)
    ap.add_argument("--degree",type=int,default=3,choices=range(5))
    args=ap.parse_args()
    out = args.output or args.input.with_name("gaussians_supersplat.ply")
    points = args.points or args.input.with_name("points_rgb.ply")
    print(convert(args.input,out,points,degree=args.degree))

if __name__=="__main__":
    main()

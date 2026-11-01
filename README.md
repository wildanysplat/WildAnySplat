# WildAnySplat

**Feed-Forward 3D Gaussian Splatting in the Wild**  
*SIGGRAPH Asia 2026*

[Project website](https://wildanysplat.github.io/) · [Paper (SIGGRAPH Asia listing)](https://sa2026.conference-schedule.org/presentation/?id=papers_2148&sess=sess109) · [Model weights (Hugging Face)](https://huggingface.co/richardoshaw/WildAnySplat) · [Source code](https://github.com/wildanysplat/WildAnySplat)

WildAnySplat reconstructs a 3D Gaussian representation from multiple images captured under challenging real-world conditions, including transient occluders and changes in appearance. This repository provides a **standalone, inference-only demo** for reconstructing a scene from an image folder or a video, with no training pipeline or dataset setup required.

The demo exports a Gaussian Splat PLY, predicted cameras, rendered input views, and an interpolated novel-view synthesis (NVS) video.

The paper link above currently opens the official SIGGRAPH Asia conference listing; it can be replaced with the ACM article link when the DOI landing page is publicly available.

## Features

- **Images or video:** reconstruct from a folder of RGB images or uniformly sampled video frames.
- **Feed-forward reconstruction:** estimate Gaussians and camera parameters without COLMAP or supplied poses.
- **View synthesis:** save input-view renders, side-by-side comparisons, and an interpolated RGB/depth video.
- **Viewer-ready exports:** open `gaussians.ply` in [SuperSplat](https://superspl.at/editor); optionally use `points_rgb.ply` as an RGB point cloud in MeshLab.
- **Opacity filtering:** exclude nearly transparent Gaussians from PLY exports without changing the model's rendered results.

## Installation

**Requirements:** Linux, Python 3.10 or newer, an NVIDIA CUDA GPU with sufficient VRAM, and a compatible CUDA-enabled PyTorch installation. Memory requirements depend on the number of input views.

```bash
git clone https://github.com/wildanysplat/WildAnySplat.git
cd WildAnySplat

conda create -n wildanysplat python=3.10 -y
conda activate wildanysplat
```

Install [PyTorch and torchvision](https://pytorch.org/get-started/locally/) for your CUDA version, then install [`torch-scatter`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for the **same** PyTorch/CUDA combination.

```bash
pip install -r requirements.txt

# Install the VGGT-Omega dependency.
git clone https://github.com/facebookresearch/vggt-omega.git ../vggt-omega
pip install -r ../vggt-omega/requirements.txt
pip install -e ../vggt-omega
```

The repository pins `gsplat==1.4.0`. Refer to the [VGGT-Omega repository](https://github.com/facebookresearch/vggt-omega) for its installation requirements and licence conditions.

## Model weights

The trained WildAnySplat model is hosted on **[Hugging Face: richardoshaw/WildAnySplat](https://huggingface.co/richardoshaw/WildAnySplat)**. This release uses two files:

```text
model.safetensors        # Trained model parameters
inference_config.yaml    # Matching inference architecture configuration
```

You can let the demo download the files automatically, or download them yourself. **No model weights are stored in the GitHub repository.**

### Option A — Download automatically (recommended)

Pass `--hf-repo richardoshaw/WildAnySplat` to the demo. It fetches `model.safetensors` and `inference_config.yaml` using `huggingface_hub` and caches them locally, so you do not need to manage checkpoint paths:

```bash
python demo.py \
  --images /path/to/scene/images \
  --hf-repo richardoshaw/WildAnySplat \
  --num-views 12 \
  --output outputs/scene
```

The first run downloads the weights; subsequent runs can reuse the Hugging Face cache.

### Option B — Download weights to `checkpoints/`

If you prefer to keep the files inside your project directory, install the Hugging Face CLI and download both files:

```bash
pip install -U huggingface_hub

hf download richardoshaw/WildAnySplat \
  model.safetensors inference_config.yaml \
  --local-dir checkpoints
```

The resulting layout is:

```text
WildAnySplat/
└── checkpoints/
    ├── model.safetensors
    └── inference_config.yaml
```

Then pass the local paths:

```bash
python demo.py \
  --images /path/to/scene/images \
  --checkpoint checkpoints/model.safetensors \
  --config checkpoints/inference_config.yaml \
  --num-views 12 \
  --output outputs/scene
```

For private or gated Hugging Face repositories, authenticate first with `hf auth login`. Public models do not require login. You can also download the files individually from the [model repository](https://huggingface.co/richardoshaw/WildAnySplat/tree/main).

## Quick start

### Reconstruct from a folder of images

```bash
python demo.py \
  --images examples/scene/images \
  --hf-repo richardoshaw/WildAnySplat \
  --num-views 12 \
  --output outputs/scene
```

Images are sorted by filename (with natural numeric ordering), and up to `--num-views` images are uniformly selected. Use **at least two views of the same scene**, preferably with good overlap.

### Reconstruct from a video

```bash
python demo.py \
  --video examples/scene.mp4 \
  --hf-repo richardoshaw/WildAnySplat \
  --num-views 12 \
  --output outputs/video_scene
```

Video frames are sampled throughout the clip. For best results, use footage with camera motion around a mostly static scene.

### Use locally downloaded weights

```bash
python demo.py \
  --images /path/to/scene/images \
  --checkpoint checkpoints/model.safetensors \
  --config checkpoints/inference_config.yaml \
  --num-views 12 \
  --output outputs/local_scene
```

Using `--hf-repo` automatically downloads both model files into the Hugging Face cache. A CUDA GPU is required for inference.

## Outputs

By default, each run produces:

```text
outputs/scene/
├── gaussians.ply                 # Gaussian Splat model
├── points_rgb.ply                # RGB-vertex point cloud
├── cameras.json                 # Camera metadata and source filenames
├── cameras.npz                  # Predicted camera matrices and intrinsics
├── rendered_input_views/        # Renders at predicted input cameras
├── input_vs_render/             # Input / reconstruction comparisons
├── rgb.mp4                      # Novel-view synthesis trajectory
├── depth.mp4                    # Colourised depth trajectory
├── nvs_cameras.npz              # Interpolated trajectory cameras
└── run_summary.json             # Run settings and export statistics
```

For video inputs, the output also includes `sampled_video_frames/`. Additional optional outputs depend on the command-line flags and model configuration.

### Gaussian export and opacity filtering

`gaussians.ply` is a standard Gaussian Splat PLY, with spherical-harmonic coefficients, log-scales, logit opacities, and quaternion rotations. It is the file to use with **SuperSplat** and other Gaussian Splat viewers. The model renders internally with SH degree 4; the PLY defaults to **SH degree 3** for broader viewer compatibility.

By default, the exporter removes invalid Gaussians and those with opacity below **1/255 (approximately 0.00392)**. To prune more aggressively:

```bash
python demo.py \
  --images examples/scene/images \
  --hf-repo richardoshaw/WildAnySplat \
  --min-opacity 0.01 \
  --output outputs/scene_pruned
```

This filtering affects **only exported PLY files**, not the Gaussian set used for the PNG and video renders. `points_rgb.ply` is a conventional RGB point cloud rather than a Gaussian Splat model.

## Useful options

| Option | Description | Default |
| --- | --- | --- |
| `--num-views N` | Maximum number of input views | `12` |
| `--min-opacity T` | Minimum opacity for exported Gaussians | `1/255` |
| `--ply-sh-degree N` | Spherical-harmonic degree in the PLY (0–4) | `3` |
| `--dc-only` | Export only SH DC colours to the PLY | Off |
| `--nvs-between N` | Intermediate cameras per adjacent input-view pair | `10` |
| `--nvs-fps N` | Output NVS video frame rate | `24` |
| `--render-size N` | Square output render size in pixels | `448` |
| `--no-render-input-views` | Skip input-view PNGs | Off |
| `--no-nvs` | Skip novel-view videos | Off |
| `--no-nvs-depth` | Skip the depth video | Off |
| `--save-nvs-frames` | Save individual NVS frames | Off |
| `--no-rgb-pointcloud` | Skip `points_rgb.ply` | Off |

For the complete list, run `python demo.py --help`.

## Inference notes

- Input images are resized and centre-cropped to **448 × 448** before inference.
- The NVS camera path interpolates between neighbouring predicted cameras, using rotation SLERP and linear interpolation for translation and intrinsics. **Input ordering matters.**
- The inference model uses a **VGGT-Omega backbone**, transient/static prediction, and static-aware Gaussian fusion. The released checkpoint's bilateral-grid prediction head is disabled.
- Camera estimates and reconstructions are produced directly by the model; external camera calibration is not required.

## Citation

If you use WildAnySplat in your research, please cite:

```bibtex
@inproceedings{shaw2026wildanysplat,
  author = {Shaw, Richard and Moreau, Arthur and Papaioannou, Athanasios and Tanay, Thomas and Zhang, Zhensong and P{\'{e}}rez-Pellitero, Eduardo},
  title = {WildAnySplat: Feed-Forward 3D Gaussian Splatting in the Wild},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year = {2026},
  doi = {10.1145/3829340.3842323}
}
```

## Acknowledgements and licensing

WildAnySplat builds on research and code from [AnySplat](https://github.com/OpenRobotLab/AnySplat) and [VGGT-Omega](https://github.com/facebookresearch/vggt-omega). See [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the `licenses/` directory for third-party notices and applicable terms. VGGT-Omega components are subject to noncommercial research licensing; check the relevant terms before use or redistribution.

<div align="center">

<h1>WildAnySplat: Feed-Forward 3D Gaussian Splatting in the Wild</h1>

<p>
  <a href="https://scholar.google.com/citations?user=9qqtzK4AAAAJ&hl=en">Richard Shaw</a> &nbsp;·&nbsp;
  <a href="https://scholar.google.com/citations?user=fMmoHX0AAAAJ&hl=en">Arthur Moreau</a> &nbsp;·&nbsp;
  <a href="https://apapaion.github.io/">Athanasios Papaioannou</a><br>
  <a href="https://thomas-tanay.github.io/">Thomas Tanay</a> &nbsp;·&nbsp;
  <a href="https://zhangzhensong.github.io/">Zhang Zhensong</a> &nbsp;·&nbsp;
  <a href="https://perezpellitero.github.io/">Eduardo Pérez-Pellitero</a>
</p>

<p>
  <strong>Huawei Noah's Ark Lab · London, UK</strong><br>
  <strong>SIGGRAPH Asia 2026 · Conference Papers</strong>
</p>

<p>
  <a href="https://wildanysplat.github.io/wildanysplat.pdf"><img alt="Paper PDF" src="https://img.shields.io/badge/Paper-PDF-D35400?style=for-the-badge&logo=adobeacrobatreader&logoColor=white"></a>
  <a href="https://wildanysplat.github.io/"><img alt="Project Website" src="https://img.shields.io/badge/Project-Website-2C3E50?style=for-the-badge&logo=googlechrome&logoColor=white"></a>
  <a href="https://wildanysplat.github.io/#comparisons"><img alt="Video Results" src="https://img.shields.io/badge/Results-Videos-2980B9?style=for-the-badge&logo=youtube&logoColor=white"></a>
  <a href="https://huggingface.co/richardoshaw/WildAnySplat"><img alt="Model Weights on Hugging Face" src="https://img.shields.io/badge/Model-Weights-FFCA28?style=for-the-badge&logo=huggingface&logoColor=black"></a>
</p>

<p><strong>Robust 3D reconstruction from in-the-wild images — in a single feed-forward pass.</strong></p>

<p><em>Official inference-only implementation and pretrained weights.</em></p>

</div>

---

WildAnySplat reconstructs **3D Gaussian Splats** from uncalibrated multi-view images captured in the wild, where transient objects, illumination differences, and camera-dependent appearance can make reconstruction challenging. Its teacher–student training framework learns to produce more consistent scene geometry and colours despite these variations.

This repository contains a **standalone inference demo**. Give it a folder of images or a video; it predicts camera parameters and Gaussians, renders the input views, and produces a smooth novel-view synthesis (NVS) video. **No training code, COLMAP, input camera poses, or dataset setup is required.**

<p align="center">
  <a href="https://wildanysplat.github.io/">
    <img src="https://wildanysplat.github.io/WildAnySplat_method.png" alt="WildAnySplat method overview: teacher–student training, transient-aware fusion and appearance modelling" width="900">
  </a>
</p>
<p align="center"><sub><b>Method overview.</b> WildAnySplat learns transient-robust 3D Gaussians using a frozen clean-image teacher and an in-the-wild student. The diagram describes the paper framework; the released inference checkpoint uses transient-aware fusion with its bilateral-grid head disabled.</sub></p>

## ✨ Highlights

- **📷 Images or video** — reconstruct a scene from an image directory or uniformly sampled video frames.
- **⚡ Feed-forward 3DGS** — infer Gaussians and camera parameters directly from uncalibrated views.
- **🎬 Rendered results** — generate input-view reconstructions, visual comparisons, and interpolated RGB/depth NVS videos.
- **🧊 Gaussian exports** — open `gaussians.ply` in [SuperSplat](https://superspl.at/editor); export an RGB point cloud for MeshLab.
- **🧹 Opacity pruning** — omit nearly transparent Gaussians from PLY exports without changing neural rendering.

## 🚀 Quick start

### 1. Install

**Requirements:** Linux, Python 3.10+, an NVIDIA GPU with CUDA and sufficient VRAM, and a compatible CUDA-enabled PyTorch installation. Memory usage depends on the number of input views.

```bash
git clone https://github.com/wildanysplat/WildAnySplat.git
cd WildAnySplat

conda create -n wildanysplat python=3.10 -y
conda activate wildanysplat
```

Install [PyTorch and torchvision](https://pytorch.org/get-started/locally/) for your CUDA version and [`torch-scatter`](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) for the **same** PyTorch/CUDA combination. Then:

```bash
pip install -r requirements.txt

# VGGT-Omega dependency
git clone https://github.com/facebookresearch/vggt-omega.git ../vggt-omega
pip install -r ../vggt-omega/requirements.txt
pip install -e ../vggt-omega
```

The inference release uses `gsplat==1.4.0`. See the [VGGT-Omega repository](https://github.com/facebookresearch/vggt-omega) for its installation and licensing requirements.

### 2. Run inference

**From a folder of images:**

```bash
python demo.py \
  --images /path/to/scene/images \
  --hf-repo richardoshaw/WildAnySplat \
  --num-views 12 \
  --output outputs/scene
```

**From a video:**

```bash
python demo.py \
  --video /path/to/scene.mp4 \
  --hf-repo richardoshaw/WildAnySplat \
  --num-views 12 \
  --output outputs/video_scene
```

The first run **downloads the pretrained model and inference configuration automatically** from [Hugging Face](https://huggingface.co/richardoshaw/WildAnySplat); subsequent runs reuse the local cache.

> **Input tips:** Use at least two views of the **same scene**, preferably with good overlap and a meaningful sequence of camera positions. Image names are sorted naturally before sampling; video frames are sampled throughout the clip. Input images are resized and centre-cropped to 448 × 448.

### 3. Explore the results

Each run writes outputs similar to:

```text
outputs/scene/
├── gaussians.ply              # Gaussian Splats (SuperSplat-compatible)
├── points_rgb.ply             # Conventional coloured point cloud (MeshLab)
├── cameras.json               # Predicted camera information and source filenames
├── cameras.npz                # Camera matrices and intrinsics
├── rendered_input_views/     # PNG renders at the predicted input cameras
├── input_vs_render/          # Side-by-side input and rendered PNGs
├── rgb.mp4                   # Interpolated novel-view synthesis video
├── depth.mp4                 # Colourised depth video
├── nvs_cameras.npz           # Interpolated NVS camera parameters
└── run_summary.json          # Run settings and export statistics
```

For video input, the demo also saves `sampled_video_frames/`. Optional outputs depend on the selected flags.

## 🤗 Model weights

**Pretrained checkpoint:** [richardoshaw/WildAnySplat](https://huggingface.co/richardoshaw/WildAnySplat)

The model repository contains the two files required for inference:

| File | Purpose |
| :--- | :--- |
| `model.safetensors` | Trained model parameters |
| `inference_config.yaml` | Matching inference architecture settings |

**Automatic download** (recommended): use `--hf-repo richardoshaw/WildAnySplat` as in the quick-start commands above.

**Manual download:**

```bash
pip install -U huggingface_hub

hf download richardoshaw/WildAnySplat \
  model.safetensors inference_config.yaml \
  --local-dir checkpoints
```

Then run with the local checkpoint:

```bash
python demo.py \
  --images /path/to/scene/images \
  --checkpoint checkpoints/model.safetensors \
  --config checkpoints/inference_config.yaml \
  --num-views 12 \
  --output outputs/local_scene
```

Public models do not require authentication. For private or gated access, run `hf auth login` first. **The model weights are not stored in this GitHub repository.**

## 🛠️ Inference options

| Argument | Description | Default |
| :--- | :--- | :--- |
| `--num-views N` | Maximum number of input frames/views | `12` |
| `--min-opacity T` | Minimum opacity for exported Gaussians | `1/255` |
| `--ply-sh-degree N` | SH degree stored in the PLY (`0–4`) | `3` |
| `--dc-only` | Export only DC colour coefficients to PLY | Off |
| `--nvs-between N` | Interpolated viewpoints between adjacent cameras | `10` |
| `--nvs-fps N` | NVS output frame rate | `24` |
| `--render-size N` | Square rendering resolution in pixels | `448` |
| `--no-render-input-views` | Skip per-input-view PNGs | Off |
| `--no-nvs` | Skip NVS videos | Off |
| `--no-nvs-depth` | Skip the depth MP4 | Off |
| `--save-nvs-frames` | Save individual NVS PNGs | Off |
| `--no-rgb-pointcloud` | Skip the RGB point-cloud export | Off |

For all options, run `python demo.py --help`.

### Gaussian export

- **`gaussians.ply`** is the standard Gaussian Splat export, compatible with SuperSplat. It includes position, SH colour coefficients, log-scales, logit opacities and quaternion rotations. The internal renderer uses SH degree 4; the PLY defaults to **degree 3** for broader viewer compatibility.
- **`points_rgb.ply`** is a conventional point cloud with RGB vertex colours for MeshLab. It is **not** a Gaussian Splat file.
- The exporter drops invalid and low-opacity Gaussians by default (**opacity < 1/255 ≈ 0.00392**). To prune more aggressively, add `--min-opacity 0.01`. This filter affects exported PLY files **only**; the RGB/depth renders use the full predicted Gaussian set.

### Rendering and model notes

The NVS video follows a trajectory interpolated between successive **predicted input camera poses**: SLERP for rotation, linear interpolation for camera translation and intrinsics. Therefore, input order matters. Frames are rendered sequentially to control memory use.

The released model uses a **VGGT-Omega backbone**, predicts transient/static information, and performs static-aware Gaussian fusion. The supplied checkpoint has the **bilateral-grid prediction head disabled**, although bilateral-grid appearance modelling is part of the method described in the paper.

## 📖 Citation

If WildAnySplat is useful to your work, please cite our SIGGRAPH Asia 2026 paper:

```bibtex
@inproceedings{shaw2026wildanysplat,
  author    = {Shaw, Richard and Moreau, Arthur and Papaioannou, Athanasios and Tanay, Thomas and Zhang, Zhensong and P{\'{e}}rez-Pellitero, Eduardo},
  title     = {WildAnySplat: Feed-Forward 3D Gaussian Splatting in the Wild},
  booktitle = {SIGGRAPH Asia 2026 Conference Papers},
  year      = {2026},
  doi       = {10.1145/3829340.3842323}
}
```

**[Read the paper (PDF)](https://wildanysplat.github.io/wildanysplat.pdf)** · **[Project website](https://wildanysplat.github.io/)**

## 🙏 Acknowledgements and licensing

This work builds on [AnySplat](https://github.com/OpenRobotLab/AnySplat) and [VGGT-Omega](https://github.com/facebookresearch/vggt-omega). Please see [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) and the `licenses/` directory for applicable third-party terms. VGGT-Omega components have noncommercial research licensing restrictions; review their terms before use or redistribution.

---

<p align="center">
  <a href="https://wildanysplat.github.io/">Project website</a> &nbsp;·&nbsp;
  <a href="https://wildanysplat.github.io/wildanysplat.pdf">Paper PDF</a> &nbsp;·&nbsp;
  <a href="https://huggingface.co/richardoshaw/WildAnySplat">Model weights</a>
</p>

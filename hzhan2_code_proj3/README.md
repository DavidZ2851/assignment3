# 16-825 Assignment 3 code (hzhan2)

All results on the webpage come from the code in this folder.

```
hzhan2_code_proj3/
    main.py            # runs every part of the assignment
    README.md
    requirements.txt
    nerf/              # Part A: neural volume rendering (run from nerf/)
        volume_rendering_main.py, renderer.py, sampler.py, implicit.py,
        ray_utils.py, losses.py, dataset.py, render_functions.py, data_utils.py
        configs/*.yaml
    GS/                # Part B: 3D Gaussian splatting (run from GS/)
        model.py, render.py, train.py, train_harder_scene.py,
        data_utils.py, data_utils_harder_scene.py, unit_test_gaussians.py
```

## Setup

1. Use the Assignment 1 environment (`torch` + `pytorch3d`), then `pip install -r requirements.txt`.
2. Data is not included in the zip. Follow the data instructions in the
   [assignment README](https://github.com/learning3d/assignment3#data):
   - `nerf/data/`: `lego.pth`, `materials.pth` and the `box_*.npy` files.
   - `GS/data/`: `chair.ply`, `truck/` and `materials/`.
3. A CUDA GPU is required.

## Running

```
python main.py --list        # list all parts
python main.py q3 q4_2       # run selected parts
python main.py all           # run everything (several hours)
```

| Part | What it runs | Outputs |
|---|---|---|
| `q1` | `volume_rendering_main.py --config-name=box` | `nerf/images/grid.png`, `rays.png`, `sample_points.png`, `depth.png`, `part_1.gif` |
| `q2` | `--config-name=train_box` | `nerf/images/part_2_*.png`, `part_2.gif`; box center and side lengths printed |
| `q3` | `--config-name=nerf_lego` | `nerf/images/part_3.gif` |
| `q4_1` | `--config-name=nerf_materials`, `nerf_materials_highres` | `nerf/images/part_3.gif` for each run |
| `q4_2` | `--config-name=nerf_lego_hierarchical`; `nerf_lego` with 192 samples | `nerf/images/part_3.gif` for each run |
| `q4_3` | `--config-name=volsdf_surface` with beta 0.05, 0.2, 0.01 | `nerf/images/part_4_3.gif`, `part_4_3_geometry.gif` |
| `q5_test` | `unit_test_gaussians.py` | test results printed |
| `q5` | `render.py --no_sh` | `GS/output/part_5_render.gif` |
| `q6` | `train.py` on the truck | `GS/output/part_6_training_progress.gif`, `part_6_training_final_renders.gif`; PSNR/SSIM printed |
| `q7_1` | `render.py` (full spherical harmonics) | `GS/output/q7_1_render.gif` |
| `q7_2` | `train_harder_scene.py --mode baseline` and `--mode improved` | `GS/output/q7_2_{baseline,improved}/` GIFs and `metrics.json` |

Every NeRF run writes `part_3.gif` / `part_4_3*.gif` under the same name, so the
webpage copies were renamed after each run (for example `part_4_2_lego_coarse_fine.gif`).
Each run gets its own checkpoint folder (set in `main.py`) and resumes from it automatically; delete that folder to retrain from scratch.

## Implementation notes

- **Q1-Q3** (`ray_utils.py`, `sampler.py`, `renderer.py`, `implicit.py`, `volume_rendering_main.py`):
  ray/pixel generation, stratified sampling, volume rendering with depth, random pixel
  sampling, MSE loss and the NeRF MLP (6 layers, 128 wide, skip connection at layer 3).
- **Q4.1**: the colour head takes a harmonic embedding of the view direction; density is position only.
- **Q4.2**: `renderer.type: volume_hierarchical` adds a fine network with 128 inverse-CDF
  samples drawn from the coarse weights (`nerf_lego_hierarchical.yaml`).
- **Q4.3**: `sdf_to_density` (VolSDF Laplace CDF), `eikonal_loss` and `NeuralSurface`.
- **Q5 / Q7.1** (`model.py`, `data_utils.py`, `render.py`): 3D-to-2D Gaussian projection,
  depth sorting, alpha/transmittance, splatting of colour, depth and mask, and degree-3
  spherical-harmonic colours. `render.py --no_sh` gives the DC-only Q5 render.
- **Q6** (`train.py`): Adam with lr means 0.0002, scales 0.001, opacities 0.01, colours 0.01; L1 loss; 1000 iterations.
- **Q7.2** (`train_harder_scene.py`): `--mode improved` adds a wide random initialization,
  anisotropic Gaussians, learning-rate decay on the means, an L1 + D-SSIM loss, simplified
  adaptive density control (clone/split/prune and opacity reset) and 2500 iterations.
  `--mode baseline` is the Q6 setup.

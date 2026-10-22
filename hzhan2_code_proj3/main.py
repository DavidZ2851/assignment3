"""Run every part of 16-825 Assignment 3.

Usage:
    python main.py --list            # show all available parts
    python main.py q1 q3             # run selected parts
    python main.py all               # run everything (slow)

Each part runs the same command that produced the results on the webpage.
NeRF parts run inside ./nerf and write to ./nerf/images; Gaussian splatting
parts run inside ./GS and write to ./GS/output. See README.md for data setup.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable

# name -> (working dir, list of commands, description)
PARTS = {
    "q1": ("nerf", [
        [PY, "volume_rendering_main.py", "--config-name=box"],
    ], "1.3-1.5 ray/point sampling and volume rendering of the box"),
    "q2": ("nerf", [
        [PY, "volume_rendering_main.py", "--config-name=train_box"],
    ], "2.1-2.3 random ray sampling and optimizing the box"),
    "q3": ("nerf", [
        [PY, "volume_rendering_main.py", "--config-name=nerf_lego",
         "training.checkpoint_path=./checkpoints_lego"],
    ], "3 NeRF on lego (128x128)"),
    "q4_1": ("nerf", [
        [PY, "volume_rendering_main.py", "--config-name=nerf_materials",
         "training.checkpoint_path=./checkpoints_materials"],
        [PY, "volume_rendering_main.py", "--config-name=nerf_materials_highres",
         "training.checkpoint_path=./checkpoints_materials_highres",
         "renderer.chunk_size=4096"],  # fits an 8 GB GPU
    ], "4.1 view-dependent NeRF on materials (128x128 and 400x400)"),
    "q4_2": ("nerf", [
        [PY, "volume_rendering_main.py", "--config-name=nerf_lego_hierarchical"],
        [PY, "volume_rendering_main.py", "--config-name=nerf_lego",
         "sampler.n_pts_per_ray=192", "training.checkpoint_path=./checkpoints_lego_192",
         "seed=2"],  # seed 1 collapsed to zero density
    ], "4.2 coarse/fine sampling vs. single network with 192 samples"),
    "q4_3": ("nerf", [
        [PY, "volume_rendering_main.py", "--config-name=volsdf_surface",
         "renderer.beta=0.05", "training.checkpoint_path=./volsdf_checkpoint"],
        [PY, "volume_rendering_main.py", "--config-name=volsdf_surface",
         "renderer.beta=0.2", "training.checkpoint_path=./volsdf_checkpoint_b0.2"],
        [PY, "volume_rendering_main.py", "--config-name=volsdf_surface",
         "renderer.beta=0.01", "training.checkpoint_path=./volsdf_checkpoint_b0.01"],
    ], "4.3 VolSDF with alpha=10 and beta in {0.05, 0.2, 0.01}"),
    "q5_test": ("GS", [
        [PY, "unit_test_gaussians.py"],
    ], "5.2 unit tests for the Gaussian projection/evaluation"),
    "q5": ("GS", [
        [PY, "render.py", "--no_sh", "--out_path", "./output"],
    ], "5.5 render pre-trained chair Gaussians, DC colour only"),
    "q6": ("GS", [
        [PY, "train.py", "--out_path", "./output", "--data_path", "./data/truck",
         "--gaussians_per_splat", "1024"],
    ], "6 train isotropic Gaussians on the truck"),
    "q7_1": ("GS", [
        [PY, "render.py", "--out_path", "./output"],
    ], "7.1 render chair Gaussians with full spherical harmonics"),
    "q7_2": ("GS", [
        [PY, "train_harder_scene.py", "--mode", "baseline",
         "--out_path", "./output/q7_2_baseline", "--data_path", "./data/materials",
         "--gaussians_per_splat", "1000"],
        [PY, "train_harder_scene.py", "--mode", "improved",
         "--out_path", "./output/q7_2_improved", "--data_path", "./data/materials",
         "--gaussians_per_splat", "1000", "--num_itrs", "2500", "--viz_freq", "50",
         "--densify_until", "1800", "--max_gaussians", "30000"],
    ], "7.2 materials scene from random init: baseline and improved"),
}


def run_part(name):
    workdir, commands, desc = PARTS[name]
    cwd = os.path.join(ROOT, workdir)
    os.makedirs(os.path.join(cwd, "images" if workdir == "nerf" else "output"), exist_ok=True)
    print(f"\n===== {name}: {desc} =====", flush=True)
    for cmd in commands:
        print(f"[{workdir}] $ {' '.join(cmd[1:])}", flush=True)
        subprocess.run(cmd, cwd=cwd, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("parts", nargs="*", help="parts to run, or 'all'")
    parser.add_argument("--list", action="store_true", help="list available parts")
    args = parser.parse_args()

    if args.list or not args.parts:
        for name, (workdir, _, desc) in PARTS.items():
            print(f"  {name:8s} [{workdir}] {desc}")
        return

    names = list(PARTS) if args.parts == ["all"] else args.parts
    unknown = [n for n in names if n not in PARTS]
    if unknown:
        parser.error(f"unknown part(s): {', '.join(unknown)}; use --list")
    for name in names:
        run_part(name)


if __name__ == "__main__":
    main()

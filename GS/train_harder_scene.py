import os
import math
import json
import time
import torch
import torch.utils.checkpoint
import imageio
import argparse
import numpy as np

from PIL import Image
from tqdm import tqdm
from model import Scene, Gaussians
from torch.utils.data import DataLoader
from data_utils_harder_scene import get_nerf_datasets, trivial_collate

from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.ops.knn import knn_points
from pytorch3d.transforms import quaternion_to_matrix
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Baseline: the exact Q6 setup (isotropic, Q6 learning rates, L1 loss).
BASELINE_LRS = {"opacities": 0.01, "scales": 0.001, "colours": 0.01, "means": 0.0002}

# Improved: anisotropic gaussians and larger initial learning rates. The means
# learning rate decays exponentially to IMPROVED_MEANS_LR_FINAL over training.
IMPROVED_LRS = {
    "opacities": 0.05, "scales": 0.005, "colours": 0.01,
    "means": 0.002, "quats": 0.001,
}
IMPROVED_MEANS_LR_FINAL = 0.00002


def make_trainable(gaussians):

    ### YOUR CODE HERE ###
    # HINT: You can access and modify parameters from gaussians
    gaussians.means.requires_grad_(True)
    gaussians.pre_act_scales.requires_grad_(True)
    gaussians.colours.requires_grad_(True)
    gaussians.pre_act_opacities.requires_grad_(True)
    if not gaussians.is_isotropic:
        gaussians.pre_act_quats.requires_grad_(True)

def setup_optimizer(gaussians, lrs=BASELINE_LRS):

    gaussians.check_if_trainable()

    ### YOUR CODE HERE ###
    # HINT: Modify the learning rates to reasonable values. We have intentionally
    # set very high learning rates for all parameters.
    # HINT: Consider reducing the learning rates for parameters that seem to vary too
    # fast with the default settings.
    # HINT: Consider setting different learning rates for different sets of parameters.
    parameters = [
        {'params': [gaussians.pre_act_opacities], 'lr': lrs["opacities"], "name": "opacities"},
        {'params': [gaussians.pre_act_scales],    'lr': lrs["scales"],    "name": "scales"},
        {'params': [gaussians.colours],           'lr': lrs["colours"],   "name": "colours"},
        {'params': [gaussians.means],             'lr': lrs["means"],     "name": "means"},
    ]
    if not gaussians.is_isotropic:
        parameters.append(
            {'params': [gaussians.pre_act_quats], 'lr': lrs["quats"], "name": "quats"}
        )
    optimizer = torch.optim.Adam(parameters, lr=0.0, eps=1e-15)

    return optimizer

def ndc_to_screen_camera(camera, img_size = (128, 128)):

    min_size = min(img_size[0], img_size[1])

    screen_focal = camera.focal_length * min_size / 2.0
    screen_principal = torch.tensor([[img_size[0]/2, img_size[1]/2]]).to(torch.float32)

    return PerspectiveCameras(
        R=camera.R, T=camera.T, in_ndc=False,
        focal_length=screen_focal, principal_point=screen_principal,
        image_size=(img_size,),
    )

def _evaluate_gaussian_2D_fast(points_2D, means_2D, cov_2D_inverse):
    # Same value as Gaussians.evaluate_gaussian_2D, written as elementwise ops.
    # The einsum there becomes a batched matmul with tiny matrices, which took
    # ~90% of the training step time.
    d = points_2D - means_2D  # (N, HW, 2)
    dx, dy = d[..., 0], d[..., 1]
    a = cov_2D_inverse[:, 0, 0][:, None]
    b = (cov_2D_inverse[:, 0, 1] + cov_2D_inverse[:, 1, 0])[:, None]
    c = cov_2D_inverse[:, 1, 1][:, None]
    return -0.5 * (a * dx * dx + b * dx * dy + c * dy * dy)  # (N, HW)

Gaussians.evaluate_gaussian_2D = staticmethod(_evaluate_gaussian_2D_fast)

def render_img(scene, camera, args, img_size=(128, 128)):
    """Same computation as Scene.render (black background), but each mini-batch of
    gaussians is splatted under gradient checkpointing. Scene.render keeps the
    (N, H, W) intermediates of every mini-batch alive for backprop, which needs
    ~9 GB for 10k gaussians at 128x128. With checkpointing only one mini-batch is
    materialised at a time and it is recomputed during the backward pass."""
    g = scene.gaussians
    z_vals = scene.compute_depth_values(camera)
    idxs = scene.get_idxs_to_filter_and_sort(z_vals)
    z_vals = z_vals[idxs]
    means_3D = g.means[idxs]
    colours = g.colours[idxs]
    quats, scales, opacities = g.apply_activations(
        g.pre_act_quats[idxs], g.pre_act_scales[idxs], g.pre_act_opacities[idxs]
    )

    per_splat = args.gaussians_per_splat if args.gaussians_per_splat > 0 else len(means_3D)
    W, H = img_size
    D = means_3D.device
    transmittance = torch.ones((1, H, W), dtype=torch.float32, device=D)
    image = torch.zeros((H, W, 3), dtype=torch.float32, device=D)
    mask = torch.zeros((H, W, 1), dtype=torch.float32, device=D)

    def splat_chunk(m, z, q, s, c, o, t):
        img_, _, mask_, t_ = scene.splat(camera, m, z, q, s, c, o, img_size, t)
        return img_, mask_, t_

    for b in range(0, len(means_3D), per_splat):
        sl = slice(b, b + per_splat)
        chunk = (means_3D[sl], z_vals[sl], quats[sl], scales[sl], colours[sl],
                 opacities[sl], transmittance)
        if torch.is_grad_enabled():
            img_, mask_, _ = torch.utils.checkpoint.checkpoint(
                splat_chunk, *chunk, use_reentrant=False
            )
        else:
            img_, mask_, _ = splat_chunk(*chunk)
        image = image + img_
        mask = mask + mask_
        # Transmittance after this mini-batch. (Scene.splat returns the
        # transmittance in front of its last gaussian, which skips that gaussian's
        # alpha and makes Scene.render depend on per_splat.) Since the weights of a
        # mini-batch sum to T_start * (1 - prod(1 - alpha)), T_end = T_start - mask_.
        transmittance = (transmittance - mask_[..., 0][None]).clamp_min(0.0)

    # Matches Scene.render with bg_colour = (0, 0, 0)
    return mask * image

def visualize_renders_chunked(scene, gt_viz_img, cameras, args):
    # Same as data_utils.visualize_renders, but respects --gaussians_per_splat so
    # that larger (densified) scenes fit in GPU memory.
    imgs = []
    with torch.no_grad():
        for cam in cameras:
            img = torch.clamp(render_img(scene, cam, args), 0.0, 1.0) * 255.0
            img = img.cpu().numpy().astype(np.uint8)
            imgs.append(np.array(Image.fromarray(img).resize((256, 256))))
    return np.concatenate((np.concatenate(imgs, axis=1), gt_viz_img), axis=0)

# ---------------------------------------------------------------------------
# Components of the improved method
# ---------------------------------------------------------------------------

def gaussian_window(size=11, sigma=1.5, channels=3, device="cuda"):
    coords = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-coords ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    w = (g[:, None] * g[None, :])[None, None]
    return w.repeat(channels, 1, 1, 1)

def ssim_torch(pred, gt, window):
    """Differentiable SSIM between two (H, W, 3) images in [0, 1]."""
    x = pred.permute(2, 0, 1)[None]
    y = gt.permute(2, 0, 1)[None]
    C = x.shape[1]
    pad = window.shape[-1] // 2
    conv = lambda t: torch.nn.functional.conv2d(t, window, padding=pad, groups=C)
    mu_x, mu_y = conv(x), conv(y)
    sxx = conv(x * x) - mu_x ** 2
    syy = conv(y * y) - mu_y ** 2
    sxy = conv(x * y) - mu_x * mu_y
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sxy + C2)) / \
               ((mu_x ** 2 + mu_y ** 2 + C1) * (sxx + syy + C2))
    return ssim_map.mean()

def init_improved_gaussians(gaussians, init_radius, init_opacity):
    """Spread the random means uniformly inside a ball that covers the whole scene
    (instead of a small gaussian blob at the origin), start with a low opacity so
    gaussians hidden behind others still receive gradients, and set the scales from
    nearest neighbour distances."""
    N = len(gaussians)
    D = gaussians.means.device
    dirs = torch.nn.functional.normalize(torch.randn((N, 3), device=D), dim=-1)
    radii = init_radius * torch.rand((N, 1), device=D) ** (1.0 / 3.0)
    gaussians.means = dirs * radii
    gaussians.pre_act_opacities = torch.logit(torch.full((N,), init_opacity, device=D))
    gaussians.colours = torch.rand((N, 3), device=D)
    dists, _, _ = knn_points(gaussians.means[None], gaussians.means[None], K=4)
    log_s = torch.log(torch.sqrt(dists[0, :, 1:]).mean(dim=1).clamp_min(1e-4))[:, None]
    if not gaussians.is_isotropic:
        log_s = log_s.repeat(1, 3)
    gaussians.pre_act_scales = log_s

def densify_and_prune(gaussians, grad_accum, grad_count, args):
    """Simplified adaptive density control (Kerbl et al. 2023), using the
    gradient of the 3D means as the densification signal:
    - clone small gaussians with large average positional gradients
    - split large gaussians with large gradients into two smaller children
    - prune gaussians that are nearly transparent or have grown too large
    Returns the number of gaussians after the update."""
    with torch.no_grad():
        means = gaussians.means.detach()
        scales_pre = gaussians.pre_act_scales.detach()
        quats_pre = gaussians.pre_act_quats.detach()
        opac_pre = gaussians.pre_act_opacities.detach()
        cols = gaussians.colours.detach()

        avg_grad = grad_accum / grad_count.clamp_min(1)
        scales = torch.exp(scales_pre).max(dim=1).values

        # Only densify the gaussians with the largest gradients if over budget
        n_now = len(means)
        budget = max(0, args.max_gaussians - n_now)
        high_grad = avg_grad >= args.densify_grad_thresh
        if high_grad.sum() > budget:
            high_grad[:] = False
            if budget > 0:
                high_grad[torch.topk(avg_grad, budget).indices] = True

        clone_mask = high_grad & (scales <= args.densify_scale_thresh)
        split_mask = high_grad & (scales > args.densify_scale_thresh)

        new_means = [means, means[clone_mask]]
        new_scales = [scales_pre, scales_pre[clone_mask]]
        new_quats = [quats_pre, quats_pre[clone_mask]]
        new_opac = [opac_pre, opac_pre[clone_mask]]
        new_cols = [cols, cols[clone_mask]]

        if split_mask.any():
            s = torch.exp(scales_pre[split_mask])
            if s.shape[1] == 1:
                s = s.repeat(1, 3)
            R = quaternion_to_matrix(torch.nn.functional.normalize(quats_pre[split_mask]))
            for _ in range(2):
                offs = torch.randn_like(s) * s
                new_means.append(means[split_mask] + torch.einsum("nij,nj->ni", R, offs))
                new_scales.append(scales_pre[split_mask] - math.log(1.6))
                new_quats.append(quats_pre[split_mask])
                new_opac.append(opac_pre[split_mask])
                new_cols.append(cols[split_mask])

        means = torch.cat(new_means)
        scales_pre = torch.cat(new_scales)
        quats_pre = torch.cat(new_quats)
        opac_pre = torch.cat(new_opac)
        cols = torch.cat(new_cols)

        # Remove split parents, transparent gaussians and huge gaussians
        keep = torch.ones(len(means), dtype=torch.bool, device=means.device)
        keep[:n_now][split_mask] = False
        keep &= torch.sigmoid(opac_pre) > args.prune_opacity
        keep &= torch.exp(scales_pre).max(dim=1).values < args.prune_scale

        gaussians.means = means[keep].contiguous()
        gaussians.pre_act_scales = scales_pre[keep].contiguous()
        gaussians.pre_act_quats = quats_pre[keep].contiguous()
        gaussians.pre_act_opacities = opac_pre[keep].contiguous()
        gaussians.colours = cols[keep].contiguous()

    return len(gaussians.means)

def reset_opacity(gaussians, max_opacity=0.01):
    with torch.no_grad():
        clipped = torch.clamp(torch.sigmoid(gaussians.pre_act_opacities), max=max_opacity)
        gaussians.pre_act_opacities = torch.logit(clipped)

def run_training(args):

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    improved = args.mode == "improved"

    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name="materials", data_root=args.data_path,
        image_size=[128, 128],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    train_itr = iter(train_loader)

    # Preparing some code for visualization
    viz_gif_path_1 = os.path.join(args.out_path, "part_7_2_training_progress.gif")
    viz_gif_path_2 = os.path.join(args.out_path, "part_7_2_training_final_renders.gif")
    viz_idxs = np.linspace(0, len(train_dataset)-1, 5).astype(np.int32)[:4]

    gt_viz_imgs = [(train_dataset[i]["image"]*255.0).numpy().astype(np.uint8) for i in viz_idxs]
    gt_viz_imgs = [np.array(Image.fromarray(x).resize((256, 256))) for x in gt_viz_imgs]
    gt_viz_img = np.concatenate(gt_viz_imgs, axis=1)

    viz_cameras = [ndc_to_screen_camera(train_dataset[i]["camera"]).cuda() for i in viz_idxs]

    # Init gaussians and scene
    gaussians = Gaussians(
        num_points=args.num_points, init_type="random",
        device=args.device, isotropic=not improved
    )
    if improved:
        init_improved_gaussians(gaussians, args.init_radius, args.init_opacity)
    scene = Scene(gaussians)

    # Making gaussians trainable and setting up optimizer
    lrs = IMPROVED_LRS if improved else BASELINE_LRS
    make_trainable(gaussians)
    optimizer = setup_optimizer(gaussians, lrs)

    ssim_window = gaussian_window(device=args.device)
    grad_accum = torch.zeros(len(gaussians), device=args.device)
    grad_count = torch.zeros(len(gaussians), device=args.device)

    # Training loop
    viz_frames = []
    t0 = time.time()
    for itr in range(args.num_itrs):

        if improved:
            # Exponential decay of the means learning rate
            t = itr / max(1, args.num_itrs - 1)
            means_lr = math.exp(
                (1 - t) * math.log(lrs["means"]) + t * math.log(IMPROVED_MEANS_LR_FINAL)
            )
            for g in optimizer.param_groups:
                if g["name"] == "means":
                    g["lr"] = means_lr

        # Fetching data
        try:
            data = next(train_itr)
        except StopIteration:
            train_itr = iter(train_loader)
            data = next(train_itr)

        gt_img = data[0]["image"].cuda()
        camera = ndc_to_screen_camera(data[0]["camera"]).cuda()

        # Rendering scene using gaussian splatting
        ### YOUR CODE HERE ###
        pred_img = render_img(scene, camera, args)

        # Compute loss
        ### YOUR CODE HERE ###
        l1 = torch.nn.functional.l1_loss(pred_img, gt_img)
        if improved:
            d_ssim = 1.0 - ssim_torch(pred_img, gt_img, ssim_window)
            loss = (1.0 - args.lambda_ssim) * l1 + args.lambda_ssim * d_ssim
        else:
            loss = l1

        loss.backward()

        if improved:
            with torch.no_grad():
                grad_accum += gaussians.means.grad.norm(dim=-1)
                grad_count += (gaussians.means.grad.abs().sum(dim=-1) > 0).float()

        optimizer.step()
        optimizer.zero_grad()

        # Simplified adaptive density control. The optimizer is rebuilt since
        # the parameter tensors change shape.
        if improved and args.densify_from <= itr < args.densify_until \
                and (itr + 1) % args.densify_every == 0:
            n = densify_and_prune(gaussians, grad_accum, grad_count, args)
            if (itr + 1) % args.opacity_reset_every == 0:
                reset_opacity(gaussians)
            make_trainable(gaussians)
            optimizer = setup_optimizer(gaussians, lrs)
            grad_accum = torch.zeros(n, device=args.device)
            grad_count = torch.zeros(n, device=args.device)

        if itr % 50 == 0:
            print(f"[*] Itr: {itr:07d} | Loss: {loss:0.4f} | N: {len(gaussians)} "
                  f"| {time.time() - t0:.0f}s", flush=True)

        if itr % args.viz_freq == 0:
            viz_frame = visualize_renders_chunked(
                scene, gt_viz_img, viz_cameras, args
            )
            viz_frames.append(viz_frame)

    print("[*] Training Completed.")
    train_time = time.time() - t0

    # Saving training progess GIF
    imageio.mimwrite(viz_gif_path_1, viz_frames, loop=0, duration=(1/10.0)*1000)

    # Creating renderings of the training views after training is completed.
    frames = []
    viz_loader = DataLoader(
        train_dataset, batch_size=1, shuffle=False, num_workers=0,
        drop_last=True, collate_fn=trivial_collate
    )
    for viz_data in tqdm(viz_loader, desc="Creating Visualization"):
        gt_img = viz_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(viz_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            pred_img = render_img(scene, camera, args)

        pred_npy = pred_img.detach().cpu().numpy()
        pred_npy = (np.clip(pred_npy, 0.0, 1.0) * 255.0).astype(np.uint8)
        frames.append(pred_npy)

    # Saving renderings
    imageio.mimwrite(viz_gif_path_2, frames, loop=0, duration=(1/10.0)*1000)

    # Running evaluation using the test dataset
    psnr_vals, ssim_vals = [], []
    for val_data in tqdm(val_loader, desc="Running Evaluation"):

        gt_img = val_data[0]["image"].cuda()
        camera = ndc_to_screen_camera(val_data[0]["camera"]).cuda()

        with torch.no_grad():

            # Rendering scene using gaussian splatting
            ### YOUR CODE HERE ###
            pred_img = render_img(scene, camera, args)

            gt_npy = gt_img.detach().cpu().numpy()
            pred_npy = pred_img.detach().cpu().numpy()
            psnr = peak_signal_noise_ratio(gt_npy, pred_npy)
            ssim = structural_similarity(gt_npy, pred_npy, channel_axis=-1, data_range=1.0)

            psnr_vals.append(psnr)
            ssim_vals.append(ssim)

    mean_psnr = np.mean(psnr_vals)
    mean_ssim = np.mean(ssim_vals)
    print(f"[*] Evaluation --- Mean PSNR: {mean_psnr:.3f}")
    print(f"[*] Evaluation --- Mean SSIM: {mean_ssim:.3f}")

    with open(os.path.join(args.out_path, "metrics.json"), "w") as f:
        json.dump({
            "mode": args.mode, "psnr": float(mean_psnr), "ssim": float(mean_ssim),
            "num_itrs": args.num_itrs, "learning_rates": lrs,
            "num_gaussians_final": len(gaussians), "train_time_s": train_time,
            "args": vars(args),
        }, f, indent=2)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_path", default="./output", type=str,
        help="Path to the directory where output should be saved to."
    )
    parser.add_argument(
        "--data_path", default="./data/materials", type=str,
        help="Path to the dataset."
    )
    parser.add_argument(
        "--gaussians_per_splat", default=-1, type=int,
        help=(
            "Number of gaussians to splat in one function call. If set to -1, "
            "then all gaussians in the scene are splat in a single function call. "
            "If set to any other positive interger, then it determines the number of "
            "gaussians to splat per function call (the last function call might splat "
            "lesser number of gaussians). In general, the algorithm can run faster "
            "if more gaussians are splat per function call, but at the cost of higher GPU "
            "memory consumption."
        )
    )
    parser.add_argument(
        "--num_itrs", default=1000, type=int,
        help="Number of iterations to train the model."
    )
    parser.add_argument(
        "--viz_freq", default=20, type=int,
        help="Frequency with which visualization should be performed."
    )
    parser.add_argument("--device", default="cuda", type=str, choices=["cuda", "cpu"])

    # Q7.2 options. "baseline" is the Q6 setup, "improved" is the best method.
    parser.add_argument("--mode", default="baseline", choices=["baseline", "improved"])
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--num_points", default=10000, type=int)
    parser.add_argument("--init_radius", default=1.2, type=float)
    parser.add_argument("--init_opacity", default=0.1, type=float)
    parser.add_argument("--lambda_ssim", default=0.2, type=float)
    parser.add_argument("--densify_from", default=300, type=int)
    parser.add_argument("--densify_until", default=3000, type=int)
    parser.add_argument("--densify_every", default=100, type=int)
    parser.add_argument("--opacity_reset_every", default=1000, type=int)
    parser.add_argument("--densify_grad_thresh", default=2e-4, type=float)
    parser.add_argument("--densify_scale_thresh", default=0.03, type=float)
    parser.add_argument("--prune_opacity", default=0.005, type=float)
    parser.add_argument("--prune_scale", default=0.5, type=float)
    parser.add_argument("--max_gaussians", default=40000, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":

    args = get_args()
    run_training(args)

import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        seg_trans = torch.exp(-rays_density * deltas)
        transmittance = torch.cumprod(
            torch.cat([torch.ones_like(seg_trans[:, :1]), seg_trans[:, :-1] + eps], dim=1),
            dim=1,
        )

        # TODO (1.5): Compute weight used for rendering from transmittance and alpha
        weights = transmittance * (1.0 - seg_trans)
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        feature = torch.sum(weights * rays_feature, dim=1)

        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)
            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))
            if self._white_background:
                feature = feature + (1.0 - weights.sum(dim=1))

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


# Q4.2: draw n_fine depths per ray from the piecewise-constant pdf given by weights
def sample_pdf(bins, weights, n_fine, deterministic=False):
    # bins: (R, Nc+1) bin edges, weights: (R, Nc)
    weights = weights + 1e-5
    pdf = weights / weights.sum(-1, keepdim=True)
    cdf = torch.cat([torch.zeros_like(pdf[:, :1]), torch.cumsum(pdf, -1)], -1)

    if deterministic:
        u = torch.linspace(0.0, 1.0, n_fine, device=cdf.device).expand(cdf.shape[0], n_fine)
    else:
        u = torch.rand(cdf.shape[0], n_fine, device=cdf.device)
    u = u.contiguous()

    idx = torch.searchsorted(cdf, u, right=True).clamp(1, cdf.shape[-1] - 1)
    lo, hi = idx - 1, idx
    cdf_lo, cdf_hi = cdf.gather(-1, lo), cdf.gather(-1, hi)
    bin_lo, bin_hi = bins.gather(-1, lo), bins.gather(-1, hi)
    t = (u - cdf_lo) / (cdf_hi - cdf_lo).clamp(min=1e-5)
    return bin_lo + t * (bin_hi - bin_lo)


# Q4.2: coarse pass with the stratified samples, then a fine pass on
# coarse + importance-sampled depths with a second network
class HierarchicalVolumeRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)
        self.n_pts_fine = cfg.n_pts_fine

    def _render(self, implicit_fn, ray_bundle):
        n_pts = ray_bundle.sample_shape[1]
        implicit_output = implicit_fn(ray_bundle)
        density = implicit_output['density']
        feature = implicit_output['feature']

        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        )[..., None]

        weights = self._compute_weights(
            deltas.view(-1, n_pts, 1),
            density.view(-1, n_pts, 1)
        )

        feature = self._aggregate(weights, feature.view(-1, n_pts, feature.shape[-1]))
        if self._white_background:
            feature = feature + (1.0 - weights.sum(dim=1))
        depth = self._aggregate(weights, depth_values.view(-1, n_pts, 1))

        return feature, depth, weights[..., 0]

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        implicit_fn_fine=None,
    ):
        B = ray_bundle.shape[0]
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Coarse pass
            coarse_bundle = sampler(cur_ray_bundle)
            feature_coarse, _, weights = self._render(implicit_fn, coarse_bundle)

            # Importance-sample fine depths from the (detached) coarse weights
            z_coarse = coarse_bundle.sample_lengths[..., 0]
            bins = 0.5 * (z_coarse[..., 1:] + z_coarse[..., :-1])
            z_fine = sample_pdf(
                bins, weights[..., 1:-1].detach(), self.n_pts_fine,
                deterministic=not self.training,
            )
            z_all, _ = torch.sort(torch.cat([z_coarse, z_fine.detach()], dim=-1), dim=-1)

            # Fine pass on all samples
            sample_points = (
                coarse_bundle.origins[:, None, :]
                + z_all[..., None] * coarse_bundle.directions[:, None, :]
            )
            fine_bundle = coarse_bundle._replace(
                sample_points=sample_points,
                sample_lengths=z_all[..., None],
            )
            feature, depth, _ = self._render(implicit_fn_fine, fine_bundle)

            chunk_outputs.append({
                'feature': feature,
                'feature_coarse': feature_coarse,
                'depth': depth,
            })

        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q4.3): Convert signed distance to density with alpha, beta parameters
    # VolSDF: alpha * Psi_beta(-s), where Psi_beta is the Laplace(0, beta) CDF
    half_exp = 0.5 * torch.exp(-torch.abs(signed_distance) / beta)
    return alpha * torch.where(signed_distance > 0, half_exp, 1.0 - half_exp)

class VolumeSDFRenderer(VolumeRenderer):
    def __init__(
        self,
        cfg
    ):
        super().__init__(cfg)

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

        self.cfg = cfg

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
        light_dir=None
    ):
        B = ray_bundle.shape[0]

        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            density = sdf_to_density(distance, self.alpha, self.beta) # TODO (Q4.3): convert SDF to density

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            geometry_color = torch.zeros_like(color)

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'feature': color,
                "geometry": geometry_color
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer,
    'volume_sdf': VolumeSDFRenderer,
    'volume_hierarchical': HierarchicalVolumeRenderer,
}

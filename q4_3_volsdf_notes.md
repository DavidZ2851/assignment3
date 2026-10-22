# Q4.3 VolSDF notes

## What changed (all under `nerf/`)

- `renderer.py:223` `sdf_to_density`: VolSDF formula, `alpha * Psi_beta(-s)` where Psi_beta is the CDF of a zero-mean Laplace with scale beta.
  - s > 0 (outside): `alpha * 0.5 * exp(-s / beta)`
  - s <= 0 (inside): `alpha * (1 - 0.5 * exp(s / beta))`
- `renderer.py:264` `VolumeSDFRenderer.forward`: `density = sdf_to_density(distance, self.alpha, self.beta)` (was `None`).
- `losses.py:4` `eikonal_loss`: `((||grad|| - 1)^2).mean()`.
- `implicit.py:306` `NeuralSurface`:
  - `__init__`: harmonic embedding, then `MLPWithInputSkips` (6 layers, 128 wide) as the shared trunk, then `distance_head` (ReLU + Linear to 1, no output activation because distance is signed and unbounded), plus `mlp_color` (2 layers, Sigmoid output) on top of the trunk features.
  - `_features`, `_color_from_features`: small helpers so distance and color share the trunk.
  - `get_distance`, `get_color`, `get_distance_color`: implemented. `get_distance_color` runs the trunk once.

Nothing else was changed, and nothing was committed.

## Runs (lego, 128x128, 250 epochs, eikonal weight 0.02)

| alpha | beta | final loss | result |
|---|---|---|---|
| 10 | 0.2  | 0.011 | trains, but color is blurrier and the mesh is blobby with a thick slab of floor geometry |
| 10 | 0.05 | 0.008 | **best**: sharp color, recognizable bulldozer mesh (some floater bits under the base) |
| 10 | 0.01 | 0.039 | failed: density never formed, renders mostly black, marching cubes found an empty mesh |

GIFs are in `nerf/images/`: `part_4_3_a10_b{0.2,0.05,0.01}.gif` and `part_4_3_geometry_a10_b{...}.gif`. `part_4_3.gif` and `part_4_3_geometry.gif` are set to the best run (beta 0.05).

Why beta 0.05 works: it is small enough that density is concentrated in a thin shell around the zero level set, so the surface is sharp, but large enough that early in training points near (not exactly on) the surface still get density and gradients, so optimization can find the surface.

## Write-up answers

**What alpha and beta do.** Density is alpha times a smoothed step function of the negative SDF. alpha is the density value deep inside the object, so it controls how opaque the solid is. beta is the width of the smoothing: density rises from near 0 outside to alpha inside over a band of roughly a few beta around the surface. As beta goes to 0 it becomes a hard step (a perfectly sharp surface).

1. **How beta biases the SDF.** High beta spreads density over a wide band around the surface, so the volume looks like a soft, blurry shell. The learned SDF only needs to be roughly right, and the surface ends up smoothed and inflated, losing thin parts. Low beta makes density a near-step at the zero level set, which pushes the SDF to put its zero crossing exactly at the visible surface, so it is biased toward sharp, thin surfaces.
2. **Easier to train?** High beta. Each sample along a ray gets non-zero density and gradient even when the SDF is still far from correct, so the gradient signal is smooth and dense. With low beta almost every sample is either empty or fully opaque, so gradients vanish except right at the surface and training can stall. The beta 0.01 run collapsed for this reason.
3. **More accurate surface?** Low beta, as long as it still trains. Volume rendering with a narrow transition means the rendered color comes from points very close to the zero level set, so the zero level set has to match the real surface. High beta lets the surface sit anywhere inside a thick soft band and still render correctly, so the extracted mesh is less accurate. In practice a middle value (0.05 here) gives the best trade-off.

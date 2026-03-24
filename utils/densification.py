"""
Densification and refinement pipelines for the tetrahedral mesh.

## Why densification needs retriangulation

The tile-based rasterizer sorts tetrahedra by depth per tile. This sorting
requires a valid Delaunay triangulation — if the tet topology doesn't match
the vertex positions, tets overlap incorrectly and rendering breaks. Any
operation that adds or moves vertices must be followed by a Delaunay
retriangulation to restore correct sorting order.

## The refinement pipeline

Refinement (`refine_bad_tets`) inserts vertices at circumcenters of
poor-quality tetrahedra. Its purpose is NOT to add model capacity — it's to
**maintain triangulation quality** so that the Delaunay produces well-shaped
tets that sort and render correctly. Without refinement, densification can
create sliver tets that cause rendering artifacts.

The pipeline after each densification event:
  1. Densification adds vertices at error-targeted locations → Delaunay
  2. Refinement inserts vertices at bad-quality tets → Delaunay (fixes slivers)
  3. Decimation removes redundant vertices → Delaunay (frees budget)

Each step triggers a Delaunay retriangulation. Each retriangulation changes
the tet topology, which disrupts the learned vertex attribute interpolation
by ~0.1-0.5 dB. The cost is cumulative — more events = more disruption.

## Decimation

Decimation (`apply_decimation`) collapses edges to remove vertices that
contribute little to rendering quality. Its purpose is to **free vertex
budget** by removing vertices in uniform/transparent regions, making room
for vertices in high-detail regions. The scoring heuristic
`edge_length / (rgb_std + eps)` prioritizes short edges in uniform-color
regions, weighted by density to protect opaque surfaces.

## MCMC relocation

MCMC relocation (`apply_mcmc_relocation`) combines decimation scoring with
error targeting: it identifies expendable vertices (via decimation heuristic)
and teleports them to high-error locations (via render stats). This
redistributes vertices without changing the total count. It still requires
Delaunay because vertex positions change.

## Measured costs (at 400-500k vertices, 20-min budget)

Operation              | PSNR disruption | Wall time | Notes
-----------------------|-----------------|-----------|------
Densification+Delaunay | ~0 (new verts)  | ~25s      | New verts start at interpolated attrs
Refine+Delaunay        | -0.1 to -0.6 dB | ~0.5s     | Topology change hurts; diminishing returns at scale
Decimation+Delaunay    | -0.0 to -0.1 dB | ~1s       | Removing redundant verts is cheap
MCMC+Delaunay          | -0.0 to -0.1 dB | ~20s      | Nearly free at 2% relocation rate
Delaunay alone (187ms) | -0.1 to -0.3 dB | ~0.2s     | Topology disruption, not compute, is the cost
"""

import gc
import cv2
import torch
from utils import safe_math
from typing import NamedTuple, List
from rmesh_renderer.render_err import render_err
from utils.decimation import build_edge_list
from icecream import ic

def get_approx_ray_intersections(split_rays_data, epsilon=1e-7):
    """
    Calculates the approximate intersection point for pairs of line segments.

    The intersection is defined as the midpoint of the shortest segment
    connecting the two input line segments.

    Args:
        split_rays_data (torch.Tensor): Tensor of shape (N, 2, 6).
            - N: Number of segment pairs.
            - 2: Represents the two segments in a pair.
            - 6: Contains [Ax, Ay, Az, Bx, By, Bz] for each segment,
                 where A and B are the segment endpoints.
                 Based on current Python code:
                 A = average_P_exit, B = average_P_entry
        epsilon (float): Small value to handle parallel lines and avoid
                         division by zero if a segment has zero length.

    Returns:
        torch.Tensor: Tensor of shape (N, 3) representing the approximate
                      "intersection" points (midpoints of closest approach).
    """
    # Segment 1 endpoints
    p1_a = split_rays_data[:, 0, 0:3]  # Endpoint A of first segments (N, 3)
    p1_b = split_rays_data[:, 0, 3:6]  # Endpoint B of first segments (N, 3)
    # Segment 2 endpoints
    p2_a = split_rays_data[:, 1, 0:3]  # Endpoint A of second segments (N, 3)
    p2_b = split_rays_data[:, 1, 3:6]  # Endpoint B of second segments (N, 3)

    # Define segment origins and direction vectors
    # Segment S1: o1 + s * d1, for s in [0, 1]
    # Segment S2: o2 + t * d2, for t in [0, 1]
    o1 = p1_a
    d1 = p1_b - p1_a  # Direction vector for segment 1 (from A to B)
    o2 = p2_a
    d2 = p2_b - p2_a  # Direction vector for segment 2 (from A to B)

    # Calculate terms for finding closest points on the infinite lines
    # containing the segments (based on standard formulas, e.g., Christer Ericson's "Real-Time Collision Detection")
    v_o = o1 - o2 # Vector from origin of line 2 to origin of line 1

    a = torch.sum(d1 * d1, dim=1)  # Squared length of d1
    b = torch.sum(d1 * d2, dim=1)  # Dot product of d1 and d2
    c = torch.sum(d2 * d2, dim=1)  # Squared length of d2
    d = torch.sum(d1 * v_o, dim=1) # d1 dot (o1 - o2)
    e = torch.sum(d2 * v_o, dim=1) # d2 dot (o1 - o2)

    denom = a * c - b * b
    
    s_line_num = (b * e) - (c * d)
    t_line_num = (a * e) - (b * d) # This corresponds to t_c = (a*e - b*d)/denom from previous thoughts for P(t) = O2 + tD2

    # Handle near-zero denominator (lines are parallel or one segment is a point)
    # We compute with a safe denominator, then clamp. Clamping is key for segments.
    denom_safe = torch.where(denom.abs() < epsilon, torch.ones_like(denom), denom)
    
    s_line = s_line_num / denom_safe
    t_line = t_line_num / denom_safe # Note: This t_line is for the parameter of d2 (from o2)

    # Clamp parameters to [0, 1] to stay within the segments
    bad_intersect = (s_line < 0) | (t_line < 0) | (s_line > 1) | (t_line > 1)
    s_seg = torch.clamp(s_line, 0.0, 1.0)
    t_seg = torch.clamp(t_line, 0.0, 1.0)

    # Points of closest approach on the segments
    pc1 = o1 + s_seg.unsqueeze(1) * d1
    pc2 = o2 + t_seg.unsqueeze(1) * d2
    
    p_int = (pc1 + pc2) / 2.0
                        
    return p_int, bad_intersect

class RenderStats(NamedTuple):
    within_var_rays: torch.Tensor         # (T, 2, 6)
    total_var_moments: torch.Tensor     # (T, 3)
    tet_moments: torch.Tensor           # (T, 4)
    tet_view_count: torch.Tensor             # (T,)
    peak_contrib: torch.Tensor              # (T,)
    top_ssim: torch.Tensor
    top_size: torch.Tensor


@torch.no_grad()
def collect_render_stats(
    sampled_cameras: List["Camera"],
    model,
    args,
    device: torch.device,
):
    n_tets = model.indices.shape[0]

    # Pre-allocate accumulators ------------------------------------------------
    tet_moments = torch.zeros((n_tets, 4), device=device)
    tet_view_count = torch.zeros((n_tets,), device=device)

    top_ssim = torch.zeros((n_tets, 2), device=device)
    top_size = torch.zeros((n_tets, 2), device=device)
    peak_contrib = torch.zeros((n_tets), device=device)
    within_var_rays = torch.zeros((n_tets, 2, 6), device=device)
    total_var_moments = torch.zeros((n_tets, 3), device=device)
    top_moments = torch.zeros((n_tets, 2, 4), device=device)

    # Main per-camera loop -----------------------------------------------------
    for cam in sampled_cameras:
        target = cam.original_image.cuda()
        gt_mask = cam.gt_alpha_mask.cuda()

        image_votes, extras = render_err( target, gt_mask, cam, model, tile_size=args.tile_size, n_quad_samples=getattr(args, 'n_quad_samples', 2))

        tc = extras["tet_count"][..., 0]
        max_T = extras["tet_count"][..., 1].float() / 65535
        peak_contrib = torch.maximum(max_T, peak_contrib)
        
        # --- Create a single mask for valid updates ---
        # Mask for tets that have a reasonable number of samples in the current view
        update_mask = (tc >= args.min_tet_count) & (tc < 8000)

        # --- Moments (s0: sum of T, s1: sum of err, s2: sum of err^2)
        image_T, image_err, image_err2 = image_votes[:, 0], image_votes[:, 1], image_votes[:, 2]
        _, _, image_ssim = image_votes[:, 3], image_votes[:, 4], image_votes[:, 5]
        N = tc
        image_ssim[~update_mask] = 0

        # -------- Within-Image Variance (Top-2 per tet) -----------------------
        within_var_mu = safe_math.safe_div(image_err, N)
        within_var_std = (safe_math.safe_div(image_err2, N) - within_var_mu**2).clip(min=0)
        within_var_std[N < 10] = 0
        within_var_std[~update_mask] = 0 # Use the unified mask

        # ray buffer: (enter | exit) → (N, 6)
        w = image_votes[:, 12:13]
        seg_exit = safe_math.safe_div(image_votes[:, 9:12], w)
        seg_enter = safe_math.safe_div(image_votes[:, 6:9], w)

        image_ssim = image_ssim / tc.clip(min=1)

        # keep top-2 candidates per tet across all views
        top_ssim, idx_sorted = torch.cat([top_ssim[:, :2], image_ssim.reshape(-1, 1)], dim=1).sort(1, descending=True)

        # -------- Other stats -------------------------------------------------
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)
        moments = torch.cat([
            image_votes[:, 13:16],
            w.reshape(-1, 1)
        ], dim=1)
        moments_3 = torch.cat([top_moments, moments.reshape(-1, 1, 4)], dim=1)
        top_moments = torch.gather(
            moments_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 4)
        )

        rays = torch.cat([seg_enter, seg_exit], dim=1)
        rays_3 = torch.cat([within_var_rays, rays[:, None]], dim=1)
        within_var_rays = torch.gather(
            rays_3, 1,
            idx_sorted[:, :2, None].expand(-1, -1, 6)
        )

        # -------- Total Variance (accumulated across images) ------------------
        total_var_moments[update_mask, 0] += image_T[update_mask]
        total_var_moments[update_mask, 1] += image_err[update_mask]
        total_var_moments[update_mask, 2] += image_err2[update_mask]

        # -------- Between-Image Variance (accumulated across images) ----------
        # We compute the variance of the mean error across different views
        mean_err_per_view = within_var_mu
        mean_err_per_view[N < 10] = 0

        # -------- Other stats -------------------------------------------------
        tet_moments[update_mask, :3] += image_votes[update_mask, 13:16]
        tet_moments[update_mask, 3] += w[update_mask].reshape(-1)

        tet_view_count[update_mask] += 1 # Count views per tet

    # done
    return RenderStats(
        within_var_rays = within_var_rays,
        total_var_moments = total_var_moments,
        tet_moments = tet_moments,
        tet_view_count = tet_view_count,
        top_ssim = top_ssim[:, :2],
        top_size = top_size[:, :2],
        peak_contrib = peak_contrib # used for determining what to clone
    )

@torch.no_grad()
def apply_densification(
    stats: RenderStats,
    model,
    tet_optim,
    args,
    device: torch.device,
    target_addition
):
    """Turns accumulated statistics into actual vertex cloning / splitting."""
    # ---------- Calculate scores from variances ------------------------------
    # 1. Total Variance Score (for growing)
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (safe_math.safe_div(s2_t, s0_t) - total_var_mu**2).clip(min=0)
    total_var_std[s0_t < 1] = 0

    N_b = stats.tet_view_count # Num views
    within_var = stats.top_ssim.sum(dim=1)
    total_var = s0_t * total_var_std
    total_var[(N_b < 2) | (s0_t < 1)] = 0

    # --- Masking and target calculation --------------------------------------
    mask_alive = stats.peak_contrib > args.clone_min_contrib
    total_var[stats.peak_contrib < args.clone_min_contrib] = 0
    within_var[stats.peak_contrib < args.split_min_contrib] = 0

    target_addition = int(min(target_addition, stats.tet_view_count.shape[0]))
    if target_addition < 0:
        return


    total_mask = torch.zeros_like(total_var, dtype=torch.bool)
    within_mask = torch.zeros_like(total_mask)

    within_mask = (within_var > args.within_thresh)
    total_mask = (total_var > args.total_thresh)
    clone_mask = within_mask | total_mask
    if clone_mask.sum() > target_addition:
        true_indices = clone_mask.nonzero().squeeze(-1)
        perm = torch.randperm(true_indices.size(0))
        selected_indices = true_indices[perm[:target_addition]]
        
        clone_mask = torch.zeros_like(clone_mask, dtype=torch.bool)
        clone_mask[selected_indices] = True

    # 1-to-4 split at centroids (with Delaunay for correct sort order)
    tet_optim.split_tets_inplace(clone_mask)

    gc.collect()
    torch.cuda.empty_cache()

    print(
        f"#Grow: {total_mask.sum():4d} #Split: {within_mask.sum():4d} | "
        f"#Alive: {mask_alive.sum():4d} | "
        f"Total Avg: {total_var.mean():.4f} Within Avg: {within_var.mean():.4f} "
    )


@torch.no_grad()
def apply_mcmc_relocation(
    model,
    tet_optim,
    args,
    device: torch.device,
    max_relocate: int = 10000,
):
    """MCMC-style edge collapse + edge split. Net zero vertex change.

    1. Score all edges (low score = expendable, high score = important)
    2. Greedy conflict resolution to select non-overlapping edges
    3. Collapse lowest-score edges (merge to midpoint, remove one vertex)
    4. Split highest-score edges (add midpoint vertex)
    5. Single retriangulation
    """
    from utils.decimation import build_edge_list, compute_edge_scores, query_tet_rgb

    n_int = model.num_int_verts
    sigma = model.sigma.data.squeeze()

    # 1. Score edges
    tet_rgb = query_tet_rgb(model)
    edges = build_edge_list(model.indices)
    scores = compute_edge_scores(edges, model.indices, model.vertices, tet_rgb, n_int)

    # Density weighting: protect high-density edges from collapse
    edge_density = (sigma[edges[:, 0]] + sigma[edges[:, 1]]) / 2
    density_weight = torch.exp(edge_density.clamp(max=5))
    scores = scores * density_weight

    # Interior-only mask
    interior = (edges[:, 0] < n_int) & (edges[:, 1] < n_int)

    # 2. Greedy conflict resolution for collapse candidates (lowest scores)
    order_low = torch.argsort(scores)
    finite_mask = scores[order_low] < float('inf')
    order_low = order_low[finite_mask & interior[order_low]]

    def greedy_select(ordered_edges, max_k):
        """Vectorized greedy: first edge claiming a vertex wins."""
        if ordered_edges.shape[0] == 0:
            return ordered_edges[:0]
        flat = ordered_edges.reshape(-1)
        eidx = torch.arange(ordered_edges.shape[0], device=device).unsqueeze(1).expand_as(ordered_edges).reshape(-1)
        first = torch.full((model.vertices.shape[0],), ordered_edges.shape[0], device=device, dtype=torch.long)
        first.scatter_reduce_(0, flat, eidx, reduce='amin', include_self=True)
        arange = torch.arange(ordered_edges.shape[0], device=device)
        valid = (first[ordered_edges[:, 0]] == arange) & (first[ordered_edges[:, 1]] == arange)
        return ordered_edges[valid][:max_k]

    collapse_edges = greedy_select(edges[order_low], max_relocate)
    n_collapse = collapse_edges.shape[0]
    if n_collapse == 0:
        return 0

    # Pick va (higher density, survives) and vb (lower density, removed)
    sa = sigma[collapse_edges[:, 0]]
    sb = sigma[collapse_edges[:, 1]]
    collapse_va = torch.where(sa >= sb, collapse_edges[:, 0], collapse_edges[:, 1])
    collapse_vb = torch.where(sa >= sb, collapse_edges[:, 1], collapse_edges[:, 0])

    # 3. Select split candidates (highest scores) — must not overlap with collapse vertices
    order_high = torch.argsort(scores, descending=True)
    order_high = order_high[interior[order_high]]
    # Exclude edges that share vertices with collapse edges
    collapsed_verts = torch.zeros(model.vertices.shape[0], dtype=torch.bool, device=device)
    collapsed_verts[collapse_va] = True
    collapsed_verts[collapse_vb] = True
    high_edges = edges[order_high]
    no_overlap = ~collapsed_verts[high_edges[:, 0]] & ~collapsed_verts[high_edges[:, 1]]
    high_edges = high_edges[no_overlap]

    split_edges = greedy_select(high_edges, n_collapse)
    n_split = split_edges.shape[0]

    # 4. Compute new vertex positions at circumcenters of tets adjacent to split edges
    if n_split > 0:
        from utils.topo_utils import calculate_circumcenters_torch

        indices = model.indices.long()
        verts = model.vertices
        T = indices.shape[0]

        # Build edge→tet mapping: for each split edge, find one tet containing it
        # Pack edges as int64 keys for fast lookup
        V_max = verts.shape[0]
        split_keys = split_edges[:, 0].long() * V_max + split_edges[:, 1].long()

        # Generate all 6 edges per tet
        pair_offsets = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        all_va, all_vb, all_tet = [], [], []
        tet_arange = torch.arange(T, device=device)
        for i, j in pair_offsets:
            ei, ej = indices[:, i], indices[:, j]
            va = torch.min(ei, ej)
            vb = torch.max(ei, ej)
            all_va.append(va)
            all_vb.append(vb)
            all_tet.append(tet_arange)
        all_va = torch.cat(all_va)
        all_vb = torch.cat(all_vb)
        all_tet = torch.cat(all_tet)
        all_keys = all_va * V_max + all_vb

        # For each split edge, find one matching tet via searchsorted
        sort_order = torch.argsort(all_keys)
        sorted_keys = all_keys[sort_order]
        sorted_tets = all_tet[sort_order]
        lookup_idx = torch.searchsorted(sorted_keys, split_keys)
        lookup_idx = lookup_idx.clamp(0, sorted_keys.shape[0] - 1)
        # Verify match
        matched = sorted_keys[lookup_idx] == split_keys
        target_tet_idx = sorted_tets[lookup_idx]

        # Compute circumcenters for matched tets
        p = verts[indices[target_tet_idx]]  # (S, 4, 3)
        cc, _ = calculate_circumcenters_torch(p.double())
        cc = cc.float()

        # Filter degenerate circumcenters
        centroid = p.mean(dim=1)
        cc_dist = (cc - centroid).norm(dim=1)
        edges_pairs_list = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        min_el = torch.full((n_split,), float('inf'), device=device)
        for i, j in edges_pairs_list:
            el = (p[:, i] - p[:, j]).norm(dim=1)
            min_el = torch.minimum(min_el, el)
        use_centroid = (~matched) | (cc_dist > min_el * 5)
        cc[use_centroid] = centroid[use_centroid]

        # Barycentric interpolation at circumcenter
        v0, v1, v2, v3 = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        def tet_vol_signed(a, b, c, d):
            return torch.sum((b - a) * torch.cross(c - a, d - a, dim=1), dim=1)

        vol_total = tet_vol_signed(v0, v1, v2, v3).unsqueeze(1)
        b0 = tet_vol_signed(cc, v1, v2, v3).unsqueeze(1) / (vol_total + 1e-12)
        b1 = tet_vol_signed(v0, cc, v2, v3).unsqueeze(1) / (vol_total + 1e-12)
        b2 = tet_vol_signed(v0, v1, cc, v3).unsqueeze(1) / (vol_total + 1e-12)
        b3 = 1.0 - b0 - b1 - b2
        bary = torch.cat([b0, b1, b2, b3], dim=1)
        outside = (bary < -0.1).any(dim=1) | (bary > 1.1).any(dim=1)
        bary[outside] = 0.25
        cc[outside] = centroid[outside]

        # Perturb to avoid degenerate Delaunay
        new_positions = cc + torch.randn_like(cc) * (min_el.unsqueeze(1) * 0.01)

        # Interpolate attributes — clamp exterior vertex indices to valid range
        tet_verts = indices[target_tet_idx]
        attr_idx = tet_verts.clamp(max=n_int - 1)  # exterior verts → use last interior's attrs
        new_sigma = (model.sigma.data[attr_idx] * bary.unsqueeze(-1)).sum(dim=1)
        new_rgb = (model.rgb.data[attr_idx] * bary.unsqueeze(-1)).sum(dim=1)
        sh_data = model.sh.data[attr_idx]
        new_sh = (sh_data * bary.unsqueeze(-1).unsqueeze(-1)).sum(dim=1)
    else:
        new_positions = torch.empty((0, 3), device=device)
        new_sigma = torch.empty((0, *model.sigma.shape[1:]), device=device)
        new_rgb = torch.empty((0, *model.rgb.shape[1:]), device=device)
        new_sh = torch.empty((0, *model.sh.shape[1:]), device=device)

    # 5. Combined collapse + split with single retriangulation
    tet_optim.collapse_and_split(collapse_va, collapse_vb,
                                 new_positions, new_sigma, new_rgb, new_sh)

    print(f"MCMC: collapsed {n_collapse} edges, split {n_split} edges "
          f"(#V: {model.vertices.shape[0]}, #T: {model.indices.shape[0]})")
    return n_collapse


@torch.no_grad()
def apply_vertex_densification(
    stats: RenderStats,
    model,
    tet_optim,
    args,
    device: torch.device,
    target_addition: int,
):
    """Vertex-error-based densification for VertexModel.

    Converts per-tet error to per-vertex error, then adds new vertices
    near the highest-error vertices by splitting their longest edges.
    """
    from utils.decimation import build_edge_list

    # 1. Compute per-tet error (same as apply_densification)
    s0_t, s1_t, s2_t = stats.total_var_moments.T
    total_var_mu = safe_math.safe_div(s1_t, s0_t)
    total_var_std = (safe_math.safe_div(s2_t, s0_t) - total_var_mu**2).clip(min=0)
    total_var_std[s0_t < 1] = 0
    tet_error = s0_t * total_var_std
    tet_error[stats.tet_view_count < 2] = 0
    tet_error[stats.peak_contrib < args.clone_min_contrib] = 0

    # 2. Scatter tet error to vertices (max of adjacent tets)
    indices = model.indices.long()
    V = model.vertices.shape[0]
    n_int = model.num_int_verts
    vertex_error = torch.zeros(V, device=device)
    for corner in range(4):
        vertex_error.scatter_reduce_(0, indices[:, corner], tet_error, reduce='amax')
    # Only interior vertices
    vertex_error[n_int:] = 0

    # 3. Build edges, score by max endpoint error
    edges = build_edge_list(model.indices)
    int_mask = (edges[:, 0] < n_int) & (edges[:, 1] < n_int)
    edge_error = torch.zeros(edges.shape[0], device=device)
    edge_error[int_mask] = torch.maximum(
        vertex_error[edges[int_mask, 0]],
        vertex_error[edges[int_mask, 1]])

    # 4. Select top-k edges by error, weighted by edge length (prefer splitting longer edges)
    verts = model.vertices
    edge_len = (verts[edges[:, 0]] - verts[edges[:, 1]]).norm(dim=1)
    edge_score = edge_error * edge_len  # error * length: split long edges near high-error vertices

    k = min(target_addition, int(int_mask.sum().item()))
    if k <= 0:
        return
    _, top_idx = edge_score.topk(k)
    edges_to_split = edges[top_idx]

    n_selected = edges_to_split.shape[0]
    print(f"Vertex-error densification: splitting {n_selected} edges "
          f"(error range: {edge_error[top_idx].min():.4f} - {edge_error[top_idx].max():.4f})")

    # 5. Split edges at midpoints with averaged attributes + Delaunay
    tet_optim.add_vertices_midpoint(edges_to_split)


@torch.no_grad()
def apply_grad_densification(
    model,
    tet_optim,
    grad_accum: torch.Tensor,
    grad_count: torch.Tensor,
    target_addition: int,
    mode: str = "edge_midpoint",
):
    """Gradient-based densification: split edges or clone vertices with highest grad norms.

    Args:
        model: VertexModel
        tet_optim: VertexOptimizer
        grad_accum: (V_int,) accumulated gradient norms for interior vertices
        grad_count: (V_int,) number of accumulations per vertex
        target_addition: max number of new vertices to add
        mode: "edge_midpoint" or "clone"
    """
    n_int = model.num_int_verts
    v_grad = grad_accum / grad_count.clamp(min=1)
    v_grad[grad_count < 10] = 0  # need enough samples

    if target_addition <= 0:
        return

    if mode == "edge_midpoint":
        # Build edge list and score edges by sum of endpoint gradients
        edges = build_edge_list(model.indices)  # (E, 2)

        # Only split edges where both vertices are interior
        int_mask = (edges[:, 0] < n_int) & (edges[:, 1] < n_int)
        edge_score = torch.zeros(edges.shape[0], device=model.device)
        edge_score[int_mask] = v_grad[edges[int_mask, 0]] + v_grad[edges[int_mask, 1]]

        # Select top-k edges
        k = min(target_addition, int(int_mask.sum().item()))
        if k == 0:
            return
        _, top_idx = edge_score.topk(k)
        edges_to_split = edges[top_idx]

        print(f"Grad densify (edge_midpoint): splitting {k} edges "
              f"(grad range: {edge_score[top_idx].min():.4f} - {edge_score[top_idx].max():.4f})")
        tet_optim.add_vertices_midpoint(edges_to_split)

    elif mode == "clone":
        # Clone top-k vertices by gradient norm
        k = min(target_addition, n_int)
        if k == 0:
            return
        _, top_idx = v_grad[:n_int].topk(k)

        print(f"Grad densify (clone): cloning {k} vertices "
              f"(grad range: {v_grad[top_idx].min():.4f} - {v_grad[top_idx].max():.4f})")
        tet_optim.clone_vertices(top_idx, offset_scale=0.001)

    else:
        raise ValueError(f"Unknown mode: {mode}")

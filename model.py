import torch
from torch import nn
from typing import Optional, Tuple
import gc
import numpy as np
from pathlib import Path
import open3d as o3d

from gdel3d import Del
from scipy.spatial import Delaunay

from utils.topo_utils import (
    tet_volumes, calculate_circumcenters_torch,
    fibonacci_spiral_on_sphere,
)
from utils.model_util import activate_output, pre_calc_cell_values, offset_normalize, RGB2SH
from utils.eval_sh_py import eval_sh
from utils.safe_math import safe_exp
from utils import optim
from utils.args import Args

# ---------------------------------------------------------------------------
# Vertex-to-tet adjacency (from experiments/test_field_transfer.py)
# ---------------------------------------------------------------------------
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """
    lr_init = max(lr_init, 1e-20)
    lr_final = max(lr_final, 1e-20)

    def helper(step):
        if max_steps == 0:
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper

# ---------------------------------------------------------------------------
# Vertex-to-tet adjacency (from experiments/test_field_transfer.py)
# ---------------------------------------------------------------------------

def build_v2t(indices: torch.Tensor, n_verts: int):
    """Build padded vertex->tet adjacency table.

    Returns:
        v2t: (V, max_valence) int64, padded with -1
        valence: (V,) int64, actual count per vertex
    """
    T = indices.shape[0]
    device = indices.device
    flat_vidx = indices.long().reshape(-1)
    flat_tidx = torch.arange(T, device=device).repeat_interleave(4)

    sort_order = torch.argsort(flat_vidx)
    sorted_vidx = flat_vidx[sort_order]
    sorted_tidx = flat_tidx[sort_order]

    valence = torch.bincount(flat_vidx, minlength=n_verts)
    offsets = torch.zeros(n_verts + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(valence, dim=0)
    max_val = valence.max().item()

    group_starts = offsets[sorted_vidx]
    local_pos = torch.arange(len(sorted_vidx), device=device) - group_starts

    v2t = torch.full((n_verts, max_val), -1, dtype=torch.long, device=device)
    v2t[sorted_vidx, local_pos] = sorted_tidx.long()

    return v2t, valence


def _min_edge_length(vertices, indices):
    """Compute minimum edge length per tet. indices: (T, 4), returns (T,)."""
    p = vertices[indices.long()]  # (T, 4, 3)
    edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
    el = torch.stack([(p[:, a] - p[:, b]).norm(dim=-1) for a, b in edges])
    return el.min(dim=0).values


EDGE_PAIRS = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
_EDGE_I = [0, 0, 0, 1, 1, 2]
_EDGE_J = [1, 2, 3, 2, 3, 3]


def _canonical_edge_keys(indices: torch.Tensor, n_verts: int) -> torch.Tensor:
    """Return (T, 6) int64 canonical edge keys (min_v * n_verts + max_v)."""
    idx = indices.long()
    vi = idx[:, _EDGE_I]  # (T, 6)
    vj = idx[:, _EDGE_J]  # (T, 6)
    va = torch.min(vi, vj)
    vb = torch.max(vi, vj)
    return va * n_verts + vb


def _edge_keys_and_indices(indices: torch.Tensor, n_verts: int):
    """Compute unique edge keys and per-tet edge indices (no padded e2t table)."""
    all_keys = _canonical_edge_keys(indices, n_verts)  # (T, 6)
    flat_keys = all_keys.reshape(-1)
    unique_keys, inverse = torch.unique(flat_keys, sorted=True, return_inverse=True)
    tet_edge_idx = inverse.reshape(-1, 6)  # (T, 6)
    return unique_keys, tet_edge_idx


def build_e2t(indices: torch.Tensor, n_verts: int):
    """Build padded edge->tet adjacency table.

    Returns:
        unique_keys: (E,) sorted unique edge keys
        e2t: (E, max_valence) int64, padded with -1
        tet_edge_idx: (T, 6) index into unique_keys for each tet's 6 edges
        valence: (E,) int64, tet count per edge
    """
    T = indices.shape[0]
    device = indices.device

    # (T, 6) canonical edge keys
    all_keys = _canonical_edge_keys(indices, n_verts)  # (T, 6)
    flat_keys = all_keys.reshape(-1)  # (6T,)
    flat_tidx = torch.arange(T, device=device).unsqueeze(1).expand(-1, 6).reshape(-1)

    # Sort by key
    sort_order = torch.argsort(flat_keys)
    sorted_keys = flat_keys[sort_order]
    sorted_tidx = flat_tidx[sort_order]

    # Unique edges
    unique_keys, inverse = torch.unique(sorted_keys, return_inverse=True)
    E = unique_keys.shape[0]

    # Valence per edge
    valence = torch.bincount(inverse, minlength=E)
    max_val = valence.max().item()

    # Build padded e2t
    offsets = torch.zeros(E + 1, dtype=torch.long, device=device)
    offsets[1:] = torch.cumsum(valence, dim=0)
    group_starts = offsets[inverse]
    local_pos = torch.arange(len(inverse), device=device) - group_starts

    e2t = torch.full((E, max_val), -1, dtype=torch.long, device=device)
    e2t[inverse, local_pos] = sorted_tidx.long()

    # Per-tet edge indices into unique_keys: use searchsorted
    tet_edge_idx = torch.searchsorted(unique_keys, all_keys.reshape(-1))
    tet_edge_idx = tet_edge_idx.reshape(T, 6)

    return unique_keys, e2t, tet_edge_idx, valence


def compute_transfer_weights(
    v2t: torch.Tensor,              # (V, max_val) padded with -1
    new_indices: torch.Tensor,       # (T_new, 4) int
    old_indices: torch.Tensor,       # (T_old, 4) int
    old_cc: torch.Tensor,            # (T_old, 3)
    new_cc: torch.Tensor,            # (T_new, 3)
    vertex_positions: torch.Tensor,  # (V, 3)
    vert_chunk: int = 50_000,
):
    """For each new tet, compute blending weights across 4 candidate old tets.

    Two-stage approach:
    1. Per-vertex: find the nearest old tet (by CC distance to the vertex)
    2. Per-new-tet: compute weights based on vertex overlap + inverse CC distance

    Returns:
        cands: (T_new, 4) long — candidate old tet indices
        weights: (T_new, 4) float — normalized blending weights
        density_scale: (T_new, 4) float — per-candidate edge-length ratio
    """
    V = vertex_positions.shape[0]
    device = new_indices.device

    # Stage 1: per-vertex best old tet
    vert_best = torch.zeros(V, dtype=torch.long, device=device)
    for start in range(0, V, vert_chunk):
        end = min(start + vert_chunk, V)
        v_pos = vertex_positions[start:end]
        v_cands = v2t[start:end]
        valid = v_cands >= 0
        safe = v_cands.clamp(min=0)
        dists = (old_cc[safe] - v_pos.unsqueeze(1)).pow(2).sum(-1)
        dists[~valid] = float("inf")
        best_local = dists.argmin(dim=1)
        vert_best[start:end] = safe.gather(1, best_local.unsqueeze(1)).squeeze(1)

    del v2t

    # Stage 2: candidates + weights
    cands = vert_best[new_indices.long()]               # (T_new, 4)

    # Vertex overlap
    cand_verts = old_indices[cands].long()               # (T_new, 4, 4)
    new_verts = new_indices.long().unsqueeze(1)           # (T_new, 1, 4)
    matches = (cand_verts.unsqueeze(-1) == new_verts.unsqueeze(2))  # (T_new, 4, 4, 4)
    overlap = matches.any(dim=-1).sum(dim=-1).float()    # (T_new, 4)

    # CC distance
    cand_cc = old_cc[cands]                             # (T_new, 4, 3)
    cc_dist_sq = (cand_cc - new_cc.unsqueeze(1)).pow(2).sum(-1)  # (T_new, 4)

    # Weight: overlap bonus * inverse distance kernel
    # Tets with more shared vertices get exponentially more weight
    overlap_weight = torch.exp(overlap * 2.0)  # e^0=1, e^2≈7, e^4≈55, e^6≈403, e^8≈2981
    dist_weight = 1.0 / (cc_dist_sq + 1e-8)
    raw_weights = overlap_weight * dist_weight
    weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)  # (T_new, 4) normalized

    # Per-candidate density scale (edge-length ratio)
    new_el = _min_edge_length(vertex_positions, new_indices)  # (T_new,)
    old_el = torch.stack([
        _min_edge_length(vertex_positions, old_indices[cands[:, i]]) for i in range(4)
    ], dim=1)  # (T_new, 4)
    density_scale = (old_el / new_el.unsqueeze(1).clamp(min=1e-8)).clamp(min=0.1, max=10.0)

    return cands, weights, density_scale


def compute_transfer_weights_edge(
    v2t: torch.Tensor,              # (V, max_val) padded with -1
    new_indices: torch.Tensor,       # (T_new, 4) int
    old_indices: torch.Tensor,       # (T_old, 4) int
    old_cc: torch.Tensor,            # (T_old, 3)
    new_cc: torch.Tensor,            # (T_new, 3)
    vertex_positions: torch.Tensor,  # (V, 3)
    n_verts: int,
    old_unique_keys: torch.Tensor = None,
    old_tet_edge_idx: torch.Tensor = None,
    vert_chunk: int = 50_000,
):
    """For each new tet, compute blending weights across 6 candidate old tets (one per edge).

    Uses edge matching: shared edges (~85%) get direct old tet lookup via searchsorted,
    unmatched edges (~15%) fall back to v2t search without vertex membership check
    (shared-edge bonus is guaranteed zero for unmatched edges).

    Returns:
        cands: (T_new, 6) long — candidate old tet indices
        weights: (T_new, 6) float — normalized blending weights
        density_scale: (T_new, 6) float — per-candidate edge-length ratio
    """
    device = new_indices.device
    T_new = new_indices.shape[0]
    T_old = old_indices.shape[0]

    # --- Precompute old edge → tet mapping ---
    if old_unique_keys is None or old_tet_edge_idx is None:
        old_unique_keys, old_tet_edge_idx = _edge_keys_and_indices(old_indices, n_verts)

    E_old = old_unique_keys.shape[0]

    # Best (nearest-CC) old tet per old edge: sort by descending CC-to-midpoint
    # distance so the closest tet writes last and wins the scatter.
    old_idx_l = old_indices.long()
    old_vi = old_idx_l[:, _EDGE_I]  # (T_old, 6)
    old_vj = old_idx_l[:, _EDGE_J]
    old_va = torch.min(old_vi, old_vj)
    old_vb = torch.max(old_vi, old_vj)
    old_edge_midpts = (vertex_positions[old_va] + vertex_positions[old_vb]) * 0.5  # (T_old, 6, 3)
    flat_cc_dist = (old_cc.unsqueeze(1) - old_edge_midpts).pow(2).sum(-1).reshape(-1)  # (T_old*6,)

    sort_order = torch.argsort(flat_cc_dist, descending=True)
    flat_edge_idx = old_tet_edge_idx.reshape(-1)
    tet_idx_expanded = torch.arange(T_old, device=device).unsqueeze(1).expand(-1, 6).reshape(-1)

    old_edge_to_tet = torch.zeros(E_old, dtype=torch.long, device=device)
    old_edge_to_tet.scatter_(0, flat_edge_idx[sort_order], tet_idx_expanded[sort_order])

    # --- Compute new edge keys and match to old ---
    new_idx = new_indices.long()
    new_all_keys = _canonical_edge_keys(new_indices, n_verts)  # (T_new, 6)
    new_flat_keys = new_all_keys.reshape(-1)  # (T_new * 6,)

    match_pos = torch.searchsorted(old_unique_keys, new_flat_keys)
    match_pos = match_pos.clamp(0, E_old - 1)
    is_matched = old_unique_keys[match_pos] == new_flat_keys

    # --- Unchanged-tet skip: identify tets with identical sorted vertex quadruples ---
    # These get exact 1:1 copy (no blending dilution).
    old_sorted = old_indices.long().sort(dim=1).values  # (T_old, 4)
    new_sorted = new_indices.long().sort(dim=1).values  # (T_new, 4)

    # Find matching tets via torch.unique on concatenated sorted vertices
    all_sorted = torch.cat([old_sorted, new_sorted], dim=0)  # (T_old + T_new, 4)
    _, inverse = torch.unique(all_sorted, dim=0, sorted=True, return_inverse=True)
    old_inverse = inverse[:T_old]
    new_inverse = inverse[T_old:]

    # Map unique tet ID → old tet index (last-write-wins; no duplicates in valid Delaunay)
    n_unique = inverse.max().item() + 1
    unique_to_old = torch.full((n_unique,), -1, dtype=torch.long, device=device)
    unique_to_old.scatter_(0, old_inverse, torch.arange(T_old, device=device))

    matched_old_tet = unique_to_old[new_inverse]  # (T_new,) — -1 if no match
    is_tet_matched = matched_old_tet >= 0
    matched_old_tet = matched_old_tet.clamp(min=0)  # safe for indexing

    n_tet_matched = is_tet_matched.sum().item()

    # --- Fast path: matched edges → direct old tet lookup ---
    flat_cands = torch.zeros(T_new * 6, dtype=torch.long, device=device)
    flat_cands[is_matched] = old_edge_to_tet[match_pos[is_matched]]

    # --- Slow path: unmatched edges → v2t search (no vertex membership check) ---
    # Exclude edges belonging to unchanged tets (they'll be overwritten below)
    is_changed_tet = ~is_tet_matched  # (T_new,)
    needs_slow = ~is_matched.reshape(T_new, 6) & is_changed_tet.unsqueeze(1)
    unmatched_idx = torch.where(needs_slow.reshape(-1))[0]

    if unmatched_idx.numel() > 0:
        # Recover va, vb for unmatched edges
        vi_all = new_idx[:, _EDGE_I].reshape(-1)  # (T_new*6,)
        vj_all = new_idx[:, _EDGE_J].reshape(-1)
        va_all = torch.min(vi_all, vj_all)
        vb_all = torch.max(vi_all, vj_all)

        flat_va = va_all[unmatched_idx]
        flat_vb = vb_all[unmatched_idx]

        tet_chunk = 50_000
        N_unmatched = unmatched_idx.shape[0]

        for start in range(0, N_unmatched, tet_chunk):
            end = min(start + tet_chunk, N_unmatched)
            va = flat_va[start:end]
            vb = flat_vb[start:end]

            midpt = (vertex_positions[va] + vertex_positions[vb]) * 0.5

            all_cands = torch.cat([v2t[va], v2t[vb]], dim=1)  # (chunk, 2*max_val)
            valid = all_cands >= 0
            safe = all_cands.clamp(min=0)

            dists = (old_cc[safe] - midpt.unsqueeze(1)).pow(2).sum(-1)
            dists[~valid] = float("inf")

            # No vertex membership check — shared-edge bonus is guaranteed zero
            # for unmatched edges (no old tet contains both va and vb)
            best_local = dists.argmin(dim=1)
            flat_cands[unmatched_idx[start:end]] = safe.gather(1, best_local.unsqueeze(1)).squeeze(1)

    del v2t

    # Reshape to (T_new, 6)
    cands = flat_cands.reshape(T_new, 6)

    # For unchanged tets: set all 6 candidates to the matched old tet
    if n_tet_matched > 0:
        cands[is_tet_matched] = matched_old_tet[is_tet_matched].unsqueeze(1).expand(-1, 6)

    # Vertex overlap for 6 candidates
    cand_verts = old_indices[cands].long()               # (T_new, 6, 4)
    new_verts = new_indices.long().unsqueeze(1)           # (T_new, 1, 4)
    matches = (cand_verts.unsqueeze(-1) == new_verts.unsqueeze(2))  # (T_new, 6, 4, 4)
    overlap = matches.any(dim=-1).sum(dim=-1).float()    # (T_new, 6)

    # CC distance
    cand_cc = old_cc[cands]                             # (T_new, 6, 3)
    cc_dist_sq = (cand_cc - new_cc.unsqueeze(1)).pow(2).sum(-1)  # (T_new, 6)

    # Weight: overlap bonus * inverse distance kernel
    overlap_weight = torch.exp(overlap * 2.0)
    dist_weight = 1.0 / (cc_dist_sq + 1e-8)
    raw_weights = overlap_weight * dist_weight
    weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)  # (T_new, 6) normalized

    n_edge_matched = is_matched.sum().item()
    n_edge_total = is_matched.numel()

    # Per-candidate density scale (edge-length ratio)
    old_el_all = _min_edge_length(vertex_positions, old_indices)  # (T_old,)
    new_el = _min_edge_length(vertex_positions, new_indices)      # (T_new,)
    old_el = old_el_all[cands]                                    # (T_new, 6)
    density_scale = (old_el / new_el.unsqueeze(1).clamp(min=1e-8)).clamp(min=0.1, max=10.0)

    return cands, weights, density_scale


def compute_transfer_weights_bary(
    new_indices: torch.Tensor,       # (T_new, 4) int
    old_indices: torch.Tensor,       # (T_old, 4) int
    old_cc: torch.Tensor,            # (T_old, 3)
    new_cc: torch.Tensor,            # (T_new, 3)
    vertex_positions: torch.Tensor,  # (V, 3)
):
    """For each new tet, compute blending weights across 5 candidate old tets
    using barycentric walk + face-neighbor blending.

    Steps:
    1. Unchanged-tet detection (exact match → skip walk)
    2. Build face adjacency + T_inv for old mesh
    3. Barycentric walk for changed tets only, seeded from matched neighbors
    4. Gather 5 candidates: containing tet + 4 face neighbors
    5. Compute overlap/distance weights

    Returns:
        cands: (T_new, 5) long — candidate old tet indices
        weights: (T_new, 5) float — normalized blending weights
        density_scale: (T_new, 5) float — per-candidate edge-length ratio
    """
    import time
    t0 = time.time()

    device = new_indices.device
    T_new = new_indices.shape[0]
    T_old = old_indices.shape[0]
    K = 5

    # --- Step 1: Unchanged-tet detection (GPU, fast) ---
    old_sorted = old_indices.long().sort(dim=1).values
    new_sorted = new_indices.long().sort(dim=1).values
    all_sorted = torch.cat([old_sorted, new_sorted], dim=0)
    _, inverse = torch.unique(all_sorted, dim=0, sorted=True, return_inverse=True)

    n_unique = inverse.max().item() + 1
    unique_to_old = torch.full((n_unique,), -1, dtype=torch.long, device=device)
    unique_to_old.scatter_(0, inverse[:T_old], torch.arange(T_old, device=device))

    matched_old_tet = unique_to_old[inverse[T_old:]]
    is_tet_matched = matched_old_tet >= 0
    matched_old_tet_safe = matched_old_tet.clamp(min=0)
    n_tet_matched = is_tet_matched.sum().item()

    # Initialize remap: matched tets get their old index, others get 0 (placeholder)
    remap = torch.where(is_tet_matched, matched_old_tet_safe,
                        torch.zeros(T_new, dtype=torch.long, device=device))

    # If all tets matched, skip walk entirely
    n_changed = T_new - n_tet_matched
    t_match = time.time() - t0

    if n_changed > 0:
        # --- Step 2: Build face adjacency + T_inv (CPU/numpy) ---
        idx = old_indices.cpu().numpy().astype(np.int64)
        V = vertex_positions.shape[0]

        face_opp = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        face_keys = []
        face_tets = []
        face_locals = []
        for f in range(4):
            fi = np.sort(idx[:, face_opp[f]], axis=1)
            keys = fi[:, 0].astype(np.int64) * V * V + fi[:, 1].astype(np.int64) * V + fi[:, 2].astype(np.int64)
            face_keys.append(keys)
            face_tets.append(np.arange(T_old))
            face_locals.append(np.full(T_old, f))

        all_keys = np.concatenate(face_keys)
        all_tets_np = np.concatenate(face_tets)
        all_locals = np.concatenate(face_locals)

        order = np.argsort(all_keys)
        sk = all_keys[order]
        st = all_tets_np[order]
        sl = all_locals[order]

        neighbors = np.full((T_old, 4), -1, dtype=np.int64)
        same = sk[:-1] == sk[1:]
        for i in np.where(same)[0]:
            t1, f1 = st[i], sl[i]
            t2, f2 = st[i + 1], sl[i + 1]
            neighbors[t1, f1] = t2
            neighbors[t2, f2] = t1

        verts_np = vertex_positions.detach().cpu().float().numpy()
        v0 = verts_np[idx[:, 0]]
        T_mat = np.stack([
            verts_np[idx[:, 1]] - v0,
            verts_np[idx[:, 2]] - v0,
            verts_np[idx[:, 3]] - v0,
        ], axis=-1)  # (T_old, 3, 3)

        det = np.linalg.det(T_mat)
        degenerate = np.abs(det) < 1e-10
        T_mat_safe = T_mat.copy()
        T_mat_safe[degenerate] = np.eye(3)
        T_inv = np.linalg.inv(T_mat_safe)
        neighbors[degenerate] = -1
        old_v0 = v0

        # --- Step 3: Walk only changed tets ---
        changed_mask = ~is_tet_matched
        changed_idx = torch.where(changed_mask)[0]  # indices into new tets
        N_walk = changed_idx.shape[0]

        new_idx_np = new_indices.cpu().numpy().astype(np.int64)
        changed_idx_np = changed_idx.cpu().numpy()
        walk_centroids = verts_np[new_idx_np[changed_idx_np]].mean(axis=1)  # (N_walk, 3)

        # Seed initialization: for each changed tet, find the best starting tet
        # via its vertices' presence in matched old tets (vertex→tet via old_indices)
        # Simple fast seed: use vertex 0 of each changed new tet to look up
        # any old tet containing that vertex via a quick v2t scatter
        n_verts = vertex_positions.shape[0]
        # Build simple v2t: last-write-wins (just need any tet per vertex)
        v2t_simple = np.zeros(n_verts, dtype=np.int64)
        for vi in range(4):
            v2t_simple[idx[:, vi]] = np.arange(T_old)

        # Seed from vertex 0 of each changed new tet
        seed_verts = new_idx_np[changed_idx_np, 0]
        current_tet = v2t_simple[seed_verts].copy()

        walk_remap = current_tet.copy()
        active = np.ones(N_walk, dtype=bool)
        max_steps = 300
        n_found = 0

        for step_i in range(max_steps):
            if not active.any():
                break
            act_idx = np.where(active)[0]
            # Early out when very few remain (diminishing returns)
            if len(act_idx) < max(100, N_walk // 1000):
                break
            ct = current_tet[act_idx]

            rel = walk_centroids[act_idx] - old_v0[ct]
            bary = np.einsum('nij,nj->ni', T_inv[ct], rel)
            b0 = 1.0 - bary.sum(axis=1)
            all_b = np.column_stack([b0, bary])

            converged = all_b.min(axis=1) >= -1e-4
            if converged.any():
                conv_idx = act_idx[converged]
                walk_remap[conv_idx] = current_tet[conv_idx]
                active[conv_idx] = False
                n_found += converged.sum()

            still = act_idx[~converged]
            if len(still) == 0:
                break
            worst_face = all_b[~converged].argmin(axis=1)
            nb = neighbors[current_tet[still], worst_face]

            at_boundary = nb < 0
            if at_boundary.any():
                boundary_idx = still[at_boundary]
                walk_remap[boundary_idx] = current_tet[boundary_idx]
                active[boundary_idx] = False

            can_walk = still[~at_boundary]
            current_tet[can_walk] = nb[~at_boundary]

        # Handle remaining active queries
        remaining_walk = np.where(active)[0]
        if len(remaining_walk) > 0:
            walk_remap[remaining_walk] = current_tet[remaining_walk]

        # Fallback: nearest old centroid for unconverged queries (GPU)
        n_bad = len(remaining_walk)
        if n_bad > 0:
            old_cents = vertex_positions[old_indices.long()].float().mean(dim=1)
            bad_global = changed_idx_np[remaining_walk]
            bad_cents = torch.as_tensor(walk_centroids[remaining_walk],
                                        dtype=torch.float32, device=device)
            chunk = 512
            cc_chunk = 50_000
            fallback_remap = np.empty(n_bad, dtype=np.int64)
            for s in range(0, n_bad, chunk):
                e = min(s + chunk, n_bad)
                best_d = torch.full((e - s,), float("inf"), device=device)
                best_i = torch.zeros(e - s, dtype=torch.long, device=device)
                for cs in range(0, T_old, cc_chunk):
                    ce = min(cs + cc_chunk, T_old)
                    d = (bad_cents[s:e].unsqueeze(1) - old_cents[cs:ce].unsqueeze(0)).pow(2).sum(-1)
                    min_d, min_i = d.min(dim=1)
                    improved = min_d < best_d
                    best_d[improved] = min_d[improved]
                    best_i[improved] = min_i[improved] + cs
                fallback_remap[s:e] = best_i.cpu().numpy()
            walk_remap[remaining_walk] = fallback_remap

        # Write walk results into remap
        walk_remap_t = torch.as_tensor(walk_remap, dtype=torch.long, device=device).clamp(0, T_old - 1)
        remap[changed_idx] = walk_remap_t
    else:
        n_found = 0
        N_walk = 0
        n_bad = 0
        neighbors = np.full((T_old, 4), -1, dtype=np.int64)
        # Still need face adjacency for candidate gathering
        idx = old_indices.cpu().numpy().astype(np.int64)
        V = vertex_positions.shape[0]
        face_opp = [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]
        face_keys = []
        face_tets = []
        face_locals = []
        for f in range(4):
            fi = np.sort(idx[:, face_opp[f]], axis=1)
            keys = fi[:, 0].astype(np.int64) * V * V + fi[:, 1].astype(np.int64) * V + fi[:, 2].astype(np.int64)
            face_keys.append(keys)
            face_tets.append(np.arange(T_old))
            face_locals.append(np.full(T_old, f))
        all_keys = np.concatenate(face_keys)
        all_tets_np = np.concatenate(face_tets)
        all_locals = np.concatenate(face_locals)
        order = np.argsort(all_keys)
        sk = all_keys[order]
        st = all_tets_np[order]
        sl = all_locals[order]
        same = sk[:-1] == sk[1:]
        for i in np.where(same)[0]:
            t1, f1 = st[i], sl[i]
            t2, f2 = st[i + 1], sl[i + 1]
            neighbors[t1, f1] = t2
            neighbors[t2, f2] = t1

    t_walk = time.time() - t0

    # --- Step 4: Gather 5 candidates (containing tet + 4 face neighbors) ---
    neighbors_t = torch.as_tensor(neighbors, dtype=torch.long, device=device)
    face_nb = neighbors_t[remap]  # (T_new, 4)
    face_nb = torch.where(face_nb >= 0, face_nb, remap.unsqueeze(1).expand(-1, 4))
    cands = torch.cat([remap.unsqueeze(1), face_nb], dim=1)  # (T_new, 5)

    # For unchanged tets: all 5 candidates = matched old tet
    if n_tet_matched > 0:
        cands[is_tet_matched] = matched_old_tet_safe[is_tet_matched].unsqueeze(1).expand(-1, K)

    # --- Step 5: Compute overlap/distance weights ---
    cand_verts = old_indices[cands].long()                    # (T_new, 5, 4)
    new_verts = new_indices.long().unsqueeze(1)               # (T_new, 1, 4)
    matches = (cand_verts.unsqueeze(-1) == new_verts.unsqueeze(2))
    overlap = matches.any(dim=-1).sum(dim=-1).float()         # (T_new, 5)

    cand_cc = old_cc[cands]
    cc_dist_sq = (cand_cc - new_cc.unsqueeze(1)).pow(2).sum(-1)

    overlap_weight = torch.exp(overlap * 2.0)
    dist_weight = 1.0 / (cc_dist_sq + 1e-8)
    raw_weights = overlap_weight * dist_weight
    weights = raw_weights / raw_weights.sum(dim=1, keepdim=True)

    # --- Step 6: Compute density_scale ---
    old_el_all = _min_edge_length(vertex_positions, old_indices)
    new_el = _min_edge_length(vertex_positions, new_indices)
    old_el = old_el_all[cands]
    density_scale = (old_el / new_el.unsqueeze(1).clamp(min=1e-8)).clamp(min=0.1, max=10.0)

    t_total = time.time() - t0
    print(f"  Bary transfer: {n_found}/{N_walk} walked ({n_tet_matched} unchanged), "
          f"fallback {n_bad}, match {t_match:.2f}s walk {t_walk:.2f}s total {t_total:.2f}s")

    return cands, weights, density_scale


# ===========================================================================
# SimpleModel
# ===========================================================================

class SimpleModel(torch.nn.Module):
    """Per-tet parameter model with adjacency-based attribute transfer.

    Like FrozenTetModel but with a working update_triangulation() that
    transfers attributes through retriangulation using adj_grad.
    """

    def __init__(
        self,
        int_vertices: torch.Tensor,
        ext_vertices: torch.Tensor,
        indices: torch.Tensor,
        density: torch.Tensor,
        rgb: torch.Tensor,
        gradient: torch.Tensor,
        sh: torch.Tensor,
        center: torch.Tensor,
        scene_scaling: torch.Tensor | float,
        *,
        max_sh_deg: int = 2,
        chunk_size: int = 408_576,
        density_offset: float = -3,
        **kwargs,
    ) -> None:
        super().__init__()

        # Geometry
        self.interior_vertices = nn.Parameter(int_vertices.cuda(), requires_grad=True)
        self.register_buffer("ext_vertices", ext_vertices.cuda())
        self.register_buffer("indices", indices.int())
        self.empty_indices = torch.empty((0, 4), dtype=indices.dtype, device='cuda')
        self.register_buffer("center", center.reshape(1, 3))
        self.register_buffer("scene_scaling", torch.as_tensor(scene_scaling))

        # Per-tet learnable parameters
        # density stores sigma (log-space); actual density = safe_exp(sigma + density_offset)
        self.density = nn.Parameter(density, requires_grad=True)
        self.gradient = nn.Parameter(gradient, requires_grad=True)
        self.rgb = nn.Parameter(rgb, requires_grad=True)
        self.sh = nn.Parameter(sh.half(), requires_grad=True)

        # Config
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = max_sh_deg
        self.chunk_size = chunk_size
        self.device = self.density.device
        self.sh_dim = ((1 + max_sh_deg) ** 2 - 1) * 3
        self.density_offset = density_offset
        self.register_buffer("_density_offset", torch.tensor(density_offset))

        self.mask_values = False
        self.frozen = False
        self.linear = False
        self.feature_dim = 7
        self.additional_attr = 0

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg + 1)

    @property
    def vertices(self) -> torch.Tensor:
        return torch.cat([self.interior_vertices, self.ext_vertices], dim=0)

    def __len__(self):
        return self.interior_vertices.shape[0] + self.ext_vertices.shape[0]

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

    def compute_batch_features(
        self,
        vertices: torch.Tensor,
        indices: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        circumcenters: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if circumcenters is None:
            circumcenter = pre_calc_cell_values(vertices, indices)
        else:
            circumcenter = circumcenters

        if mask is not None:
            sigma = self.density[mask]
            grd = self.gradient[mask]
            rgb = self.rgb[mask]
            sh = self.sh[mask]
        else:
            sigma = self.density
            grd = self.gradient
            rgb = self.rgb
            sh = self.sh

        # Activate: sigma → density (matching iNGP parameterization)
        density = safe_exp(sigma + self._density_offset)

        sh_dim = (self.max_sh_deg + 1) ** 2 - 1
        attr = torch.empty((density.shape[0], 0), device=grd.device)
        if sh_dim == 0:
            sh_out = torch.empty((density.shape[0], 0, 3), device=grd.device, dtype=sh.dtype)
        else:
            sh_out = sh.reshape(-1, sh_dim, 3)
        return circumcenter, density, rgb, grd, sh_out, attr

    def compute_features(self, offset=False):
        vertices = self.vertices
        indices = self.indices
        circumcenters, density, rgb, grd, sh, attr = self.compute_batch_features(
            vertices, indices
        )
        tets = vertices[indices]
        if offset:
            base_color_v0_raw, normed_grd = offset_normalize(
                rgb, grd, circumcenters, tets
            )
            return circumcenters, density, rgb, normed_grd, sh
        else:
            return circumcenters, density, rgb, grd, sh

    def get_cell_values(
        self,
        camera,
        mask: Optional[torch.Tensor] = None,
        all_circumcenters: Optional[torch.Tensor] = None,
        radii: Optional[torch.Tensor] = None,
    ):
        indices = self.indices[mask] if mask is not None else self.indices
        vertices = self.vertices
        cc, density, rgb, grd, sh, attr = self.compute_batch_features(
            vertices, indices, mask, circumcenters=all_circumcenters
        )
        cell_output = activate_output(
            camera.camera_center.to(self.device),
            density, rgb, grd,
            sh,
            attr,
            indices,
            cc,
            vertices,
            self.current_sh_deg,
            self.max_sh_deg,
        )
        return sh, cell_output

    def compute_adjacency(self):
        vols = tet_volumes(self.vertices[self.indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            self.indices[reverse_mask] = self.indices[reverse_mask][:, [1, 0, 2, 3]]
        self.faces, self.side_index = get_tet_adjacency(self.indices)

    def calc_tet_density(self):
        _, densities, _, _, _, _ = self.compute_batch_features(
            self.vertices, self.indices
        )
        return densities.reshape(-1)

    @torch.no_grad()
    def update_triangulation_bary(self):
        """Recompute Delaunay + transfer parameters via barycentric walk.

        Builds face adjacency from our gdel3d mesh, then uses barycentric
        walks to locate each new tet's centroid in the old mesh.

        Returns (remap, new_indices) where remap[i] = old tet index for new tet i.
        """
        old_indices = self.indices.clone()
        verts = self.vertices
        T_old = old_indices.shape[0]
        device = old_indices.device

        # --- Build face adjacency for old mesh ---
        idx = old_indices.cpu().numpy().astype(np.int64)
        V = verts.shape[0]
        # 4 faces per tet, each face = sorted triple of vertices (opposite one vertex)
        # Face f of tet t is opposite vertex f
        face_opp = [[1,2,3], [0,2,3], [0,1,3], [0,1,2]]
        face_keys = []
        face_tets = []
        face_locals = []
        for f in range(4):
            fi = np.sort(idx[:, face_opp[f]], axis=1)  # (T, 3) sorted
            keys = fi[:, 0] * V * V + fi[:, 1] * V + fi[:, 2]
            face_keys.append(keys)
            face_tets.append(np.arange(T_old))
            face_locals.append(np.full(T_old, f))

        all_keys = np.concatenate(face_keys)      # (4*T,)
        all_tets = np.concatenate(face_tets)       # (4*T,)
        all_locals = np.concatenate(face_locals)   # (4*T,)

        # Sort by key to group faces
        order = np.argsort(all_keys)
        sk = all_keys[order]
        st = all_tets[order]
        sl = all_locals[order]

        # Build neighbor array: neighbor[t, f] = tet across face f, or -1
        neighbors = np.full((T_old, 4), -1, dtype=np.int64)
        # Consecutive pairs with same key are neighbors
        same = sk[:-1] == sk[1:]
        for i in np.where(same)[0]:
            t1, f1 = st[i], sl[i]
            t2, f2 = st[i+1], sl[i+1]
            neighbors[t1, f1] = t2
            neighbors[t2, f2] = t1

        # --- New Delaunay triangulation (gdel3d) ---
        v = Del(verts.shape[0])
        indices_np, prev = v.compute(verts.detach().cpu().double())
        valid_mask = (indices_np >= 0) & (indices_np < verts.shape[0])
        indices_np = indices_np[valid_mask.all(axis=1)]
        del prev

        new_indices = torch.as_tensor(indices_np).cuda()
        vols = tet_volumes(verts[new_indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            new_indices[reverse_mask] = new_indices[reverse_mask][:, [1, 0, 2, 3]]

        T_new = new_indices.shape[0]

        # --- Vectorized barycentric walk to locate new centroids in old mesh ---
        verts_np = verts.detach().cpu().float().numpy()
        new_centroids = verts_np[new_indices.cpu().numpy().astype(np.int64)].mean(axis=1)  # (T_new, 3)

        # Precompute inverse transform for barycentric coords of all old tets
        v0 = verts_np[idx[:, 0]]  # (T_old, 3)
        T_mat = np.stack([
            verts_np[idx[:, 1]] - v0,
            verts_np[idx[:, 2]] - v0,
            verts_np[idx[:, 3]] - v0,
        ], axis=-1)  # (T_old, 3, 3)
        # Handle degenerate tets: use pseudo-inverse and mark bad tets
        det = np.linalg.det(T_mat)
        degenerate = np.abs(det) < 1e-10
        # Replace degenerate matrices with identity to avoid NaN in inv
        T_mat_safe = T_mat.copy()
        T_mat_safe[degenerate] = np.eye(3)
        T_inv = np.linalg.inv(T_mat_safe)  # (T_old, 3, 3)
        # Mark degenerate tets so walk avoids them
        neighbors[degenerate] = -1  # degenerate tets have no valid neighbors
        old_v0 = v0
        n_degenerate = degenerate.sum()
        if n_degenerate > 0:
            print(f"  {n_degenerate} degenerate tets excluded from walk")

        # Initialize: each query starts at a tet (use index % T_old for spread)
        current_tet = np.arange(T_new, dtype=np.int64) % T_old
        remap = current_tet.copy()
        active = np.ones(T_new, dtype=bool)
        max_steps = 500
        n_found = 0

        for step_i in range(max_steps):
            if not active.any():
                break
            act_idx = np.where(active)[0]
            ct = current_tet[act_idx]

            # Compute bary coords: T_inv[ct] @ (p - v0[ct])
            rel = new_centroids[act_idx] - old_v0[ct]  # (N_active, 3)
            # Batched matrix-vector multiply: (N, 3, 3) @ (N, 3) → (N, 3)
            bary = np.einsum('nij,nj->ni', T_inv[ct], rel)  # (N_active, 3)
            b0 = 1.0 - bary.sum(axis=1)  # (N_active,)

            # Check convergence: all bary coords >= -eps
            all_b = np.column_stack([b0, bary])  # (N_active, 4)
            converged = all_b.min(axis=1) >= -1e-4
            if converged.any():
                conv_idx = act_idx[converged]
                remap[conv_idx] = current_tet[conv_idx]
                active[conv_idx] = False
                n_found += converged.sum()

            # Walk unconverged queries toward most negative bary coord
            still = act_idx[~converged]
            if len(still) == 0:
                break
            worst_face = all_b[~converged].argmin(axis=1)  # (N_still,)
            nb = neighbors[current_tet[still], worst_face]  # (N_still,)
            # If neighbor is -1 (boundary), stop walking
            at_boundary = nb < 0
            if at_boundary.any():
                boundary_idx = still[at_boundary]
                remap[boundary_idx] = current_tet[boundary_idx]
                active[boundary_idx] = False
            # Update current tet for those that can walk
            can_walk = still[~at_boundary]
            current_tet[can_walk] = nb[~at_boundary]

        # Handle any remaining active queries (didn't converge) — use last visited
        remaining = np.where(active)[0]
        if len(remaining) > 0:
            remap[remaining] = current_tet[remaining]

        n_outside = T_new - n_found

        # For unfound queries, improve remap via nearest old centroid (GPU)
        unfound_idx = np.where(~np.isin(np.arange(T_new), np.where(~active)[0] if n_found > 0 else np.array([])))[0]
        # Actually: unfound = queries that didn't converge OR hit boundary
        not_converged = np.where(active)[0]  # still active = didn't converge
        # For boundary hits + unconverged, do nearest centroid search on GPU
        n_bad = len(remaining)
        if n_bad > 0:
            remap_t_temp = torch.as_tensor(remap, dtype=torch.long, device=device)
            new_cents = verts[new_indices.long()].float().mean(dim=1)  # (T_new, 3)
            old_cents = verts[old_indices.long()].float().mean(dim=1)  # (T_old, 3)
            bad_idx_t = torch.as_tensor(remaining, dtype=torch.long, device=device)
            bad_cents = new_cents[bad_idx_t]  # (n_bad, 3)
            # Chunked nearest search on GPU
            chunk = 2048
            for s in range(0, n_bad, chunk):
                e = min(s + chunk, n_bad)
                d = (bad_cents[s:e].unsqueeze(1) - old_cents.unsqueeze(0)).pow(2).sum(-1)
                remap[remaining[s:e]] = d.argmin(dim=1).cpu().numpy()
        remap_t = torch.as_tensor(remap, dtype=torch.long, device=device)

        print(f"  Bary transfer: {n_found}/{T_new} located via walk, {n_outside} not found")

        # DEBUG: check for NaN in remap result
        remap_t = torch.as_tensor(remap, dtype=torch.long, device=device)
        bad = (remap_t < 0) | (remap_t >= T_old)
        if bad.any():
            print(f"  WARNING: {bad.sum()} out-of-bounds remap indices! Clamping.")
            remap_t = remap_t.clamp(0, T_old - 1)

        return remap_t, new_indices

    @torch.no_grad()
    def update_triangulation(
        self,
        high_precision=False,
        density_threshold=0.0,
        alpha_threshold=0.0,
    ):
        """Recompute Delaunay triangulation only (geometry).

        Does NOT modify per-tet parameters — the optimizer handles that.

        Returns:
            (cands, weights, density_scale, new_indices, new_cc, old_cc,
             old_indices) or None
        """
        torch.cuda.empty_cache()

        old_indices = self.indices.clone()
        verts = self.vertices

        old_cc, _ = calculate_circumcenters_torch(
            verts[old_indices.long()].double()
        )
        old_cc = old_cc.float()

        # Retriangulate
        if high_precision:
            indices_np = Delaunay(verts.detach().cpu().numpy()).simplices.astype(
                np.int32
            )
        else:
            v = Del(verts.shape[0])
            indices_np, prev = v.compute(verts.detach().cpu().double())
            valid_mask = (indices_np >= 0) & (indices_np < verts.shape[0])
            indices_np = indices_np[valid_mask.all(axis=1)]
            del prev

        # Ensure positive volumes
        new_indices = torch.as_tensor(indices_np).cuda()
        vols = tet_volumes(verts[new_indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            new_indices[reverse_mask] = new_indices[reverse_mask][:, [1, 0, 2, 3]]

        new_cc, _ = calculate_circumcenters_torch(verts[new_indices.long()].double())
        new_cc = new_cc.float()

        # Compute transfer weights via barycentric walk + neighbor blending
        cands, weights, density_scale = compute_transfer_weights_bary(
            new_indices, old_indices, old_cc, new_cc, verts)

        torch.cuda.empty_cache()
        return (cands, weights, density_scale, new_indices, new_cc, old_cc,
                old_indices)

    @torch.no_grad()
    def fast_retriangulate(self):
        """Fast Delaunay retriangulation with direct tet-to-tet copy (no blending).

        For periodic retriangulation where vertices moved slightly:
        - Unchanged tets (same sorted vertices): direct 1:1 parameter copy
        - Changed tets: copy from nearest old tet by circumcenter distance

        Returns (remap, new_indices) where remap[i] = old tet index for new tet i,
        or None on failure.
        """
        old_indices = self.indices.clone()
        verts = self.vertices
        T_old = old_indices.shape[0]
        device = old_indices.device

        # Retriangulate
        v = Del(verts.shape[0])
        indices_np, prev = v.compute(verts.detach().cpu().double())
        valid_mask = (indices_np >= 0) & (indices_np < verts.shape[0])
        indices_np = indices_np[valid_mask.all(axis=1)]
        del prev

        new_indices = torch.as_tensor(indices_np).cuda()
        vols = tet_volumes(verts[new_indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            new_indices[reverse_mask] = new_indices[reverse_mask][:, [1, 0, 2, 3]]

        T_new = new_indices.shape[0]

        # --- Match unchanged tets (same sorted vertex quadruple) ---
        old_sorted = old_indices.long().sort(dim=1).values
        new_sorted = new_indices.long().sort(dim=1).values
        all_sorted = torch.cat([old_sorted, new_sorted], dim=0)
        _, inverse = torch.unique(all_sorted, dim=0, sorted=True, return_inverse=True)

        n_unique = inverse.max().item() + 1
        unique_to_old = torch.full((n_unique,), -1, dtype=torch.long, device=device)
        unique_to_old.scatter_(0, inverse[:T_old], torch.arange(T_old, device=device))

        remap = unique_to_old[inverse[T_old:]]  # (T_new,) — -1 if no match
        is_matched = remap >= 0
        n_matched = is_matched.sum().item()

        # --- For unmatched tets: find nearest old tet by centroid distance ---
        if n_matched < T_new:
            unmatched_idx = torch.where(~is_matched)[0]
            new_centroids = verts[new_indices[unmatched_idx].long()].float().mean(dim=1)
            old_centroids = verts[old_indices.long()].float().mean(dim=1)

            # Chunked nearest-neighbor search (small chunks to avoid OOM)
            chunk = 512
            cc_chunk = 50_000
            for s in range(0, unmatched_idx.shape[0], chunk):
                e = min(s + chunk, unmatched_idx.shape[0])
                best_d = torch.full((e - s,), float("inf"), device=device)
                best_i = torch.zeros(e - s, dtype=torch.long, device=device)
                for cs in range(0, T_old, cc_chunk):
                    ce = min(cs + cc_chunk, T_old)
                    d = (new_centroids[s:e].unsqueeze(1) - old_centroids[cs:ce].unsqueeze(0)).pow(2).sum(-1)
                    min_d, min_i = d.min(dim=1)
                    improved = min_d < best_d
                    best_d[improved] = min_d[improved]
                    best_i[improved] = min_i[improved] + cs
                remap[unmatched_idx[s:e]] = best_i

        return remap, new_indices, n_matched, T_new, old_indices

    @staticmethod
    def init_from_pcd(
        point_cloud,
        cameras,
        device,
        max_sh_deg=2,
        voxel_size=0.00,
        density_offset=-3,
        **kwargs,
    ):
        torch.manual_seed(2)

        ccenters = torch.stack(
            [c.camera_center.reshape(3) for c in cameras], dim=0
        ).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(
            ccenters - center.reshape(1, 3), dim=1, ord=torch.inf
        ).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(vertices.numpy())
        if voxel_size > 0:
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

        vertices = torch.as_tensor(np.asarray(o3d_pcd.points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        pcd_scaling = torch.linalg.norm(
            vertices - center.cpu().reshape(1, 3), dim=1, ord=2
        ).max()
        new_radius = pcd_scaling.cpu().item()

        num_ext = 1000
        ext_vertices = (
            fibonacci_spiral_on_sphere(num_ext, new_radius, device="cpu")
            + center.reshape(1, 3).cpu()
        )

        # Concatenate exterior into interior (same as Model.init_from_pcd)
        vertices = torch.cat([vertices, ext_vertices], dim=0)
        ext_vertices = torch.empty((0, 3))

        int_vertices = vertices.to(device)
        ext_verts = ext_vertices.to(device)

        # Initial Delaunay
        all_verts = torch.cat([int_vertices, ext_verts], dim=0)
        v = Del(all_verts.shape[0])
        indices_np, prev = v.compute(all_verts.detach().cpu().double())
        valid_mask = (indices_np >= 0) & (indices_np < all_verts.shape[0])
        indices_np = indices_np[valid_mask.all(axis=1)]
        del prev
        indices = torch.as_tensor(indices_np).to(device)
        vols = tet_volumes(all_verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        T = indices.shape[0]
        sh_dim = ((1 + max_sh_deg) ** 2 - 1) * 3

        # Initialize per-tet parameters (density stores sigma; actual density = exp(sigma + offset))
        density = torch.zeros((T, 1), device=device)  # sigma=0 → density=exp(offset)
        rgb = torch.full((T, 3), 0.5, device=device)
        gradient = torch.zeros((T, 1, 3), device=device)
        sh = torch.zeros((T, sh_dim // 3, 3), device=device)

        model = SimpleModel(
            int_vertices=int_vertices,
            ext_vertices=ext_verts,
            indices=indices,
            density=density,
            rgb=rgb,
            gradient=gradient,
            sh=sh,
            center=center,
            scene_scaling=scaling,
            max_sh_deg=max_sh_deg,
            density_offset=density_offset,
        )
        return model

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)

        int_vertices = ckpt["interior_vertices"]
        ext_vertices = ckpt["ext_vertices"]
        indices = ckpt["indices"]
        if "empty_indices" in ckpt:
            del ckpt["empty_indices"]

        density = ckpt["density"]
        rgb = ckpt["rgb"]
        gradient = ckpt["gradient"]
        sh = ckpt["sh"]
        center = ckpt["center"]
        scene_scaling = ckpt["scene_scaling"]

        model = SimpleModel(
            int_vertices=int_vertices.to(device),
            ext_vertices=ext_vertices.to(device),
            indices=indices.to(device),
            density=density.to(device),
            rgb=rgb.to(device),
            gradient=gradient.to(device),
            sh=sh.to(device),
            center=center.to(device),
            scene_scaling=scene_scaling.to(device),
            max_sh_deg=config.max_sh_deg,
        )
        model.load_state_dict(ckpt)
        model.min_t = config.min_t
        return model

    @torch.no_grad
    def save2ply(self, path):
        path.parent.mkdir(exist_ok=True, parents=True)

        xyz = self.vertices.detach().cpu().numpy().astype(np.float32)  # shape (num_vertices, 3)

        vertex_dict = {
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }

        N = self.indices.shape[0]
        sh_dim = ((self.max_sh_deg+1)**2-1)

        circumcenters, density, base_color_v0_raw, normed_grd, sh = self.compute_features(offset=True)

        base_color_v0_raw = base_color_v0_raw.cpu().numpy().astype(np.float32)
        grds = normed_grd.reshape(-1, 3).cpu().numpy().astype(np.float32)
        densities = density.reshape(-1).cpu().numpy().astype(np.float32)
        if sh_dim > 0:
            sh_coeffs = sh.reshape(-1, sh_dim, 3).cpu().numpy().astype(np.float32)
        else:
            sh_coeffs = np.empty((N, 0, 3), dtype=np.float32)

        tetra_dict = {}
        tetra_dict["indices"] = self.indices.cpu().numpy().astype(np.int32)
        # tetra_dict["mask"] = self.mask.cpu().numpy().astype(np.uint8)
        tetra_dict["s"] = np.ascontiguousarray(densities)
        for i, co in enumerate(["x", "y", "z"]):
            tetra_dict[f"grd_{co}"]         = np.ascontiguousarray(grds[:, i])

        sh_0 = RGB2SH(base_color_v0_raw)
        tetra_dict[f"sh_0_r"] = np.ascontiguousarray(sh_0[:, 0])
        tetra_dict[f"sh_0_g"] = np.ascontiguousarray(sh_0[:, 1])
        tetra_dict[f"sh_0_b"] = np.ascontiguousarray(sh_0[:, 2])
        for i in range(sh_coeffs.shape[1]):
            tetra_dict[f"sh_{i+1}_r"] = np.ascontiguousarray(sh_coeffs[:, i, 0])
            tetra_dict[f"sh_{i+1}_g"] = np.ascontiguousarray(sh_coeffs[:, i, 1])
            tetra_dict[f"sh_{i+1}_b"] = np.ascontiguousarray(sh_coeffs[:, i, 2])


        data_dict = {
            "vertex": vertex_dict,
            "tetrahedron": tetra_dict,
        }

        tinyplypy.write_ply(str(path), data_dict, is_binary=True)


# ===========================================================================
# SimpleOptimizer
# ===========================================================================

class SimpleOptimizer:
    """Optimizer for SimpleModel. Mirrors FrozenTetOptimizer but with
    working update_triangulation that remaps optimizer state."""

    def __init__(
        self,
        model: SimpleModel,
        *,
        freeze_lr: float = 1e-3,
        final_freeze_lr: float = 1e-4,
        lr_delay_multi=1e-8,
        lr_delay=0,
        vertices_lr: float = 4e-4,
        final_vertices_lr: float = 4e-7,
        vert_lr_delay: int = 500,
        vertices_lr_delay_multi: float = 0.01,
        freeze_start: int = 15000,
        iterations: int = 30000,
        spike_duration: int = 20,
        densify_interval: int = 500,
        densify_end: int = 15000,
        densify_start: int = 2000,
        split_std: float = 0.5,
        **kwargs,
    ) -> None:
        self.model = model
        self.split_std = split_std

        self.optim = optim.CustomAdam([
            {"params": [model.density], "lr": freeze_lr * 1.5, "name": "density"},
            {"params": [model.rgb], "lr": freeze_lr, "name": "color"},
            {"params": [model.gradient], "lr": freeze_lr, "name": "gradient"},
        ], eps=1e-15)
        self.sh_optim = optim.CustomAdam([
            {"params": [model.sh], "lr": freeze_lr, "name": "sh"},
        ], eps=1e-15)
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {
                "params": [model.interior_vertices],
                "lr": self.vert_lr_multi * vertices_lr,
                "name": "interior_vertices",
            },
        ])

        self.freeze_start = freeze_start
        self.scheduler = get_expon_lr_func(
            lr_init=freeze_lr,
            lr_final=final_freeze_lr,
            lr_delay_mult=lr_delay_multi,
            max_steps=iterations,
            lr_delay_steps=lr_delay,
        )

        self.vertex_lr = self.vert_lr_multi * vertices_lr
        self.vertex_scheduler_args = get_expon_lr_func(
            lr_init=self.vertex_lr,
            lr_final=self.vert_lr_multi * final_vertices_lr,
            lr_delay_mult=vertices_lr_delay_multi,
            max_steps=iterations,
            lr_delay_steps=vert_lr_delay,
        )

        # Alias for compatibility
        self.net_optim = self.optim

    # --- optimizer steps ---
    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def main_step(self):
        self.step()

    def main_zero_grad(self):
        self.zero_grad()

    def update_learning_rate(self, iteration):
        for param_group in self.optim.param_groups:
            lr = self.scheduler(iteration)
            param_group["lr"] = lr

        for param_group in self.sh_optim.param_groups:
            lr = self.scheduler(iteration)
            param_group["lr"] = lr

        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "interior_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertex_lr = lr
                param_group["lr"] = lr

    def regularizer(self, render_pkg, lambda_weight_decay=0, **kwargs):
        return 0.0

    def _rebuild_optim(self):
        """Rebuild per-tet optimizers from current model parameters.
        This fully resets Adam state (step counter + momentum)."""
        lr = self.optim.param_groups[0]["lr"]
        sh_lr = self.sh_optim.param_groups[0]["lr"]
        self.optim = optim.CustomAdam([
            {"params": [self.model.density], "lr": lr, "name": "density"},
            {"params": [self.model.rgb], "lr": lr, "name": "color"},
            {"params": [self.model.gradient], "lr": lr, "name": "gradient"},
        ], eps=1e-15)
        self.sh_optim = optim.CustomAdam([
            {"params": [self.model.sh], "lr": sh_lr, "name": "sh"},
        ], eps=1e-15)
        self.net_optim = self.optim

    @staticmethod
    def _save_adam_state(custom_adam):
        """Save Adam state (exp_avg, exp_avg_sq, step) for all param groups."""
        saved = {}
        for group in custom_adam.param_groups:
            s = custom_adam.optimizer.state.get(group['params'][0])
            if s and 'exp_avg' in s:
                saved[group['name']] = {
                    'exp_avg': s['exp_avg'].clone(),
                    'exp_avg_sq': s['exp_avg_sq'].clone(),
                    'step': s['step'].clone(),
                }
        return saved

    @staticmethod
    def _blend_old_values(old_tensor, cands, weights):
        """Weighted blend of old tensor values across candidates.
        old_tensor: (T_old, ...), cands: (T_new, K), weights: (T_new, K)
        Returns: (T_new, ...)
        """
        # Gather candidates: (T_new, 4, ...)
        gathered = old_tensor[cands]
        # Expand weights to match tensor dims
        w = weights
        for _ in range(old_tensor.dim() - 1):
            w = w.unsqueeze(-1)
        return (gathered * w).sum(dim=1)

    @staticmethod
    def _restore_adam_state_blended(custom_adam, saved, cands, weights, cull_mask=None):
        """Restore Adam state using weighted blend across candidates."""
        for group in custom_adam.param_groups:
            name = group['name']
            if name not in saved:
                continue
            param = group['params'][0]
            old = saved[name]

            # Weighted blend of momentum across candidates
            w = weights
            ea = old['exp_avg']
            ea_sq = old['exp_avg_sq']

            # Expand weights for broadcasting
            w_exp = w
            for _ in range(ea.dim() - 1):
                w_exp = w_exp.unsqueeze(-1)

            new_exp_avg = (ea[cands] * w_exp).sum(dim=1)
            new_exp_avg_sq = (ea_sq[cands] * w_exp).sum(dim=1)

            if cull_mask is not None:
                new_exp_avg = new_exp_avg[cull_mask]
                new_exp_avg_sq = new_exp_avg_sq[cull_mask]

            custom_adam.optimizer.state[param] = {
                'step': old['step'].clone(),
                'exp_avg': new_exp_avg,
                'exp_avg_sq': new_exp_avg_sq,
            }

    # --- barycentric walk transfer (no blending, direct copy) ---
    def bary_update_triangulation(self, density_threshold=0.0, alpha_threshold=0.0, **kwargs):
        """Retriangulate + transfer via barycentric walk. Direct 1:1 copy, no blending."""
        # Save Adam state before transfer
        old_optim_state = self._save_adam_state(self.optim)
        old_sh_state = self._save_adam_state(self.sh_optim)

        result = self.model.update_triangulation_bary()
        if result is None:
            return
        remap, new_indices = result
        old_indices = self.model.indices

        # Copy parameters with density_scale correction for sigma
        old_el = _min_edge_length(self.model.vertices, old_indices)   # (T_old,)
        new_el = _min_edge_length(self.model.vertices, new_indices)   # (T_new,)
        density_scale = (old_el[remap] / new_el.clamp(min=1e-8)).clamp(min=0.1, max=10.0)
        log_ds = torch.log(density_scale).unsqueeze(-1)  # (T_new, 1)

        new_sigma = self.model.density.data[remap] + log_ds
        if torch.isnan(new_sigma).any():
            print(f"  WARNING: NaN in new_sigma! density NaN: {torch.isnan(self.model.density.data[remap]).any()}, log_ds NaN: {torch.isnan(log_ds).any()}")
        if torch.isnan(self.model.rgb.data[remap]).any():
            print(f"  WARNING: NaN in remapped rgb!")
        self.model.density = nn.Parameter(new_sigma.contiguous().requires_grad_(True))
        self.model.rgb = nn.Parameter(self.model.rgb.data[remap].contiguous().requires_grad_(True))
        self.model.gradient = nn.Parameter(self.model.gradient.data[remap].contiguous().requires_grad_(True))
        self.model.sh = nn.Parameter(self.model.sh.data[remap].contiguous().requires_grad_(True))
        self.model.indices = new_indices.int()

        # Cull low-density tets
        cull_mask = None
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.model.calc_tet_density()
            cull_mask = tet_density > density_threshold
            self.model.empty_indices = self.model.indices[~cull_mask]
            self.model.indices = self.model.indices[cull_mask]
            self.model.density = nn.Parameter(self.model.density.data[cull_mask].contiguous().requires_grad_(True))
            self.model.rgb = nn.Parameter(self.model.rgb.data[cull_mask].contiguous().requires_grad_(True))
            self.model.gradient = nn.Parameter(self.model.gradient.data[cull_mask].contiguous().requires_grad_(True))
            self.model.sh = nn.Parameter(self.model.sh.data[cull_mask].contiguous().requires_grad_(True))
        else:
            self.model.empty_indices = torch.empty((0, 4), dtype=self.model.indices.dtype, device="cuda")

        # Rebuild optimizers properly then restore remapped Adam state
        self._rebuild_optim()

        # Remap Adam state: index old state by remap (+ optional cull)
        def restore_remapped(custom_adam, saved, remap_idx, cull=None):
            for group in custom_adam.param_groups:
                name = group['name']
                if name not in saved:
                    continue
                param = group['params'][0]
                old = saved[name]
                new_ea = old['exp_avg'][remap_idx]
                new_ea_sq = old['exp_avg_sq'][remap_idx]
                if cull is not None:
                    new_ea = new_ea[cull]
                    new_ea_sq = new_ea_sq[cull]
                custom_adam.optimizer.state[param] = {
                    'step': old['step'].clone(),
                    'exp_avg': new_ea,
                    'exp_avg_sq': new_ea_sq,
                }

        if old_optim_state:
            restore_remapped(self.optim, old_optim_state, remap, cull_mask)
        if old_sh_state:
            restore_remapped(self.sh_optim, old_sh_state, remap, cull_mask)

        self.model.device = self.model.density.device
        torch.cuda.empty_cache()

    # --- fast retriangulation (direct copy, no blending) ---
    def fast_update_triangulation(self):
        """Fast path: direct tet-to-tet copy with no blending or optimizer rebuild.
        Returns False if match rate is too low (caller should use full path)."""
        result = self.model.fast_retriangulate()
        if result is None:
            return True
        remap, new_indices, n_matched, T_new, old_indices = result

        # Fall back to full path if too many tets changed
        match_rate = n_matched / T_new if T_new > 0 else 1.0
        if match_rate < 0.5:
            print(f"  Fast Delaunay: match rate {match_rate:.1%} too low, using full path")
            # Restore old indices and use full path
            self.model.indices = old_indices.int()
            return False

        # Direct index copy of all parameters
        self.model.density = nn.Parameter(self.model.density.data[remap].contiguous().requires_grad_(True))
        self.model.rgb = nn.Parameter(self.model.rgb.data[remap].contiguous().requires_grad_(True))
        self.model.gradient = nn.Parameter(self.model.gradient.data[remap].contiguous().requires_grad_(True))
        self.model.sh = nn.Parameter(self.model.sh.data[remap].contiguous().requires_grad_(True))
        self.model.indices = new_indices.int()
        self.model.empty_indices = torch.empty((0, 4), dtype=self.model.indices.dtype, device="cuda")

        # Remap Adam state by direct indexing (no blending)
        def remap_adam(custom_adam, remap_idx):
            for group in custom_adam.param_groups:
                s = custom_adam.optimizer.state.get(group['params'][0])
                old_step = s['step'].clone() if s and 'step' in s else None
                old_ea = s['exp_avg'][remap_idx] if s and 'exp_avg' in s else None
                old_ea_sq = s['exp_avg_sq'][remap_idx] if s and 'exp_avg_sq' in s else None

                if group['params'][0] in custom_adam.optimizer.state:
                    del custom_adam.optimizer.state[group['params'][0]]

                # Update param reference to new tensor
                if group['name'] == 'density':
                    group['params'][0] = self.model.density
                elif group['name'] == 'color':
                    group['params'][0] = self.model.rgb
                elif group['name'] == 'gradient':
                    group['params'][0] = self.model.gradient
                elif group['name'] == 'sh':
                    group['params'][0] = self.model.sh

                if old_ea is not None:
                    custom_adam.optimizer.state[group['params'][0]] = {
                        'step': old_step,
                        'exp_avg': old_ea,
                        'exp_avg_sq': old_ea_sq,
                    }

        remap_adam(self.optim, remap)
        remap_adam(self.sh_optim, remap)
        self.net_optim = self.optim
        self.model.device = self.model.density.device
        print(f"  Fast Delaunay: {n_matched}/{T_new} matched ({match_rate:.1%})")
        return True

    # --- triangulation update with optimizer state transfer ---
    def update_triangulation(self, density_threshold=0.0, alpha_threshold=0.0, **kwargs):
        # Save Adam state before retriangulation
        old_optim_state = self._save_adam_state(self.optim)
        old_sh_state = self._save_adam_state(self.sh_optim)

        result = self.model.update_triangulation(
            density_threshold=density_threshold,
            alpha_threshold=alpha_threshold,
            **kwargs,
        )
        if result is None:
            return
        (cands, weights, density_scale, new_indices, new_cc, old_cc,
         old_indices) = result

        # Weighted blend of parameter values across 5 candidates
        # density stores sigma (log-space); blend sigma + additive edge-length correction
        old_sigma = self.model.density.data  # (T_old, 1) — sigma values
        log_density_scale = torch.log(density_scale.unsqueeze(-1).clamp(min=1e-8))  # (T_new, 5, 1)
        scaled_sigma = old_sigma[cands] + log_density_scale
        w = weights.unsqueeze(-1)
        new_sigma = (scaled_sigma * w).sum(dim=1)

        new_rgb = self._blend_old_values(self.model.rgb.data, cands, weights)
        new_gradient = self._blend_old_values(self.model.gradient.data, cands, weights)
        new_sh = self._blend_old_values(self.model.sh.data, cands, weights)

        # Skip edge-local conservation — rely on weighted blending alone

        # Override orphan tets (containing new vertices not in old mesh)
        # by copying from the nearest old tet by centroid distance.
        n_old = old_indices.max().item() + 1
        orphan_tets = (new_indices >= n_old).any(dim=1)
        if orphan_tets.any():
            verts = self.model.vertices
            device = verts.device
            orphan_idx = torch.where(orphan_tets)[0]
            orphan_centroids = verts[new_indices[orphan_idx].long()].float().mean(dim=1)
            old_centroids = verts[old_indices.long()].float().mean(dim=1)

            n_orphan = orphan_idx.shape[0]
            T_old = old_indices.shape[0]
            nearest = torch.zeros(n_orphan, dtype=torch.long, device=device)
            chunk = 512
            cc_chunk = 50_000
            for s in range(0, n_orphan, chunk):
                e = min(s + chunk, n_orphan)
                best_d = torch.full((e - s,), float("inf"), device=device)
                for cs in range(0, T_old, cc_chunk):
                    ce = min(cs + cc_chunk, T_old)
                    d = (orphan_centroids[s:e].unsqueeze(1) - old_centroids[cs:ce].unsqueeze(0)).pow(2).sum(-1)
                    min_d, min_i = d.min(dim=1)
                    improved = min_d < best_d
                    best_d[improved] = min_d[improved]
                    nearest[s:e][improved] = min_i[improved] + cs

            new_sigma[orphan_idx] = self.model.density.data[nearest]
            new_rgb[orphan_idx] = self.model.rgb.data[nearest]
            new_gradient[orphan_idx] = self.model.gradient.data[nearest]
            new_sh[orphan_idx] = self.model.sh.data[nearest]

        self.model.density = nn.Parameter(new_sigma.contiguous().requires_grad_(True))
        self.model.rgb = nn.Parameter(new_rgb.contiguous().requires_grad_(True))
        self.model.gradient = nn.Parameter(new_gradient.contiguous().requires_grad_(True))
        self.model.sh = nn.Parameter(new_sh.contiguous().requires_grad_(True))

        # Update model indices
        self.model.indices = new_indices.int()

        # Cull low-density tets
        cull_mask = None
        if density_threshold > 0 or alpha_threshold > 0:
            tet_density = self.model.calc_tet_density()
            cull_mask = tet_density > density_threshold
            if alpha_threshold > 0 and hasattr(self.model, 'calc_tet_alpha'):
                tet_alpha = self.model.calc_tet_alpha(mode="min", density=tet_density)
                cull_mask = cull_mask | (tet_alpha > alpha_threshold)
            self.model.empty_indices = self.model.indices[~cull_mask]
            self.model.indices = self.model.indices[cull_mask]
            self.model.density = nn.Parameter(self.model.density.data[cull_mask].contiguous().requires_grad_(True))
            self.model.rgb = nn.Parameter(self.model.rgb.data[cull_mask].contiguous().requires_grad_(True))
            self.model.gradient = nn.Parameter(self.model.gradient.data[cull_mask].contiguous().requires_grad_(True))
            self.model.sh = nn.Parameter(self.model.sh.data[cull_mask].contiguous().requires_grad_(True))
        else:
            self.model.empty_indices = torch.empty(
                (0, 4), dtype=self.model.indices.dtype, device="cuda"
            )

        # Rebuild optimizers then restore blended Adam state
        self._rebuild_optim()
        if old_optim_state:
            self._restore_adam_state_blended(
                self.optim, old_optim_state, cands, weights, cull_mask)
        if old_sh_state:
            self._restore_adam_state_blended(
                self.sh_optim, old_sh_state, cands, weights, cull_mask)

        self.model.device = self.model.density.device
        torch.cuda.empty_cache()

    # --- densification ---
    def add_points(self, new_verts: torch.Tensor, raw_verts=False):
        self.model.interior_vertices = self.vertex_optim.cat_tensors_to_optimizer(
            dict(interior_vertices=new_verts)
        )["interior_vertices"]
        # Retriangulate and transfer per-tet state via optimizer
        self.update_triangulation()

    def remove_points(self, keep_mask: torch.Tensor):
        keep_mask = keep_mask[: self.model.interior_vertices.shape[0]]
        self.model.interior_vertices = self.vertex_optim.prune_optimizer(keep_mask)[
            "interior_vertices"
        ]
        self.update_triangulation()

    @torch.no_grad()
    def split(self, split_point, **kwargs):
        self.add_points(split_point)

    @staticmethod
    def _restore_adam_state_split(custom_adam, saved, keep_mask, split_idx):
        """Restore Adam state for 1-to-4 split: keep non-split, repeat split 4x."""
        for group in custom_adam.param_groups:
            name = group['name']
            if name not in saved:
                continue
            param = group['params'][0]
            old = saved[name]

            ea_kept = old['exp_avg'][keep_mask]
            ea_split = old['exp_avg'][split_idx].repeat_interleave(4, dim=0)

            ea_sq_kept = old['exp_avg_sq'][keep_mask]
            ea_sq_split = old['exp_avg_sq'][split_idx].repeat_interleave(4, dim=0)

            custom_adam.optimizer.state[param] = {
                'step': old['step'].clone(),
                'exp_avg': torch.cat([ea_kept, ea_split], dim=0),
                'exp_avg_sq': torch.cat([ea_sq_kept, ea_sq_split], dim=0),
            }

    @torch.no_grad()
    def split_tets_inplace(self, split_mask):
        """1-to-4 split of selected tets at their centroids.
        Sub-tets get exact copies of parent parameters.
        No retriangulation — the next periodic delaunay_interval handles that."""
        model = self.model
        split_idx = torch.where(split_mask)[0]
        K = split_idx.shape[0]
        if K == 0:
            return

        # 1. Compute centroids of tets to split
        split_tet_verts = model.indices[split_idx].long()  # (K, 4)
        verts = model.vertices
        V_old = verts.shape[0]
        centroids = verts[split_tet_verts].float().mean(dim=1)  # (K, 3)

        # 2. Add centroid vertices to interior_vertices (extends optimizer)
        model.interior_vertices = self.vertex_optim.cat_tensors_to_optimizer(
            dict(interior_vertices=centroids)
        )["interior_vertices"]

        centroid_idx = torch.arange(V_old, V_old + K, device=model.device)

        # 3. Build new indices: keep non-split tets, replace each split tet with 4 sub-tets
        keep_mask = ~split_mask
        kept_indices = model.indices[keep_mask]  # (T-K, 4)

        v0 = split_tet_verts[:, 0]
        v1 = split_tet_verts[:, 1]
        v2 = split_tet_verts[:, 2]
        v3 = split_tet_verts[:, 3]
        c = centroid_idx

        sub_tets = torch.stack([
            torch.stack([c, v1, v2, v3], dim=1),
            torch.stack([v0, c, v2, v3], dim=1),
            torch.stack([v0, v1, c, v3], dim=1),
            torch.stack([v0, v1, v2, c], dim=1),
        ], dim=1).reshape(-1, 4)  # (4*K, 4)

        new_indices = torch.cat([kept_indices, sub_tets], dim=0).int()

        # 4. Build new per-tet parameters: keep non-split, repeat split 4x
        old_optim_state = self._save_adam_state(self.optim)
        old_sh_state = self._save_adam_state(self.sh_optim)

        def keep_and_repeat(tensor):
            return torch.cat([tensor[keep_mask], tensor[split_idx].repeat_interleave(4, dim=0)], dim=0)

        model.density = nn.Parameter(keep_and_repeat(model.density.data).contiguous().requires_grad_(True))
        model.rgb = nn.Parameter(keep_and_repeat(model.rgb.data).contiguous().requires_grad_(True))
        model.gradient = nn.Parameter(keep_and_repeat(model.gradient.data).contiguous().requires_grad_(True))
        model.sh = nn.Parameter(keep_and_repeat(model.sh.data).contiguous().requires_grad_(True))
        model.indices = new_indices

        # 5. Rebuild optimizers and restore state with same keep/repeat pattern
        self._rebuild_optim()
        if old_optim_state:
            self._restore_adam_state_split(self.optim, old_optim_state, keep_mask, split_idx)
        if old_sh_state:
            self._restore_adam_state_split(self.sh_optim, old_sh_state, keep_mask, split_idx)

        model.device = model.density.device
        torch.cuda.empty_cache()

        # Store metadata for undo — centroids are at end of vertex array,
        # siblings are at end of tet array
        n_kept = kept_indices.shape[0]
        self._split_info = {
            'centroid_vert_idx': centroid_idx.clone(),   # (K,) vertex indices
            'parent_vert_idx': split_tet_verts.clone(),  # (K, 4) original tet vertices
            'sibling_start': n_kept,                     # start of siblings in tet array
            'K': K,
            'V_old': V_old,                              # vertex count before adding centroids
        }

        print(f"Split {K} tets in-place (1-to-4): "
              f"#T: {new_indices.shape[0]}, #V: {model.vertices.shape[0]}")

    @torch.no_grad()
    def undo_useless_splits(self, threshold=0.5):
        """Check sibling groups from last split. Undo splits where parameters
        haven't diverged — remove centroid vertex, restore parent tet."""
        if not hasattr(self, '_split_info') or self._split_info is None:
            return 0

        info = self._split_info
        K = info['K']
        sib_start = info['sibling_start']
        centroid_vi = info['centroid_vert_idx']
        parent_vi = info['parent_vert_idx']
        V_old = info['V_old']
        model = self.model

        # Check divergence for each group of 4 siblings
        undo = torch.zeros(K, dtype=torch.bool, device=model.device)
        for i in range(K):
            s = sib_start + 4 * i
            if s + 4 > model.density.shape[0]:
                continue
            sib_sigma = model.density.data[s:s+4].squeeze()
            if sib_sigma.var() < threshold:
                undo[i] = True

        n_undo = undo.sum().item()
        if n_undo == 0:
            self._split_info = None
            return 0

        # Build tet keep mask: keep all non-sibling tets + kept (diverged) siblings
        n_tets = model.indices.shape[0]
        tet_keep = torch.ones(n_tets, dtype=torch.bool, device=model.device)
        for i in range(K):
            if undo[i]:
                s = sib_start + 4 * i
                tet_keep[s:s+4] = False

        # Build vertex keep mask: remove undone centroids
        n_int = model.interior_vertices.shape[0]
        vert_keep = torch.ones(n_int, dtype=torch.bool, device=model.device)
        undo_centroid_vi = centroid_vi[undo]
        vert_keep[undo_centroid_vi[undo_centroid_vi < n_int]] = False

        # Build vertex remap (old index → new index after removal)
        # Centroids are at end, so original vertices (< V_old) are unaffected
        n_ext = model.ext_vertices.shape[0]
        full_keep = torch.cat([vert_keep, torch.ones(n_ext, dtype=torch.bool, device=model.device)])
        remap = torch.cumsum(full_keep.int(), dim=0) - 1

        # Remap surviving tet indices
        kept_indices = remap[model.indices[tet_keep].long()].int()

        # Build restored parent tets (vertex indices < V_old, unaffected by remap)
        parent_tets = parent_vi[undo]  # (n_undo, 4)

        # Parent params: average of 4 siblings
        parent_density = []
        parent_rgb = []
        parent_gradient = []
        parent_sh = []
        for i in range(K):
            if undo[i]:
                s = sib_start + 4 * i
                parent_density.append(model.density.data[s:s+4].mean(dim=0))
                parent_rgb.append(model.rgb.data[s:s+4].mean(dim=0))
                parent_gradient.append(model.gradient.data[s:s+4].mean(dim=0))
                parent_sh.append(model.sh.data[s:s+4].mean(dim=0))

        new_indices = torch.cat([kept_indices, parent_tets.int()], dim=0)

        new_density = torch.cat([model.density.data[tet_keep], torch.stack(parent_density)], dim=0)
        new_rgb = torch.cat([model.rgb.data[tet_keep], torch.stack(parent_rgb)], dim=0)
        new_gradient = torch.cat([model.gradient.data[tet_keep], torch.stack(parent_gradient)], dim=0)
        new_sh = torch.cat([model.sh.data[tet_keep], torch.stack(parent_sh)], dim=0)

        model.density = nn.Parameter(new_density.contiguous().requires_grad_(True))
        model.rgb = nn.Parameter(new_rgb.contiguous().requires_grad_(True))
        model.gradient = nn.Parameter(new_gradient.contiguous().requires_grad_(True))
        model.sh = nn.Parameter(new_sh.contiguous().requires_grad_(True))
        model.indices = new_indices

        # Prune vertices from optimizer
        model.interior_vertices = self.vertex_optim.prune_optimizer(vert_keep)["interior_vertices"]

        # Rebuild per-tet optimizers (fresh state)
        self._rebuild_optim()
        model.device = model.density.device
        self._split_info = None
        torch.cuda.empty_cache()

        print(f"Undid {n_undo}/{K} splits, removed {n_undo} vertices "
              f"(#V: {model.vertices.shape[0]}, #T: {model.indices.shape[0]})")
        return n_undo

    def clip_grad_norm_(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self.model.density, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.rgb, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.gradient, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.sh, max_norm)


# ===========================================================================
# VertexModel — per-vertex attributes with 2-sample quadrature
# ===========================================================================

class VertexModel(torch.nn.Module):
    """Per-vertex parameter model. Retriangulation just swaps the index buffer."""

    def __init__(
        self,
        int_vertices: torch.Tensor,
        ext_vertices: torch.Tensor,
        indices: torch.Tensor,
        sigma: torch.Tensor,
        rgb: torch.Tensor,
        sh: torch.Tensor,
        center: torch.Tensor,
        scene_scaling: torch.Tensor | float,
        *,
        max_sh_deg: int = 2,
        density_offset: float = -3,
        **kwargs,
    ) -> None:
        super().__init__()

        # Geometry
        self.interior_vertices = nn.Parameter(int_vertices.cuda(), requires_grad=True)
        self.register_buffer("ext_vertices", ext_vertices.cuda())
        self.register_buffer("indices", indices.int())
        self.empty_indices = torch.empty((0, 4), dtype=indices.dtype, device='cuda')
        self.register_buffer("center", center.reshape(1, 3))
        self.register_buffer("scene_scaling", torch.as_tensor(scene_scaling))

        # Per-vertex learnable parameters
        self.sigma = nn.Parameter(sigma, requires_grad=True)       # (V, 1) log-density
        self.rgb = nn.Parameter(rgb, requires_grad=True)           # (V, 3) DC color
        self.sh = nn.Parameter(sh.half(), requires_grad=True)      # (V, sh_dim//3, 3)

        # Config
        self.max_sh_deg = max_sh_deg
        self.current_sh_deg = max_sh_deg
        self.device = self.sigma.device
        self.sh_dim = ((1 + max_sh_deg) ** 2 - 1) * 3
        self.density_offset = density_offset
        self.register_buffer("_density_offset", torch.tensor(density_offset))

        self.mask_values = False
        self.feature_dim = 4   # sigma + rgb
        self.additional_attr = 0

    def sh_up(self):
        self.current_sh_deg = min(self.max_sh_deg, self.current_sh_deg + 1)

    @property
    def vertices(self) -> torch.Tensor:
        return torch.cat([self.interior_vertices, self.ext_vertices], dim=0)

    def __len__(self):
        return self.interior_vertices.shape[0] + self.ext_vertices.shape[0]

    @property
    def num_int_verts(self):
        return self.interior_vertices.shape[0]

    def get_vertex_values(self, camera) -> torch.Tensor:
        """Compute per-vertex (sigma, r, g, b) tensor of shape (V, 4).

        Softplus(sigma, beta=40) for density: always positive, smooth, with
        gradient=0.5 at sigma=0. At sigma=0: density=ln(2)/40=0.017, gradient=0.5
        (28x stronger than exp(-4)=0.018).
        """
        density = torch.nn.functional.softplus(self.sigma, beta=40.0)  # (V, 1)

        sh_dim = (self.max_sh_deg + 1) ** 2 - 1
        if sh_dim == 0:
            sh_rest = torch.empty((self.sigma.shape[0], 0, 3), device=self.device, dtype=self.sh.dtype)
        else:
            sh_rest = self.sh.reshape(-1, sh_dim, 3)

        color_raw = eval_sh(
            self.vertices,
            RGB2SH(self.rgb),
            sh_rest,
            camera.camera_center.to(self.device),
            self.current_sh_deg).float()
        color = torch.nn.functional.softplus(color_raw.reshape(-1, 3), beta=10)

        return torch.cat([density, color], dim=-1).float()  # (V, 4)

    def calc_tet_density(self):
        """Average vertex density per tet (for culling)."""
        density = torch.nn.functional.softplus(self.sigma, beta=40.0).reshape(-1)
        tet_verts = self.indices.long()  # (T, 4)
        return density[tet_verts].mean(dim=1)

    @torch.no_grad()
    def update_triangulation(self, **kwargs):
        """Recompute Delaunay — just swap index buffer."""
        verts = self.vertices
        v = Del(verts.shape[0])
        indices_np, prev = v.compute(verts.detach().cpu().double())
        valid_mask = (indices_np >= 0) & (indices_np < verts.shape[0])
        indices_np = indices_np[valid_mask.all(axis=1)]
        del prev

        new_indices = torch.as_tensor(indices_np).cuda()
        vols = tet_volumes(verts[new_indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            new_indices[reverse_mask] = new_indices[reverse_mask][:, [1, 0, 2, 3]]

        self.indices = new_indices.int()
        self.empty_indices = torch.empty((0, 4), dtype=self.indices.dtype, device="cuda")

    @staticmethod
    def init_from_pcd(
        point_cloud,
        cameras,
        device,
        max_sh_deg=2,
        voxel_size=0.00,
        density_offset=-3,
        **kwargs,
    ):
        torch.manual_seed(2)

        ccenters = torch.stack(
            [c.camera_center.reshape(3) for c in cameras], dim=0
        ).to(device)
        center = ccenters.mean(dim=0)
        scaling = torch.linalg.norm(
            ccenters - center.reshape(1, 3), dim=1, ord=torch.inf
        ).max()
        print(f"Scene scaling: {scaling}. Center: {center}")

        vertices = torch.as_tensor(point_cloud.points).float()

        o3d_pcd = o3d.geometry.PointCloud()
        o3d_pcd.points = o3d.utility.Vector3dVector(vertices.numpy())
        if voxel_size > 0:
            o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

        vertices = torch.as_tensor(np.asarray(o3d_pcd.points)).float()
        vertices = vertices + torch.randn(*vertices.shape) * 1e-3

        pcd_scaling = torch.linalg.norm(
            vertices - center.cpu().reshape(1, 3), dim=1, ord=2
        ).max()
        new_radius = pcd_scaling.cpu().item()

        num_ext = 1000
        ext_vertices = (
            fibonacci_spiral_on_sphere(num_ext, new_radius, device="cpu")
            + center.reshape(1, 3).cpu()
        )

        # Concatenate exterior into interior (same as SimpleModel)
        vertices = torch.cat([vertices, ext_vertices], dim=0)
        ext_vertices = torch.empty((0, 3))

        int_vertices = vertices.to(device)
        ext_verts = ext_vertices.to(device)

        # Initial Delaunay
        all_verts = torch.cat([int_vertices, ext_verts], dim=0)
        v = Del(all_verts.shape[0])
        indices_np, prev = v.compute(all_verts.detach().cpu().double())
        valid_mask = (indices_np >= 0) & (indices_np < all_verts.shape[0])
        indices_np = indices_np[valid_mask.all(axis=1)]
        del prev
        indices = torch.as_tensor(indices_np).to(device)
        vols = tet_volumes(all_verts[indices])
        reverse_mask = vols < 0
        if reverse_mask.sum() > 0:
            indices[reverse_mask] = indices[reverse_mask][:, [1, 0, 2, 3]]

        V = all_verts.shape[0]
        sh_dim = ((1 + max_sh_deg) ** 2 - 1) * 3

        # Per-vertex parameters — softplus(0, beta=40) = ln(2)/40 ≈ 0.017
        sigma = torch.zeros((V, 1), device=device)
        rgb = torch.full((V, 3), 0.5, device=device)
        sh = torch.zeros((V, sh_dim // 3, 3), device=device)

        model = VertexModel(
            int_vertices=int_vertices,
            ext_vertices=ext_verts,
            indices=indices,
            sigma=sigma,
            rgb=rgb,
            sh=sh,
            center=center,
            scene_scaling=scaling,
            max_sh_deg=max_sh_deg,
            density_offset=density_offset,
        )
        return model

    @staticmethod
    def load_ckpt(path: Path, device):
        ckpt_path = path / "ckpt.pth"
        config_path = path / "config.json"
        config = Args.load_from_json(str(config_path))
        ckpt = torch.load(ckpt_path)

        model = VertexModel(
            int_vertices=ckpt["interior_vertices"].to(device),
            ext_vertices=ckpt["ext_vertices"].to(device),
            indices=ckpt["indices"].to(device),
            sigma=ckpt["sigma"].to(device),
            rgb=ckpt["rgb"].to(device),
            sh=ckpt["sh"].to(device),
            center=ckpt["center"].to(device),
            scene_scaling=ckpt["scene_scaling"].to(device),
            max_sh_deg=config.max_sh_deg,
        )
        model.load_state_dict(ckpt)
        model.min_t = config.min_t
        return model


# ===========================================================================
# VertexOptimizer
# ===========================================================================

class VertexOptimizer:
    """Optimizer for VertexModel. Retriangulation is trivial (no state remap)."""

    def __init__(
        self,
        model: VertexModel,
        *,
        freeze_lr: float = 1e-3,
        final_freeze_lr: float = 1e-4,
        lr_delay_multi=1e-8,
        lr_delay=0,
        vertices_lr: float = 4e-4,
        final_vertices_lr: float = 4e-7,
        vert_lr_delay: int = 500,
        vertices_lr_delay_multi: float = 0.01,
        iterations: int = 30000,
        **kwargs,
    ) -> None:
        self.model = model

        self.optim = optim.CustomAdam([
            {"params": [model.sigma], "lr": freeze_lr * 1.5, "name": "sigma"},
            {"params": [model.rgb], "lr": freeze_lr, "name": "color"},
        ], eps=1e-15)
        self.sh_optim = optim.CustomAdam([
            {"params": [model.sh], "lr": freeze_lr, "name": "sh"},
        ], eps=1e-7)  # half-precision safe eps
        self.vert_lr_multi = float(model.scene_scaling.cpu())
        self.vertex_optim = optim.CustomAdam([
            {
                "params": [model.interior_vertices],
                "lr": self.vert_lr_multi * vertices_lr,
                "name": "interior_vertices",
            },
        ])

        self.scheduler = get_expon_lr_func(
            lr_init=freeze_lr,
            lr_final=final_freeze_lr,
            lr_delay_mult=lr_delay_multi,
            max_steps=iterations,
            lr_delay_steps=lr_delay,
        )

        self.vertex_lr = self.vert_lr_multi * vertices_lr
        self.vertex_scheduler_args = get_expon_lr_func(
            lr_init=self.vertex_lr,
            lr_final=self.vert_lr_multi * final_vertices_lr,
            lr_delay_mult=vertices_lr_delay_multi,
            max_steps=iterations,
            lr_delay_steps=vert_lr_delay,
        )

        self.net_optim = self.optim

    def step(self):
        self.optim.step()

    def zero_grad(self):
        self.optim.zero_grad()

    def main_step(self):
        self.step()

    def main_zero_grad(self):
        self.zero_grad()

    def update_learning_rate(self, iteration):
        for param_group in self.optim.param_groups:
            lr = self.scheduler(iteration)
            param_group["lr"] = lr

        for param_group in self.sh_optim.param_groups:
            lr = self.scheduler(iteration)
            param_group["lr"] = lr

        for param_group in self.vertex_optim.param_groups:
            if param_group["name"] == "interior_vertices":
                lr = self.vertex_scheduler_args(iteration)
                self.vertex_lr = lr
                param_group["lr"] = lr

    def regularizer(self, render_pkg, **kwargs):
        return 0.0

    @torch.no_grad()
    def update_triangulation(self, **kwargs):
        """Retriangulate — just swap indices, no parameter transfer needed."""
        self.model.update_triangulation(**kwargs)

    def _rebuild_attr_optim(self):
        """Rebuild per-vertex attribute optimizers preserving LR."""
        lr_sigma = self.optim.param_groups[0]["lr"]
        lr_color = self.optim.param_groups[1]["lr"] if len(self.optim.param_groups) > 1 else lr_sigma
        sh_lr = self.sh_optim.param_groups[0]["lr"]

        self.optim = optim.CustomAdam([
            {"params": [self.model.sigma], "lr": lr_sigma, "name": "sigma"},
            {"params": [self.model.rgb], "lr": lr_color, "name": "color"},
        ], eps=1e-15)
        self.sh_optim = optim.CustomAdam([
            {"params": [self.model.sh], "lr": sh_lr, "name": "sh"},
        ], eps=1e-7)  # half-precision safe eps
        self.net_optim = self.optim

    @torch.no_grad()
    def add_vertices(self, new_positions: torch.Tensor):
        """Add new vertices, init attributes from nearest existing vertex."""
        model = self.model
        V_old = model.vertices.shape[0]

        # Find nearest existing vertex for each new position
        # Chunked to avoid OOM
        n_new = new_positions.shape[0]
        nearest = torch.zeros(n_new, dtype=torch.long, device=model.device)
        old_verts = model.vertices.detach()
        chunk = 2048
        for s in range(0, n_new, chunk):
            e = min(s + chunk, n_new)
            d = (new_positions[s:e].unsqueeze(1) - old_verts.unsqueeze(0)).pow(2).sum(-1)
            nearest[s:e] = d.argmin(dim=1)

        new_sigma = model.sigma.data[nearest]
        new_rgb = model.rgb.data[nearest]
        new_sh = model.sh.data[nearest]

        # Extend interior_vertices via optimizer (preserves Adam state)
        model.interior_vertices = self.vertex_optim.cat_tensors_to_optimizer(
            dict(interior_vertices=new_positions)
        )["interior_vertices"]

        # Extend per-vertex params
        model.sigma = nn.Parameter(
            torch.cat([model.sigma.data, new_sigma]).contiguous().requires_grad_(True))
        model.rgb = nn.Parameter(
            torch.cat([model.rgb.data, new_rgb]).contiguous().requires_grad_(True))
        model.sh = nn.Parameter(
            torch.cat([model.sh.data, new_sh]).contiguous().requires_grad_(True))

        self._rebuild_attr_optim()
        model.update_triangulation()
        model.device = model.sigma.device

        print(f"Added {n_new} vertices (#V: {model.vertices.shape[0]}, #T: {model.indices.shape[0]})")

    @torch.no_grad()
    def split_tets_inplace(self, split_mask):
        """Densification: add vertices at centroids of selected tets."""
        model = self.model
        split_idx = torch.where(split_mask)[0]
        K = split_idx.shape[0]
        if K == 0:
            return

        # Compute centroids of tets to split
        split_tet_verts = model.indices[split_idx].long()  # (K, 4)
        verts = model.vertices
        centroids = verts[split_tet_verts].float().mean(dim=1)  # (K, 3)

        self.add_vertices(centroids)

    def clip_grad_norm_(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self.model.sigma, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.rgb, max_norm)
        torch.nn.utils.clip_grad_norm_(self.model.sh, max_norm)

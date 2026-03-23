import os, glob

# Clear stale lock files before any imports trigger JIT compilation
# (Do NOT clear .slangtorch_cache — recompiling from scratch is slow and can hang)
for _lock in glob.glob(os.path.expanduser("~/.cache/torch_extensions/*/sort_by_keys/lock")):
    os.remove(_lock)
for _lock in glob.glob(os.path.join(os.path.dirname(__file__), "rmesh_renderer", "slang", ".slangtorch_cache", "*.lock")):
    os.remove(_lock)

import time
import math
import torch
import random
from data import loader
import numpy as np
from utils.train_util import render, SimpleSampler
from model import VertexModel as Model, VertexOptimizer as TetOptimizer
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
import json
import test_util
import gc
from utils.densification import collect_render_stats, apply_densification, apply_grad_densification
from utils.decimation import apply_decimation


torch.set_num_threads(1)

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        return super().default(o)

eps = torch.finfo(torch.float).eps
args = Args()
args.tile_size = 16
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.ckpt = ""
args.resolution = 1
args.render_train = False

# Light Settings
args.max_sh_deg = 1
args.sh_interval = 0
args.sh_step = 1

# Per-vertex LR: ~1/5 of per-tet LR to compensate for gradient accumulation
args.freeze_lr = 6e-3
args.final_freeze_lr = 6e-4
args.additional_attr = 0
args.n_quad_samples = 2
args.density_offset = -4

# Vertex Settings
args.lr_delay = 0
args.vert_lr_delay = 0
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.vertices_lr_delay_multi = 1e-8
args.delaunay_interval = 300

# Distortion Settings
args.lambda_dist = 0.0
args.lambda_norm = 0.0
args.lambda_sh = 0.0
args.lambda_opacity = 0.0

# Clone Settings
args.num_samples = 50
args.k_samples = 1
args.trunc_sigma = 0.35
args.min_tet_count = 9
args.densify_start = 200
args.densify_end = 900
args.densify_interval = 300

args.within_thresh = 0.05
args.total_thresh = 1.0
args.clone_min_contrib = 2/255
args.split_min_contrib = 10/255

args.lambda_ssim = 0.0
args.lambda_ssim_bw = 0.2
args.min_t = 0.4
args.sample_cam = 8
args.data_device = 'cpu'
args.density_threshold = 0.0
args.alpha_threshold = 0.0
args.contrib_threshold = 0.0
args.threshold_start = 2500
args.voxel_size = 0.01

# Decimation Settings — after densification
args.decimate_start = 450
args.decimate_end = 700
args.decimate_interval = 100
args.decimate_count = 5000
args.decimate_threshold = 0.0

# Edge Length Regularization
args.lambda_edge_length = 0.0

# MCMC-style vertex noise (SGLD)
args.noise_lr = 0.0


# Don't touch this portion
args = Args.from_namespace(args.get_parser().parse_args())
train_cameras, test_cameras, scene_info = loader.load_dataset(
    args.dataset_path, args.image_folder, data_device=args.data_device, eval=True, resolution=args.resolution)


device = torch.device('cuda')
if len(args.ckpt) > 0:
    model = Model.load_ckpt(Path(args.ckpt), device)
else:
    model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device,
                                **args.as_dict())
min_t = args.min_t

tet_optim = TetOptimizer(model, **args.as_dict())

images = []
psnrs = []
inds = list(range(len(train_cameras)))
random.shuffle(inds)

num_densify_iter = args.densify_end - args.densify_start
N = num_densify_iter // args.densify_interval + 1
S = model.vertices.shape[0]

dschedule = list(range(args.densify_start, args.densify_end, args.densify_interval))
dschedule_decimate = list(range(args.decimate_start, args.decimate_end, args.decimate_interval))

densification_sampler = SimpleSampler(len(train_cameras), args.num_samples, device)

torch.cuda.empty_cache()

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# Upfront densification: add perturbed copies of initial vertices
init_upsample = 4  # 4x upfront then error-targeted densification at step 400
if init_upsample > 1:
    with torch.no_grad():
        n_int = model.interior_vertices.shape[0]
        old_pos = model.interior_vertices.data
        n_copies = init_upsample - 1
        new_positions_list = []
        for _ in range(n_copies):
            noise = torch.randn_like(old_pos) * 0.05
            new_positions_list.append(old_pos + noise)
        new_positions = torch.cat(new_positions_list, dim=0)

        # Initialize attributes by copying from parent vertices
        parent_idx = torch.arange(n_int, device=device).repeat(n_copies)
        new_sigma = model.sigma.data[parent_idx].clone()
        new_rgb = model.rgb.data[parent_idx].clone()
        new_sh = model.sh.data[parent_idx].clone()

        model.interior_vertices = tet_optim.vertex_optim.cat_tensors_to_optimizer(
            dict(interior_vertices=new_positions)
        )["interior_vertices"]

        model.sigma = torch.nn.Parameter(
            torch.cat([model.sigma.data, new_sigma]).contiguous().requires_grad_(True))
        model.rgb = torch.nn.Parameter(
            torch.cat([model.rgb.data, new_rgb]).contiguous().requires_grad_(True))
        model.sh = torch.nn.Parameter(
            torch.cat([model.sh.data, new_sh]).contiguous().requires_grad_(True))

        tet_optim._rebuild_attr_optim()
        model.update_triangulation()
        model.device = model.sigma.device
        print(f"Upfront densification: {n_int} to {model.vertices.shape[0]} vertices, {model.indices.shape[0]} tets")

# Gradient-based densification (disabled for this experiment)
grad_accum = torch.zeros(model.interior_vertices.shape[0], device=device)
grad_count = torch.zeros(model.interior_vertices.shape[0], device=device)
densify_grad_interval = 100
densify_grad_start = 99999
densify_grad_end = 99999
densify_grad_target = 3000  # vertices per round

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    do_delaunay = False
    # Error-targeted densification: one round at step 400
    do_cloning = (step == 400 and model.vertices.shape[0] < test_util.VERT_BUDGET)
    do_grad_densify = False
    # Refinement after densification to clean up
    do_refine = (step in [500, 600]
                 and model.vertices.shape[0] < test_util.VERT_BUDGET)
    do_sh_up = not args.sh_interval == 0 and step % args.sh_interval == 0 and step > 0
    do_sh_step = step % args.sh_step == 0
    do_decimation = (step >= args.decimate_start and step < args.decimate_end
                     and step % args.decimate_interval == 0)

    if do_delaunay:
        # Measure PSNR before retriangulation
        with torch.no_grad():
            pre_render = render(camera, model, ray_jitter=torch.ones((camera.image_height, camera.image_width, 2), device=device)*0.5, **args.as_dict())
            pre_img = pre_render['render']
            pre_l2 = ((target - pre_img)**2 * gt_mask).mean()
            pre_psnr = -20 * math.log10(math.sqrt(pre_l2.cpu().clip(min=1e-6).item()))
        old_n_tets = model.indices.shape[0]
        st = time.time()
        tet_optim.update_triangulation(
            density_threshold=args.density_threshold if step > args.threshold_start else 0,
            alpha_threshold=args.alpha_threshold if step > args.threshold_start else 0,
            high_precision=False)
        # Measure PSNR after retriangulation
        with torch.no_grad():
            post_render = render(camera, model, ray_jitter=torch.ones((camera.image_height, camera.image_width, 2), device=device)*0.5, **args.as_dict())
            post_img = post_render['render']
            post_l2 = ((target - post_img)**2 * gt_mask).mean()
            post_psnr = -20 * math.log10(math.sqrt(post_l2.cpu().clip(min=1e-6).item()))
        new_n_tets = model.indices.shape[0]
        dt_del = time.time() - st
        print(f"[DELAUNAY step {step}] PSNR: {pre_psnr:.2f} -> {post_psnr:.2f} (delta={post_psnr-pre_psnr:+.2f}) "
              f"tets: {old_n_tets} -> {new_n_tets} time={dt_del:.2f}s")

    if len(inds) == 0:
        print(f"TRAIN PSNR: {sum(psnrs)/len(psnrs)} #V: {len(model)} #T: {model.indices.shape[0]}")
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs = []
    ind = inds.pop()
    camera = train_cameras[ind]
    target = camera.original_image.cuda()
    gt_mask = camera.gt_alpha_mask.cuda()

    st = time.time()
    ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)
    render_pkg = render(camera, model, ray_jitter=ray_jitter, **args.as_dict())
    image = render_pkg['render']

    if step == 11:
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserved = torch.cuda.memory_reserved() / (1024**2)
        print(f"[VRAM] allocated={alloc:.0f}MB reserved={reserved:.0f}MB after render")
    l1_loss = ((target - image).abs() * gt_mask).mean()
    l2_loss = ((target - image)**2 * gt_mask).mean()
    reg = tet_optim.regularizer(render_pkg, **args.as_dict())
    ssim_loss = (1-fused_ssim(image.unsqueeze(0), target.unsqueeze(0))).clip(min=0, max=1)
    dl_loss = render_pkg.get('distortion_loss', 0.0)
    loss = (1-args.lambda_ssim)*l1_loss + \
           args.lambda_ssim*ssim_loss + \
           reg + \
           args.lambda_opacity * (1-render_pkg['alpha']).mean() + \
           args.lambda_dist * dl_loss
    if args.lambda_ssim_bw > 0:
        bw_weights = torch.tensor([0.2989, 0.5870, 0.1140], device=image.device).view(3, 1, 1)
        image_bw = (image * bw_weights).sum(dim=0, keepdim=True)
        target_bw = (target * bw_weights).sum(dim=0, keepdim=True)
        ssim_bw_loss = (1-fused_ssim(image_bw.unsqueeze(0), target_bw.unsqueeze(0))).clip(min=0, max=1)
        loss = loss + args.lambda_ssim_bw * ssim_bw_loss

    if args.lambda_edge_length > 0:
        idx = model.indices.long()
        v = model.vertices
        edge_len_sq = (
            (v[idx[:, 0]] - v[idx[:, 1]]).pow(2).sum(-1) +
            (v[idx[:, 0]] - v[idx[:, 2]]).pow(2).sum(-1) +
            (v[idx[:, 0]] - v[idx[:, 3]]).pow(2).sum(-1) +
            (v[idx[:, 1]] - v[idx[:, 2]]).pow(2).sum(-1) +
            (v[idx[:, 1]] - v[idx[:, 3]]).pow(2).sum(-1) +
            (v[idx[:, 2]] - v[idx[:, 3]]).pow(2).sum(-1)
        ) / 6
        loss = loss + args.lambda_edge_length * edge_len_sq.mean()

    loss.backward()

    # Accumulate vertex position gradient norms for densification
    if model.interior_vertices.grad is not None and step < densify_grad_end:
        g = model.interior_vertices.grad.detach()
        n_cur = min(g.shape[0], grad_accum.shape[0])
        grad_accum[:n_cur] += g[:n_cur].norm(dim=1)
        grad_count[:n_cur] += 1

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if True:
        tet_optim.vertex_optim.step()
        tet_optim.vertex_optim.zero_grad()

        # MCMC-style SGLD noise on vertex positions
        if args.noise_lr > 0:
            with torch.no_grad():
                vlr = tet_optim.vertex_lr
                noise = torch.randn_like(model.interior_vertices) * args.noise_lr * vlr
                model.interior_vertices.data.add_(noise)

    tet_optim.update_learning_rate(step)

    if do_sh_up:
        model.sh_up()

    if do_cloning:
        with torch.no_grad():
            psnrs = []
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]

            gc.collect()
            torch.cuda.empty_cache()
            model.eval()
            stats = collect_render_stats(sampled_cams, model, args, device)
            model.train()
            target_addition = min(test_util.VERT_BUDGET - model.vertices.shape[0], 200000)

            apply_densification(
                stats,
                model       = model,
                tet_optim   = tet_optim,
                args        = args,
                device      = device,
                target_addition= target_addition
            )
            del stats
            gc.collect()
            torch.cuda.empty_cache()

    if do_refine:
        with torch.no_grad():
            # Measure PSNR impact of refinement
            pre_render = render(camera, model, ray_jitter=torch.ones((camera.image_height, camera.image_width, 2), device=device)*0.5, **args.as_dict())
            pre_l2 = ((target - pre_render['render'])**2 * gt_mask).mean()
            pre_psnr = -20 * math.log10(math.sqrt(pre_l2.cpu().clip(min=1e-6).item()))

            budget_left = test_util.VERT_BUDGET - model.vertices.shape[0]
            n_add = min(5000, budget_left)
            if n_add > 0:
                added = tet_optim.refine_bad_tets(max_vertices=n_add)

                post_render = render(camera, model, ray_jitter=torch.ones((camera.image_height, camera.image_width, 2), device=device)*0.5, **args.as_dict())
                post_l2 = ((target - post_render['render'])**2 * gt_mask).mean()
                post_psnr = -20 * math.log10(math.sqrt(post_l2.cpu().clip(min=1e-6).item()))
                print(f"  [REFINE step {step}] PSNR: {pre_psnr:.2f} -> {post_psnr:.2f} (delta={post_psnr-pre_psnr:+.2f})")

    if do_grad_densify:
        with torch.no_grad():
            target = min(50000, test_util.VERT_BUDGET - model.vertices.shape[0])
            if target > 0:
                apply_grad_densification(
                    model, tet_optim, grad_accum, grad_count,
                    target_addition=target, mode="edge_midpoint")
                # Resize accumulators for new vertex count
                new_n = model.interior_vertices.shape[0]
                new_accum = torch.zeros(new_n, device=device)
                new_count = torch.zeros(new_n, device=device)
                n_old = min(grad_accum.shape[0], new_n)
                new_accum[:n_old] = grad_accum[:n_old]
                new_count[:n_old] = grad_count[:n_old]
                grad_accum = new_accum
                grad_count = new_count
                gc.collect()
                torch.cuda.empty_cache()

    if do_decimation:
        with torch.no_grad():
            n_removed = apply_decimation(model, tet_optim, args, device)
            gc.collect()
            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = t1 - t0

    if step > 10:
        total_training_time += dt

    psnr = -20 * math.log10(math.sqrt(l2_loss.detach().cpu().clip(min=1e-6).item()))
    psnrs.append(psnr)

    if step % 50 == 0:
        print(f"[step {step:5d}] PSNR={psnr:.2f} #V={len(model)} #T={model.indices.shape[0]} t={total_training_time:.1f}s")

    if step == 0:
        gc.collect()
        gc.freeze()
        gc.disable()
    elif (step + 1) % 5000 == 0:
        gc.collect()
        

    if step > 10 and total_training_time >= test_util.TIME_BUDGET:
        break

    step += 1

print()

torch.cuda.synchronize()
torch.cuda.empty_cache()

splits = zip(['test'], [test_cameras])
results = test_util.evaluate(model, splits, "", args.tile_size, min_t, save=False, n_quad_samples=args.n_quad_samples)

all_data = dict(
    n_vertices = model.vertices.shape[0],
    n_interior_vertices = model.interior_vertices.shape[0],
    n_tets = model.indices.shape[0],
    **results
)
print("----------")
for k, v in all_data.items():
    print(f"{k}: {v}")

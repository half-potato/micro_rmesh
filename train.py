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
from utils.densification import collect_render_stats, apply_densification, apply_grad_densification, apply_vertex_densification, apply_mcmc_relocation
from utils.decimation import apply_decimation
from utils.train_util import pad_image2even
from utils import cam_util
import imageio
import mediapy


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
args.freeze_lr = 1e-2
args.final_freeze_lr = 1e-3
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
args.iterations = 30000  # slow LR decay — model needs high LR for recovery after densification

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

# Decimation — light cleanup only after densification
args.decimate_start = 450
args.decimate_end = 1350
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

DEBUG_PSNR = True  # measure PSNR over all train cameras (slow but accurate)

# Debug video: sample frame rendered periodically + 360 orbit at end
if DEBUG_PSNR:
    debug_output = Path("./") / "debug_output"
    debug_output.mkdir(exist_ok=True)
    sample_camera = test_cameras[args.sample_cam] if len(test_cameras) > args.sample_cam else train_cameras[args.sample_cam]
    video_writer = imageio.get_writer(str(debug_output / "training.mp4"), fps=10)

def measure_psnr(model, cameras, args, device):
    """Mean PSNR over all cameras."""
    total_mse = 0.0
    for i, cam in enumerate(cameras):
        out = render(cam, model, ray_jitter=torch.ones((cam.image_height, cam.image_width, 2), device=device)*0.5, **args.as_dict())
        img = out['render']
        if torch.isnan(img).any() or torch.isinf(img).any():
            print(f"  [measure_psnr] camera {i}: NaN={torch.isnan(img).any()}, Inf={torch.isinf(img).any()}, "
                  f"min={img[~torch.isnan(img)].min():.4f}, max={img[~torch.isnan(img)].max():.4f}")
            continue
        gt = cam.original_image.to(device)
        mask = cam.gt_alpha_mask.to(device) if cam.gt_alpha_mask is not None else 1.0
        total_mse += ((gt - img)**2 * mask).mean().item()
    avg_mse = total_mse / len(cameras)
    return -10 * math.log10(max(avg_mse, 1e-10))

t_start_training = time.time()
smooth_train_loss = 0
total_training_time = 0
step = 0

# Upfront densification: add perturbed copies of initial vertices
init_upsample = 8  # 8x upfront (411k) + error densify to fill toward 1M
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

        tet_optim._cat_attrs(new_sigma, new_rgb, new_sh)
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
    # Error-targeted densification at 400 and 1200, MCMC at 800. No refine — too disruptive at scale.
    do_cloning = False #(step in [400, 1200] and model.vertices.shape[0] < test_util.VERT_BUDGET)
    do_mcmc = step > 0 and step % 100 == 0
    do_grad_densify = False
    do_refine = step > 0 and step % 25 == 0 and not do_mcmc
    do_sh_up = not args.sh_interval == 0 and step % args.sh_interval == 0 and step > 0
    do_sh_step = step % args.sh_step == 0
    do_decimation = (step >= args.decimate_start and step < args.decimate_end
                     and step % args.decimate_interval == 0)

    if do_delaunay:
        if DEBUG_PSNR:
            with torch.no_grad():
                model.eval()
                pre_psnr = measure_psnr(model, train_cameras, args, device)
                model.train()
        old_n_tets = model.indices.shape[0]
        st = time.time()
        tet_optim.update_triangulation(
            density_threshold=args.density_threshold if step > args.threshold_start else 0,
            alpha_threshold=args.alpha_threshold if step > args.threshold_start else 0,
            high_precision=False)
        new_n_tets = model.indices.shape[0]
        dt_del = time.time() - st
        if DEBUG_PSNR:
            with torch.no_grad():
                model.eval()
                post_psnr = measure_psnr(model, train_cameras, args, device)
                model.train()
            print(f"[DELAUNAY step {step}] PSNR: {pre_psnr:.2f} -> {post_psnr:.2f} (delta={post_psnr-pre_psnr:+.2f}) "
                  f"tets: {old_n_tets} -> {new_n_tets} time={dt_del:.2f}s")
        else:
            print(f"[DELAUNAY step {step}] tets: {old_n_tets} -> {new_n_tets} time={dt_del:.2f}s")

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

    if do_mcmc or do_refine:
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

    if do_mcmc:
        with torch.no_grad():
            if DEBUG_PSNR:
                model.eval()
                pre_psnr = measure_psnr(model, train_cameras, args, device)
                model.train()

            st_mcmc = time.time()
            n_relocated = apply_mcmc_relocation(
                model, tet_optim, args, device, max_relocate=int(0.05 * model.vertices.shape[0]))
            dt_mcmc = time.time() - st_mcmc

            # Debug: check model state after MCMC
            v = model.vertices
            assert not torch.isnan(v).any(), \
                f"NaN in vertices after MCMC! int_nan={torch.isnan(model.interior_vertices.data).any()}, ext_nan={torch.isnan(model.ext_vertices).any()}"
            assert not torch.isnan(model.sigma.data).any(), "NaN in sigma after MCMC!"
            assert model.sigma.shape[0] == model.interior_vertices.shape[0], \
                f"sigma/verts mismatch: sigma={model.sigma.shape[0]} verts={model.interior_vertices.shape[0]}"
            idx = model.indices.long()
            assert idx.max() < v.shape[0], f"index OOB: max_idx={idx.max()} n_verts={v.shape[0]}"
            assert idx.min() >= 0, f"negative index: min_idx={idx.min()}"
            # Check a test render
            test_cam = train_cameras[0]
            test_out = render(test_cam, model, ray_jitter=torch.ones((test_cam.image_height, test_cam.image_width, 2), device=device)*0.5, **args.as_dict())
            test_img = test_out['render']
            print(f"  [MCMC debug] render has_nan={torch.isnan(test_img).any()}, has_inf={torch.isinf(test_img).any()}, "
                  f"min={test_img.min():.4f}, max={test_img.max():.4f}")

            if DEBUG_PSNR:
                model.eval()
                post_psnr = measure_psnr(model, train_cameras, args, device)
                model.train()
                print(f"  [MCMC step {step}] relocated={n_relocated}, PSNR: {pre_psnr:.2f} -> {post_psnr:.2f} (delta={post_psnr-pre_psnr:+.2f}) time={dt_mcmc:.2f}s")
            else:
                print(f"  [MCMC step {step}] relocated={n_relocated} time={dt_mcmc:.2f}s")

            gc.collect()
            torch.cuda.empty_cache()

    if do_refine:
        with torch.no_grad():
            if DEBUG_PSNR:
                model.eval()
                pre_psnr = measure_psnr(model, train_cameras, args, device)
                model.train()

            st_refine = time.time()
            budget_left = test_util.VERT_BUDGET - model.vertices.shape[0]
            n_add = min(5000, budget_left)
            if n_add > 0:
                added = tet_optim.refine_bad_tets(max_vertices=n_add)
                dt_refine = time.time() - st_refine

                if DEBUG_PSNR:
                    model.eval()
                    post_psnr = measure_psnr(model, train_cameras, args, device)
                    model.train()
                    print(f"  [REFINE step {step}] added={added}, PSNR: {pre_psnr:.2f} -> {post_psnr:.2f} (delta={post_psnr-pre_psnr:+.2f}) time={dt_refine:.2f}s")
                else:
                    print(f"  [REFINE step {step}] added={added} time={dt_refine:.2f}s")

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
            if DEBUG_PSNR:
                model.eval()
                pre_psnr = measure_psnr(model, train_cameras, args, device)
                model.train()

            st_dec = time.time()
            n_removed = apply_decimation(model, tet_optim, args, device)
            dt_dec = time.time() - st_dec

            if DEBUG_PSNR:
                model.eval()
                post_psnr = measure_psnr(model, train_cameras, args, device)
                model.train()
                print(f"  [DECIMATE step {step}] removed={n_removed}, PSNR: {pre_psnr:.2f} -> {post_psnr:.2f} (delta={post_psnr-pre_psnr:+.2f}) time={dt_dec:.2f}s")
            else:
                print(f"  [DECIMATE step {step}] removed={n_removed} time={dt_dec:.2f}s")

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

    if DEBUG_PSNR and step % 10 == 0:
        with torch.no_grad():
            render_pkg = render(sample_camera, model, ray_jitter=torch.ones((sample_camera.image_height, sample_camera.image_width, 2), device=device)*0.5, **args.as_dict())
            sample_image = render_pkg['render'].permute(1, 2, 0)
            sample_image = (sample_image.clamp(0, 1).detach().cpu().numpy() * 255).astype(np.uint8)
            video_writer.append_data(pad_image2even(sample_image))

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

if DEBUG_PSNR:
    video_writer.close()
    print(f"Training video saved to {debug_output / 'training.mp4'}")

    # 360 orbit video
    with torch.no_grad():
        model.eval()
        epath = cam_util.generate_cam_path(train_cameras, 400)
        eimages = []
        for cam in epath:
            render_pkg = render(cam, model, ray_jitter=torch.ones((cam.image_height, cam.image_width, 2), device=device)*0.5, **args.as_dict())
            image = render_pkg['render'].permute(1, 2, 0).clamp(0, 1).detach().cpu().numpy()
            eimages.append(pad_image2even(image))
        model.train()
    mediapy.write_video(str(debug_output / "rotating.mp4"), eimages)
    print(f"360 video saved to {debug_output / 'rotating.mp4'}")

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

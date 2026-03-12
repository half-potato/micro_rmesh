import time
import math
import torch
import random
from data import loader
import numpy as np
from utils.train_util import render, SimpleSampler
from model import SimpleModel as Model, SimpleOptimizer as TetOptimizer
from fused_ssim import fused_ssim
from pathlib import Path, PosixPath
from utils.args import Args
import json
import test_util
import gc
from utils.densification import collect_render_stats, apply_densification
from utils.decimation import apply_decimation


torch.set_num_threads(1)

class CustomEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, PosixPath):
            return str(o)
        return super().default(o)

eps = torch.finfo(torch.float).eps
args = Args()
args.tile_size = 4
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.ckpt = ""
args.resolution = 1
args.render_train = False

# Light Settings
args.max_sh_deg = 3
args.sh_interval = 0
args.sh_step = 1

# SimpleModel uses freeze_lr/final_freeze_lr for per-tet param LR
args.freeze_lr = 3e-2
args.final_freeze_lr = 3e-3
args.additional_attr = 0
args.density_offset = -3

# Vertex Settings
args.lr_delay = 0
args.vert_lr_delay = 0
args.vertices_lr = 1e-4
args.final_vertices_lr = 1e-6
args.vertices_lr_delay_multi = 1e-8
args.delaunay_interval = 10

# Distortion Settings
args.lambda_dist = 0.0
args.lambda_norm = 0.0
args.lambda_sh = 0.0
args.lambda_opacity = 0.0

# Clone Settings
args.num_samples = 200
args.k_samples = 1
args.trunc_sigma = 0.35
args.min_tet_count = 9
args.densify_start = 2000
args.densify_end = 16000
args.densify_interval = 500

args.within_thresh = 0.3 / 2.7
args.total_thresh = 2.0
args.clone_min_contrib = 5/255
args.split_min_contrib = 10/255

args.lambda_ssim = 0.0
args.lambda_ssim_bw = 0.2
args.min_t = 0.4
args.sample_cam = 8
args.data_device = 'cpu'
args.density_threshold = 0.1
args.alpha_threshold = 0.1
args.contrib_threshold = 0.0
args.threshold_start = 4500
args.voxel_size = 0.01

# Decimation Settings
args.decimate_start = 4000
args.decimate_end = 17000
args.decimate_interval = 2000
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
entropies = []
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

while True:
    torch.cuda.synchronize()
    t0 = time.time()
    do_delaunay = step % args.delaunay_interval == 0
    do_cloning = step in dschedule
    do_sh_up = not args.sh_interval == 0 and step % args.sh_interval == 0 and step > 0
    do_sh_step = step % args.sh_step == 0
    do_decimation = step in dschedule_decimate

    if do_delaunay:
        st = time.time()
        tet_optim.update_triangulation(
            density_threshold=args.density_threshold if step > args.threshold_start else 0,
            alpha_threshold=args.alpha_threshold if step > args.threshold_start else 0,
            high_precision=False)

    if len(inds) == 0:
        avg_entropy = sum(entropies)/len(entropies) if entropies else 0.0
        print(f"TRAIN PSNR: {sum(psnrs)/len(psnrs):.2f} ENTROPY: {avg_entropy:.4f} #V: {len(model)} #T: {model.indices.shape[0]}")
        inds = list(range(len(train_cameras)))
        random.shuffle(inds)
        psnrs = []
        entropies = []
    ind = inds.pop()
    camera = train_cameras[ind]
    target = camera.original_image.cuda()
    gt_mask = camera.gt_alpha_mask.cuda()

    st = time.time()
    ray_jitter = torch.rand((camera.image_height, camera.image_width, 2), device=device)
    render_pkg = render(camera, model, ray_jitter=ray_jitter, **args.as_dict())
    image = render_pkg['render']

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

    tet_optim.main_step()
    tet_optim.main_zero_grad()

    if do_sh_step and tet_optim.sh_optim is not None:
        tet_optim.sh_optim.step()
        tet_optim.sh_optim.zero_grad()

    if do_delaunay:
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
            sampled_cams = [train_cameras[i] for i in densification_sampler.nextids()]

            gc.collect()
            torch.cuda.empty_cache()
            model.eval()
            stats = collect_render_stats(sampled_cams, model, args, device)
            model.train()
            target_addition = test_util.VERT_BUDGET - model.vertices.shape[0]

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
    entropies.append(dl_loss.detach().cpu().item() if torch.is_tensor(dl_loss) else dl_loss)

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
results = test_util.evaluate(model, splits, "", args.tile_size, min_t, save=False)

all_data = dict(
    n_vertices = model.vertices.shape[0],
    n_interior_vertices = model.interior_vertices.shape[0],
    n_tets = model.indices.shape[0],
    **results
)
print("----------")
for k, v in all_data.items():
    print(f"{k}: {v}")

import time
import torch
import numpy as np
from gdel3d import Del
from utils.topo_utils import tet_volumes
from pathlib import Path
from data import loader
from model import VertexModel as Model
from utils.args import Args

args = Args()
args.tile_size = 16
args.image_folder = "images_4"
args.dataset_path = Path("/optane/nerf_datasets/360/bicycle")
args.resolution = 1
args.max_sh_deg = 1
args.density_offset = -4
args.voxel_size = 0.01
args = Args.from_namespace(args.get_parser().parse_args())
train_cameras, _, scene_info = loader.load_dataset(args.dataset_path, args.image_folder, data_device='cpu', eval=True, resolution=args.resolution)
device = torch.device('cuda')
model = Model.init_from_pcd(scene_info.point_cloud, train_cameras, device, **args.as_dict())

# Simulate 4x upfront
old_pos = model.interior_vertices.data
copies = [old_pos + torch.randn_like(old_pos) * 0.05 for _ in range(3)]
all_verts = torch.cat([old_pos] + copies, dim=0).cuda()
V = all_verts.shape[0]
print(f"\nBenchmarking with {V} vertices\n")

for trial in range(3):
    torch.cuda.synchronize()
    
    t0 = time.time()
    verts_cpu = all_verts.detach().cpu().double()
    t1 = time.time()
    
    v = Del(V)
    indices_np, prev = v.compute(verts_cpu)
    t2 = time.time()
    
    valid_mask = (indices_np >= 0) & (indices_np < V)
    indices_np = indices_np[valid_mask.all(axis=1)]
    del prev
    t3 = time.time()
    
    new_indices = torch.as_tensor(indices_np).cuda()
    torch.cuda.synchronize()
    t4 = time.time()
    
    vols = tet_volumes(all_verts[new_indices])
    reverse_mask = vols < 0
    if reverse_mask.sum() > 0:
        new_indices[reverse_mask] = new_indices[reverse_mask][:, [1, 0, 2, 3]]
    torch.cuda.synchronize()
    t5 = time.time()
    
    print(f"Trial {trial}: total={t5-t0:.3f}s | cpu_xfer={t1-t0:.3f}s | gdel3d={t2-t1:.3f}s | filter={t3-t2:.3f}s | gpu_xfer={t4-t3:.3f}s | vol_fix={t5-t4:.3f}s | #tets={new_indices.shape[0]}")

# scipy comparison
from scipy.spatial import Delaunay as ScipyDelaunay
verts_np = verts_cpu.numpy()
for trial in range(2):
    t0 = time.time()
    tri = ScipyDelaunay(verts_np)
    t1 = time.time()
    print(f"\nScipy trial {trial}: {t1-t0:.3f}s, #tets={tri.simplices.shape[0]}")

import cv2
import math
import torch
import numpy as np
from data.camera import Camera
from delaunay_rasterization.internal.alphablend_tiled_slang_interp import AlphaBlendTiledRender
from delaunay_rasterization.internal.render_grid import RenderGrid
from delaunay_rasterization.internal.tile_shader_slang import vertex_and_tile_shader
import time
from icecream import ic

def render(camera: Camera, model, cell_values=None, tile_size=4, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None,
           **kwargs):
    device = model.device
    if ray_jitter is None:
        ray_jitter = 0.5*torch.ones((camera.image_height, camera.image_width, 2), device=device)
    else:
        assert(ray_jitter.shape[0] == camera.image_height)
        assert(ray_jitter.shape[1] == camera.image_width)
        assert(ray_jitter.shape[2] == 2)
    vertices = model.vertices
    
    render_grid = RenderGrid(camera.image_height,
                             camera.image_width,
                             tile_height=tile_size,
                             tile_width=tile_size)
    tcam = dict(
        tile_height=tile_size,
        tile_width=tile_size,
        grid_height=render_grid.grid_height,
        grid_width=render_grid.grid_width,
        min_t=min_t,
        **camera.to_dict(device)
    )
    sorted_tetra_idx, tile_ranges, vs_tetra, circumcenter, mask, _ = vertex_and_tile_shader(
        model.indices,
        vertices,
        tcam,
        render_grid)
    extras = {}
    if cell_values is None:
        cell_values = torch.zeros((mask.shape[0], model.feature_dim), device=circumcenter.device)
        if mask.sum() > 0 and model.mask_values:
            shs, values = model.get_cell_values(camera, mask, circumcenter[mask])
            cell_values[mask] = values
        else:
            shs, cell_values = model.get_cell_values(camera, all_circumcenters=circumcenter)

    image_rgb, xyzd_img, distortion_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        vertices,
        cell_values,
        render_grid,
        tcam,
        ray_jitter,
        model.additional_attr)
    alpha = image_rgb.permute(2,0,1)[3, ...].exp()
    total_density = (distortion_img[:, :, 2]**2).clip(min=1e-6)
    distortion_loss = (((distortion_img[:, :, 0] - distortion_img[:, :, 1]) + distortion_img[:, :, 4]) / total_density).clip(min=0)

    # unrotate the xyz part of the xyzd_img
    rotated = xyzd_img[..., :3].reshape(-1, 3) @ camera.world_view_transform[:3, :3].to(device)
    rxyzd_img = torch.cat([rotated.reshape(xyzd_img[..., :3].shape), xyzd_img[..., 3:]], dim=-1)
    
    render_pkg = {
        'aux': image_rgb.permute(2,0,1)[4:, ...] * camera.gt_alpha_mask.to(device),
        'render': image_rgb.permute(2,0,1)[:3, ...] * camera.gt_alpha_mask.to(device),
        'alpha': alpha,
        'distortion_loss': distortion_loss.mean(),
        'mask': mask,
        'xyzd': rxyzd_img,
        'weight_square': image_rgb.permute(2,0,1)[4:5, ...],
        "cell_values": cell_values,
        **extras
    }
    return render_pkg

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


def pad_hw2even(h, w):
    return int(math.ceil(h / 2))*2, int(math.ceil(w / 2))*2

def pad_image2even(im, fnp=np):
    h, w = im.shape[:2]
    nh, nw = pad_hw2even(h, w)
    im_full = fnp.zeros((nh, nw, 3), dtype=im.dtype)
    im_full[:h, :w] = im
    return im_full

class SimpleSampler:
    def __init__(self, total_num_samples, batch_size, device):
        self.total_num_samples = total_num_samples
        self.batch_size = batch_size
        self.curr = total_num_samples
        self.ids = None
        self.device = device

    def nextids(self, batch_size=None):
        batch_size = self.batch_size if batch_size is None else batch_size
        self.curr += batch_size
        if self.curr + batch_size > self.total_num_samples:
            # self.ids = torch.LongTensor(np.random.permutation(self.total_num_samples))
            self.ids = torch.randperm(self.total_num_samples, dtype=torch.long, device=self.device)
            self.curr = 0
        ids = self.ids[self.curr : self.curr + batch_size]
        return ids

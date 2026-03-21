import cv2
import math
import torch
import numpy as np
from data.camera import Camera
from rmesh_renderer.alphablend_tiled_slang_interp import AlphaBlendTiledRender
from rmesh_renderer.render_grid import RenderGrid
from rmesh_renderer.tile_shader_slang import vertex_and_tile_shader
import time
from icecream import ic

def render(camera: Camera, model, vertex_values=None, tile_size=4, min_t=0.1,
           scene_scaling=1, clip_multi=0, ray_jitter=None, n_quad_samples=2,
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
    if vertex_values is None:
        vertex_values = model.get_vertex_values(camera)

    image_rgb, xyzd_img, distortion_img, tet_alive = AlphaBlendTiledRender.apply(
        sorted_tetra_idx,
        tile_ranges,
        model.indices,
        vertices,
        vertex_values,
        render_grid,
        tcam,
        ray_jitter,
        model.additional_attr,
        n_quad_samples)
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
        "vertex_values": vertex_values,
        **extras
    }
    return render_pkg


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

#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getWorld2View2_tensor, getWorld2View, focal2fov, fov2focal

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", time = 0,
                 mask = None, depth=None
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.T0 = T.copy()
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.time = time
        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")
        

        self.original_image = image.clamp(0.0, 1.0)[:3,:,:]

        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask

        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width))

        if depth is not None:
            self.invdepthmap = 1.0 / (depth + 1e-7)
        

        if depth is not None:
            self.depth_mask = torch.logical_and(self.invdepthmap > 0, self.invdepthmap < (1.0 / 1e-7))

        self.depth_reliable = True
        
        self.mask = mask
        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        # .cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        # .cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=data_device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=data_device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=data_device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=data_device)
        )
    def move_to_device(self):
        # move all tensors to device
        self.original_image = self.original_image.to(self.data_device)
        self.world_view_transform = self.world_view_transform.to(self.data_device)
        self.projection_matrix = self.projection_matrix.to(self.data_device)
        self.full_proj_transform = self.full_proj_transform.to(self.data_device)
        self.camera_center = self.camera_center.to(self.data_device)

    def update_RT(self, R, t):
        self.R = R
        self.T = t
        self.world_view_transform = torch.tensor(getWorld2View2(self.R, self.T, self.trans, self.scale)).transpose(0, 1).cuda().float()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda().float()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time
    
class Render_Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy, image, gt_alpha_mask=None, mono_depth=None, conf=None, 
                 trans=torch.tensor([0.0, 0.0, 0.0]), scale=1.0, data_device="cuda",
                 white_background=False
                 ):
        super(Render_Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.mask = gt_alpha_mask.to(self.data_device)
            self.original_image[(self.mask <= 0.5).expand_as(self.original_image)] = 1.0 if white_background else 0.0
        else:
            self.mask = None

        if mono_depth is not None:
            self.mono_depth = mono_depth.to(data_device)
        
        if conf is not None:
            self.conf = conf.to(data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans.float()
        self.scale = scale

        self.world_view_transform = getWorld2View2_tensor(R, T).transpose(0, 1).cuda().float()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda().float()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.cam_rot_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=data_device)
        )
        self.cam_trans_delta = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=data_device)
        )

        self.exposure_a = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=data_device)
        )
        self.exposure_b = nn.Parameter(
            torch.tensor([0.0], requires_grad=True, device=data_device)
        )


def align_cameras_with_K(src_caminfo, dst_caminfo):
    """
    Info is CameraInfo type, src_caminfo: test, dst: train
    Directly use the focal from dust3r 
    """
    dst_FovY = dst_caminfo.FovY
    dst_FovX = dst_caminfo.FovX
    dst_focalx = fov2focal(dst_caminfo.FovX, dst_caminfo.width)
    dst_focaly = fov2focal(dst_caminfo.FovY, dst_caminfo.height)

    src_FovY = src_caminfo.FovY
    src_FovX = src_caminfo.FovX
    src_focalx = fov2focal(src_caminfo.FovX, dst_caminfo.width)
    src_focaly = fov2focal(src_caminfo.FovY, dst_caminfo.height)

    # get RT matrix
    src_world_view_transform = getWorld2View(src_caminfo.R, src_caminfo.T)
    dst_world_view_transform = getWorld2View(dst_caminfo.R, dst_caminfo.T)

    # take K into consider
    K_dst = np.eye(4)
    K_src = np.eye(4)

    K_dst[0, 0] = dst_focalx
    K_dst[1, 1] = dst_focaly
    K_dst[0, 2] = dst_caminfo.width / 2
    K_dst[1, 2] = dst_caminfo.height / 2

    K_src[0, 0] = src_focalx
    K_src[1, 1] = src_focaly
    K_src[0, 2] = dst_caminfo.width / 2
    K_src[1, 2] = dst_caminfo.height / 2

    # get affine matrix from src to dst
    transformation_matrix = np.linalg.inv(src_world_view_transform) @ np.linalg.inv(K_src) @ K_dst @ dst_world_view_transform

    return dst_FovY, dst_FovX, transformation_matrix

    
def align_cameras(src_caminfo, dst_caminfo):
    """
    Info is CameraInfo type
    Directly use the focal from dust3r 
    """
    dst_FovY = dst_caminfo.FovY
    dst_FovX = dst_caminfo.FovX
    # get RT matrix
    src_world_view_transform = getWorld2View(src_caminfo.R, src_caminfo.T)
    dst_world_view_transform = getWorld2View(dst_caminfo.R, dst_caminfo.T)
    # get affine matrix
    transformation_matrix = np.linalg.inv(src_world_view_transform) @ dst_world_view_transform

    return dst_FovY, dst_FovX, transformation_matrix


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

import os
import sys
import cv2
from PIL import Image
from scene.cameras import Camera
from scene.cameras import align_cameras, align_cameras_with_K

from typing import NamedTuple, Optional, List, Tuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
import torchvision.transforms as transforms
import copy
# from utils.graphics_utils import getWorld2View2, getWorld2View, focal2fov, fov2focal
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal, getWorld2View,  \
            create_spheric_poses_along_y, pose_opengl_to_opencv, opengl_c2w_to_opencv_c2w, render_path_spiral_default, create_spheric_poses_along_y
import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
from utils.general_utils import PILtoTorch
from tqdm import tqdm
import camtools as ct


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    time : float
    mask: np.array
    mono_depth: Optional[np.ndarray] = None
   
class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    render_cameras: list
    video_cameras: list
    nerf_normalization: dict
    ply_path: str
    maxtime: int

def load_raw_depth(fpath="raw.png"):
    depth = np.load(fpath)
    depth = (depth).astype(np.float32) # type: ignore

    return depth

def load_raw_conf(fpath="raw.png"):
    conf = np.load(fpath)
    return conf

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    # breakpoint()
    return {"translate": translate, "radius": radius}


def readLLFFCameras(cam_extrinsics, cam_intrinsics, images_folder, bound=1.0):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec) * bound

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
            print("Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!")
            # assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time=idx, mask=None)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos
    

def readDust3RInfo(args, path, transformsfile, white_background, extension=".png", ply_name=""):
    cam_infos = []
    bound = 1

    mono_depth_raw_size = 1024
    if os.path.exists(os.path.join(path, "train_info.json")): #using train_info.json to evaluate. will be update in the future version
        with open(os.path.join(path, "train_info.json")) as json_file:
            contents = json.load(json_file)
        dust3r_name = contents[0]['dust3r_name']
        input_name = contents[0]['input_name']
        test_name = contents[0]['test_name']
    else:
        import warnings
        warnings.warn("train_info.json is not exsit!!! Use eye instead!")
        dust3r_name = None
        input_name = None
        if args is not None:
            dust3r_name = args.dust3r_name
            input_name = args.input_name
        if args is not None and args.test_name != "default":
            test_name = args.test_name
    
    print(f"[red] The input image name is {input_name} [/red]")
    print(f"[green] The test image name is {test_name} [/green]")

    
    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        if dust3r_name is None:
            dust3r_name = contents[0]["file_path"]
        dusr3r_caminfo = None

        name_base = [frame["file_path"].replace('rgb/','').split("_")[0] for frame in contents[1:]]
        name_base_id = list(set(name_base))
        name_base_id.sort()
        print("name_base_id ", name_base_id)

        for idx, frame in enumerate(contents):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            num_frames = 15

            R = np.array(frame['rotation'])
            T = np.array(frame['position']) * bound
            c2w = np.eye(4)
            c2w[:3, :3] = R
            c2w[:3, 3] = T
            w2c = np.linalg.inv(c2w)
            R = w2c[:3, :3].T
            T = w2c[:3, 3]
            focal_length_y = frame['fy'] # / bound
            focal_length_x = frame['fx'] # / bound
            FovY = focal2fov(focal_length_y, frame['height'])
            FovX = focal2fov(focal_length_x, frame['width'])

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))
            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")


            depth_path = os.path.join(path, 'depth_anything_v2_align', str(cam_name).split("/")[-1].replace(".png",".npy"))
            if os.path.exists(depth_path):
                loaded_depth = load_raw_depth(depth_path)
                mono_depth = torch.from_numpy(loaded_depth).reshape(1, 1, -1, mono_depth_raw_size)
                mono_depth = F.interpolate(mono_depth, size=(image.size[1], image.size[0]), mode='bicubic').squeeze(0)
            else:
                mono_depth = None
                print("Depth_path Not Exist", depth_path)


            name_idx = frame["file_path"].replace("frame","").replace("Round-RI_","").split("_")

            if len(name_idx) < 3 or "frame0" in frame["file_path"]:
                time = [0., 0.]
            elif len(name_idx) < 7:
                if "U" in frame["file_path"]:
                    time = [0, int(name_idx[-1])]
                elif "D" in frame["file_path"]:
                    time = [0, 0 -int(name_idx[-1])]
                elif "L" in frame["file_path"] or "f90" in frame["file_path"]:
                    time = [int(name_idx[-1]), 0]
                elif "R" in frame["file_path"] or "90" in frame["file_path"]:
                    time = [0 -int(name_idx[-1]), 0]
                else:
                    print(frame["file_path"], " not found")
                    print(name_idx)
            else:
                if "U" == name_idx[-4]:
                    if "U" in name_idx[-2]:
                        time = [0, int(name_idx[-3]) + int(name_idx[-1])]
                    elif "D" in name_idx[-2]:
                        time = [0, int(name_idx[-3]) - int(name_idx[-1])]
                    elif "L" in name_idx[-2] or "f90" in name_idx[-2]:
                        time = [int(name_idx[-1]), int(name_idx[-3])]
                    elif "R" in name_idx[-2] or "90" in name_idx[-2]:
                        time = [0 - int(name_idx[-1]), int(name_idx[-3])]
                    else:
                        print(frame["file_path"], " not found")
                        print(name_idx)
                elif "D" == name_idx[-4]:
                    # time = [0, 0 -int(name_idx[-1])]
                    if "U" in name_idx[-2]:
                        time = [0, 0 -int(name_idx[-3]) + int(name_idx[-1])]
                    elif "D" in name_idx[-2]:
                        time = [0, 0 -int(name_idx[-3]) - int(name_idx[-1])]
                    elif "L" in name_idx[-2] or "f90" in name_idx[-2]:
                        time = [int(name_idx[-1]), 0 -int(name_idx[-3])]
                    elif "R" in name_idx[-2] or "90" in name_idx[-2]:
                        time = [0 -int(name_idx[-1]), 0 -int(name_idx[-3])]
                    else:
                        print(frame["file_path"], " not found")
                        print(name_idx)
                elif "L" == name_idx[-4] or "f90" in name_idx[-4]:
                    # time = [int(name_idx[-1], 0)]
                    if "U" in name_idx[-2]:
                        time = [int(name_idx[-3]) , int(name_idx[-1])]
                    elif "D" in name_idx[-2]:
                        time = [int(name_idx[-3]),  0 -int(name_idx[-1])]
                    elif "L" in name_idx[-2] or "f90" in name_idx[-2]:
                        time = [int(name_idx[-3]) + int(name_idx[-1]), 0]
                    elif "R" in name_idx[-2] or "90" in name_idx[-2]:
                        time = [int(name_idx[-3]) - int(name_idx[-1]), 0]
                    else:
                        print(frame["file_path"], " not found")
                        print(name_idx)
                elif "R" == name_idx[-4] or "90" in name_idx[-4]:
                    # time = [0 -int(name_idx[-1], 0)]
                    if "U" in name_idx[-2]:
                        time = [0 -int(name_idx[-3]) , int(name_idx[-1])]
                    elif "D" in name_idx[-2]:
                        time = [0 -int(name_idx[-3]),  0 -int(name_idx[-1])]
                    elif "L" in name_idx[-2] or "f90" in name_idx[-2]:
                        time = [0 -int(name_idx[-3]) + int(name_idx[-1]), 0]
                    elif "R" in name_idx[-2] or "90" in name_idx[-2]:
                        time = [0 -int(name_idx[-3]) - int(name_idx[-1]), 0]
                    else:
                        print(frame["file_path"], " not found")
                        print(name_idx)
                else:
                    print(frame["file_path"], " not found")
                    print(name_idx)
            
            norm_time = [(time[0] + num_frames*2.0)/(num_frames*4.0), (time[1] + num_frames*2.0)/(num_frames*4.0)]

            time = norm_time

            if dust3r_name and dust3r_name in cam_name.split("/")[-1]:
                dusr3r_caminfo = CameraInfo(uid=idx*2, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            time=[0., 0.], mask=None)
                print("dusr3r_caminfo image name is:", dusr3r_caminfo.image_name)

            if mono_depth is not None:
                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image, mono_depth=mono_depth,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            time = time, mask=None))
            else:
                cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1],
                            time = time, mask=None))

    nerf_normalization = getNerfppNorm(cam_infos)
    if ply_name != "":
        ply_path = os.path.join(path, ply_name)
    else: 
        ply_path = os.path.join(path, transformsfile.replace('json', 'ply'))

    if "dust3r" in ply_path or "input_view0" in ply_path :
        import open3d as o3d
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

        point_cloud = o3d.io.read_point_cloud(ply_path)

        coords = np.array(point_cloud.points)
        rgb = np.array(point_cloud.colors)

        point_cloud = o3d.t.geometry.PointCloud.from_legacy(point_cloud)
        
        pcd = BasicPointCloud(points=coords * bound, colors=rgb, normals=np.zeros((coords.shape[0], 3)))
        print("load ply with points:", coords.shape)
    else:
        print("Not found ", ply_path)


    all_poses = []
    # all_poses += create_spheric_poses_along_y()
    all_poses += render_path_spiral_default(N=360) # This pose is used to render the pose of the final 3dgs.
    cv_poses = pose_opengl_to_opencv(all_poses)
    cv_w2cs = [np.linalg.inv(c2w) for c2w in cv_poses]
    render_num = len(cv_poses)
    if dusr3r_caminfo is not None:
        render_infos=[CameraInfo(uid=i, R=cv_w2cs[i][:3,:3].transpose(), T=cv_w2cs[i][:3, 3], 
                FovY=dusr3r_caminfo.FovY, FovX=dusr3r_caminfo.FovX, image=dusr3r_caminfo.image, 
                image_path=dusr3r_caminfo.image_path, image_name=dusr3r_caminfo.image_name, 
                width=dusr3r_caminfo.width, height=dusr3r_caminfo.height,
                time=dusr3r_caminfo.time, mask=dusr3r_caminfo.mask) for i in range(render_num)]

        # now transformt all the pose in the cv_poses to dust3r coordinates
        _, _, transformation_matrix_render = align_cameras(render_infos[0], dusr3r_caminfo)

        new_render_cam_infos = []
        for info in render_infos:
            w2c = getWorld2View(info.R, info.T)
            new_w2c = w2c @ transformation_matrix_render
            # new_w2c[0, 3] += 0.6
            # new_w2c[1, 3] -= 0.3
            new_R = new_w2c[:3,:3].transpose()
            new_T = new_w2c[:3, 3]
            new_cam_info = CameraInfo(uid=info.uid, R=new_R, T=new_T, 
                                    FovY=info.FovY, FovX=info.FovX, image=info.image, 
                                    image_path=info.image_path, image_name=info.image_name, 
                                    width=info.width, height=info.height,
                                    time=info.time, mask=info.mask) # maybe use the test K is right
            new_render_cam_infos.append(new_cam_info)


    # scene_info = SceneInfo(point_cloud=pcd,
    #                        train_cameras=cam_infos,
    #                        test_cameras=new_test_cam_infos,
    #                        render_cameras=new_render_cam_infos,
    #                        nerf_normalization=nerf_normalization,
    #                        ply_path=ply_path)
    # if len(new_test_cam_infos) > 0:
    #     # cam_infos; new_test_cam_infos
    #     scene_info = SceneInfo(point_cloud=pcd,
    #                             train_cameras=cam_infos,
    #                             test_cameras=new_test_cam_infos,
    #                             render_cameras=new_render_cam_infos,
    #                             video_cameras=cam_infos,
    #                             maxtime=300,
    #                             nerf_normalization=nerf_normalization,
    #                             ply_path=ply_path)
    # else:
    scene_info = SceneInfo(point_cloud=pcd,
                            train_cameras=cam_infos,
                            test_cameras=cam_infos, # No test_cameras in single image to 3D
                            render_cameras=new_render_cam_infos,
                            video_cameras=cam_infos,
                            maxtime=300,
                            nerf_normalization=nerf_normalization,
                            ply_path=ply_path)
    # if new_test_cam_infos = []:
    #     scene_info = SceneInfo(point_cloud=pcd,
    #                         train_cameras=cam_infos,
    #                         test_cameras=cam_infos,
    #                         video_cameras=cam_infos,
    #                         maxtime=300,
    #                         nerf_normalization=nerf_normalization,
    #                         ply_path=ply_path)
    # else:
    #     scene_info = SceneInfo(point_cloud=pcd,
    #                         train_cameras=cam_infos,
    #                         test_cameras=new_test_cam_infos,
    #                         video_cameras=cam_infos,
    #                         maxtime=300,
    #                         nerf_normalization=nerf_normalization,
    #                         ply_path=ply_path)
    # scene_info = SceneInfo(point_cloud=pcd,
    #                         train_cameras=cam_infos,
    #                         test_cameras=cam_infos,
    #                         video_cameras=cam_infos,
    #                         maxtime=300,
    #                         nerf_normalization=nerf_normalization,
    #                         ply_path=ply_path)
    
    return scene_info, point_cloud
    
def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model in ["SIMPLE_PINHOLE", "SIMPLE_RADIAL"]:
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model == "OPENCV":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)
        image = PILtoTorch(image,None)
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height,
                              time = float(idx/len(cam_extrinsics)), mask=None) # default by monocular settings.
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    # breakpoint()
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    # breakpoint()
    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    
    try:
        pcd = fetchPly(ply_path)
        
    except:
        pcd = None
    
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=train_cam_infos,
                           maxtime=0,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def generateCamerasFromTransforms(path, template_transformsfile, extension, maxtime):
    trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

    rot_phi = lambda phi : torch.Tensor([
        [1,0,0,0],
        [0,np.cos(phi),-np.sin(phi),0],
        [0,np.sin(phi), np.cos(phi),0],
        [0,0,0,1]]).float()

    rot_theta = lambda th : torch.Tensor([
        [np.cos(th),0,-np.sin(th),0],
        [0,1,0,0],
        [np.sin(th),0, np.cos(th),0],
        [0,0,0,1]]).float()
    def pose_spherical(theta, phi, radius):
        c2w = trans_t(radius)
        c2w = rot_phi(phi/180.*np.pi) @ c2w
        c2w = rot_theta(theta/180.*np.pi) @ c2w
        c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
        return c2w
    cam_infos = []
    # generate render poses and times
    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,160+1)[:-1]], 0)
    render_times = torch.linspace(0,maxtime,render_poses.shape[0])
    with open(os.path.join(path, template_transformsfile)) as json_file:
        template_json = json.load(json_file)
        try:
            fovx = template_json["camera_angle_x"]
        except:
            fovx = focal2fov(template_json["fl_x"], template_json['w'])
    print("hello!!!!")
    # breakpoint()
    # load a single image to get image info.
    for idx, frame in enumerate(template_json["frames"]):
        cam_name = os.path.join(path, frame["file_path"] + extension)
        image_path = os.path.join(path, cam_name)
        image_name = Path(cam_name).stem
        image = Image.open(image_path)
        im_data = np.array(image.convert("RGBA"))
        image = PILtoTorch(image,(800,800))
        break
    # format information
    for idx, (time, poses) in enumerate(zip(render_times,render_poses)):
        time = time/maxtime
        matrix = np.linalg.inv(np.array(poses))
        R = -np.transpose(matrix[:3,:3])
        R[:,0] = -R[:,0]
        T = -matrix[:3, 3]
        fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
        FovY = fovy 
        FovX = fovx
        cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=None, image_name=None, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
    return cam_infos

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png", mapper = {}):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        try:
            fovx = contents["camera_angle_x"]
        except:
            fovx = focal2fov(contents['fl_x'],contents['w'])
        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)
            time = mapper[frame["time"]]
            matrix = np.linalg.inv(np.array(frame["transform_matrix"]))
            R = -np.transpose(matrix[:3,:3])
            R[:,0] = -R[:,0]
            T = -matrix[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")
            image = PILtoTorch(image,(800,800))
            fovy = focal2fov(fov2focal(fovx, image.shape[1]), image.shape[2])
            FovY = fovy 
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[1], height=image.shape[2],
                            time = time, mask=None))
            
    return cam_infos

def read_timeline(path):
    with open(os.path.join(path, "transforms_train.json")) as json_file:
        train_json = json.load(json_file)
    with open(os.path.join(path, "transforms_test.json")) as json_file:
        test_json = json.load(json_file)  
    time_line = [frame["time"] for frame in train_json["frames"]] + [frame["time"] for frame in test_json["frames"]]
    time_line = set(time_line)
    time_line = list(time_line)
    time_line.sort()
    timestamp_mapper = {}
    max_time_float = max(time_line)
    for index, time in enumerate(time_line):
        # timestamp_mapper[time] = index
        timestamp_mapper[time] = time/max_time_float

    return timestamp_mapper, max_time_float

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    timestamp_mapper, max_time = read_timeline(path)
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension, timestamp_mapper)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension, timestamp_mapper)
    print("Generating Video Transforms")
    video_cam_infos = generateCamerasFromTransforms(path, "transforms_train.json", extension, max_time)
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "fused.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 2000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))
    # storePly(ply_path, xyz, SH2RGB(shs) * 255)
    else:
        pcd = fetchPly(ply_path)
        # xyz = -np.array(pcd.points)
        # pcd = pcd._replace(points=xyz)


    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           video_cameras=video_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           maxtime=max_time
                           )
    return scene_info

def format_infos(dataset,split):
    # loading
    cameras = []
    image = dataset[0][0]
    if split == "train":
        for idx in tqdm(range(len(dataset))):
            image_path = None
            image_name = f"{idx}"
            time = dataset.image_times[idx]
            # matrix = np.linalg.inv(np.array(pose))
            R,T = dataset.load_pose(idx)
            FovX = focal2fov(dataset.focal[0], image.shape[1])
            FovY = focal2fov(dataset.focal[0], image.shape[2])
            cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                                image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                                time = time, mask=None))

    return cameras


def format_render_poses(poses,data_infos):
    cameras = []
    tensor_to_pil = transforms.ToPILImage()
    len_poses = len(poses)
    times = [i/len_poses for i in range(len_poses)]
    image = data_infos[0][0]
    for idx, p in tqdm(enumerate(poses)):
        # image = None
        image_path = None
        image_name = f"{idx}"
        time = times[idx]
        pose = np.eye(4)
        pose[:3,:] = p[:3,:]
        # matrix = np.linalg.inv(np.array(pose))
        R = pose[:3,:3]
        R = - R
        R[:,0] = -R[:,0]
        T = -pose[:3,3].dot(R)
        FovX = focal2fov(data_infos.focal[0], image.shape[2])
        FovY = focal2fov(data_infos.focal[0], image.shape[1])
        cameras.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.shape[2], height=image.shape[1],
                            time = time, mask=None))
    return cameras

def add_points(pointsclouds, xyz_min, xyz_max):
    add_points = (np.random.random((100000, 3)))* (xyz_max-xyz_min) + xyz_min
    add_points = add_points.astype(np.float32)
    addcolors = np.random.random((100000, 3)).astype(np.float32)
    addnormals = np.random.random((100000, 3)).astype(np.float32)
    # breakpoint()
    new_points = np.vstack([pointsclouds.points,add_points])
    new_colors = np.vstack([pointsclouds.colors,addcolors])
    new_normals = np.vstack([pointsclouds.normals,addnormals])
    pointsclouds=pointsclouds._replace(points=new_points)
    pointsclouds=pointsclouds._replace(colors=new_colors)
    pointsclouds=pointsclouds._replace(normals=new_normals)
    return pointsclouds
    # breakpoint()
    # new_

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True
    )
    return cam
def plot_camera_orientations(cam_list, xyz):
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    # ax2 = fig.add_subplot(122, projection='3d')
    # xyz = xyz[xyz[:,0]<1]
    threshold=2
    xyz = xyz[(xyz[:, 0] >= -threshold) & (xyz[:, 0] <= threshold) &
                         (xyz[:, 1] >= -threshold) & (xyz[:, 1] <= threshold) &
                         (xyz[:, 2] >= -threshold) & (xyz[:, 2] <= threshold)]

    ax.scatter(xyz[:,0],xyz[:,1],xyz[:,2],c='r',s=0.1)
    for cam in tqdm(cam_list):
        # 提取 R 和 T
        R = cam.R
        T = cam.T

        direction = R @ np.array([0, 0, 1])

        ax.quiver(T[0], T[1], T[2], direction[0], direction[1], direction[2], length=1)

    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.savefig("output.png")
    # breakpoint()

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo,
    "Dust3R" : readDust3RInfo,
}

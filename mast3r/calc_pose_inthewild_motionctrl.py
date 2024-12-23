import os
import cv2
import json
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dust3r.dust3r.viz import add_scene_cam, CAM_COLORS, OPENGL, pts3d_to_trimesh, cat_meshes
# from dust3r.inference import inference, load_model
from dust3r.dust3r.inference import inference, loss_of_one_batch
from mast3r.model import AsymmetricMASt3R
# from dust3r.utils.image import load_images
from dust3r.dust3r.utils.device import to_numpy
from dust3r.dust3r.image_pairs import make_pairs
from dust3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from scene.colmap_loader import rotmat2qvec
from argparse import ArgumentParser, Namespace
from dust3r.dust3r.utils.image import load_images, rgb
import cv2
import matplotlib
from PIL import Image
from scipy.spatial.transform import Rotation

# from transformers import pipeline
import PIL
import requests
from depth_anything_v2.dpt import DepthAnythingV2
import time
from glob import *
import imageio.v2 as iio

# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
UINT16_MAX = 65535

def to01(arr):
    return (arr - arr.min()) / (arr.max() - arr.min())


def _convert_scene_output_to_glb(outdir, imgs, pts3d, mask, focals, cams2world, cam_size=0.05,
                                 cam_color=None, as_pointcloud=False,
                                 transparent_cams=False, silent=False, data_path=None):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    #assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        # mesh.export("./test_mesh_foryc_florwer.ply")
        mesh.export(os.path.join(data_path, f'dust3r_27_fxfy_mesh_zero.ply'))
        scene.add_geometry(mesh)

    # add each camera
    for i, pose_c2w in enumerate(cams2world):
        if isinstance(cam_color, list):
            camera_edge_color = cam_color[i]
        else:
            camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
        # add_scene_cam(scene, pose_c2w, camera_edge_color,
        #               None if transparent_cams else imgs[i], focals,
        #               imsize=imgs[i].shape[1::-1], screen_width=cam_size)
        #print(str(i) + "  pose_c2w  "+ str(pose_c2w))
        add_scene_cam(scene, pose_c2w, camera_edge_color,
                      None if transparent_cams else imgs[i], focals[i],
                      imsize=imgs[i].shape[1::-1], screen_width=cam_size)
    
    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler('y', np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    outfile = os.path.join(outdir, 'scene_zero.glb')
    if not silent:
        print('(exporting 3D scene to', outfile, ')')
    scene.export(file_obj=outfile)
    return outfile


def get_3D_model_from_scene(outdir, silent, scene, min_conf_thr=3, as_pointcloud=True, mask_sky=False,
                            clean_depth=True, transparent_cams=False, cam_size=0.05, data_path=None):
    """
    extract 3D_model (glb file) from a reconstructed scene
    """
    if scene is None:
        return None
    # post processes
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    # get optimized values from scene
    rgbimg = scene.imgs
    focals = scene.get_focals().cpu()
    #focals = scene.get_focals()
    cams2world = scene.get_im_poses().cpu()
    # 3D pointcloud from depthmap, poses and intrinsics
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())
    return _convert_scene_output_to_glb(outdir, rgbimg, pts3d, msk, focals, cams2world, as_pointcloud=as_pointcloud,
                                        transparent_cams=transparent_cams, cam_size=cam_size, silent=silent, data_path=data_path)

def qvec2rvec(q):
    w, x, y, z = q
    theta = 2 * np.arccos(w)
    sin_theta_over_two = np.sin(theta / 2)
    if sin_theta_over_two > 0:
        vx = x / sin_theta_over_two
        vy = y / sin_theta_over_two
        vz = z / sin_theta_over_two
        return theta * np.array([vx, vy, vz])
    else:
        print('zeros')
        return np.array([0, 0, 0])

# def save_raw_depth(depth, fpath="raw.png", width=1024, height=768):
#     if isinstance(depth, torch.Tensor):
#         depth = depth.squeeze().cpu().numpy()
    
#     assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
#     assert depth.ndim == 2, "Depth must be 2D"
#     depth = depth * 1000  # scale for 16-bit png
#     depth = depth.astype(np.uint32)
#     depth = Image.fromarray(depth)
#     newsize = (width, height)
#     depth = depth.resize(newsize, Image.LANCZOS)   
#     depth.save(fpath)
def save_raw_depth(depth, fpath="raw.png"):
    if isinstance(depth, torch.Tensor):
        depth = depth.squeeze().cpu().numpy()
    
    if isinstance(depth, PIL.Image.Image):
        #print("type(depth) ", type(depth))
        depth = np.array(depth) / 255.0
    
    assert isinstance(depth, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert depth.ndim == 2, "Depth must be 2D"
    # print("depth.shape ", depth.shape)
    # print("depth.range ", depth.min(), depth.max())
    depth = depth * 1000  # scale for 16-bit png
    depth = depth.astype(np.uint16)
    depth = Image.fromarray(depth)
    depth.save(fpath)

def save_raw_cmap(cmap, fpath="raw.png", width=1024, height=768):
    if isinstance(cmap, torch.Tensor):
        cmap = cmap.squeeze().cpu().numpy()
    
    assert isinstance(cmap, np.ndarray), "Depth must be a torch tensor or numpy array"
    assert cmap.ndim == 2, "Depth must be 2D"
    cmap = cmap * 1000  # scale for 16-bit png
    cmap = cmap.astype(np.uint32)
    cmap = Image.fromarray(cmap)
    newsize = (width, height)
    cmap = cmap.resize(newsize, Image.LANCZOS)  
    cmap.save(fpath)

# hamburger dog6 dog4 desk1 cup1 beach_house_2 beach_house_1 361 293 277 216 215 166 098 095 053

def colorize(value, vmin=None, vmax=None, cmap='gray_r', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img


def check_already(file_name, image_root):
    # print("image_root ", image_root, " file_name ", file_name)
    if (os.path.exists(os.path.join(image_root,file_name+'_D')) and os.path.exists(os.path.join(image_root,file_name+'_U')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90')) and os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_D_frame15_D')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90_frame15_D')) and os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_D')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_U_frame15_U')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90_frame15_U')) and os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_U')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_D_frame15_Round-RI_90')) and os.path.exists(os.path.join(image_root,file_name+'_U_frame15_Round-RI_90')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_90_frame15_Round-RI_90')) and
        
        os.path.exists(os.path.join(image_root,file_name+'_D_frame15_Round-RI_f90')) and os.path.exists(os.path.join(image_root,file_name+'_U_frame15_Round-RI_f90')) and
        os.path.exists(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_Round-RI_f90')) ) and (

        len(os.listdir(os.path.join(image_root,file_name+'_D')))>=14 and len(os.listdir(os.path.join(image_root,file_name+'_U')))>=14 and
        len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_90')))>=14 and len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_f90')))>=14 and
        
        len(os.listdir(os.path.join(image_root,file_name+'_D_frame15_D')))>=14 and
        len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_90_frame15_D')))>=14 and len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_D')))>=14 and
        
        len(os.listdir(os.path.join(image_root,file_name+'_U_frame15_U')))>=14 and
        len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_90_frame15_U')))>=14 and len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_U')))>=14 and
        
        len(os.listdir(os.path.join(image_root,file_name+'_D_frame15_Round-RI_90')))>=14 and len(os.listdir(os.path.join(image_root,file_name+'_U_frame15_Round-RI_90')))>=14 and
        len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_90_frame15_Round-RI_90')))>=14 and
        
        len(os.listdir(os.path.join(image_root,file_name+'_D_frame15_Round-RI_f90')))>=14 and len(os.listdir(os.path.join(image_root,file_name+'_U_frame15_Round-RI_f90')))>=14 and
        len(os.listdir(os.path.join(image_root,file_name+'_Round-RI_f90_frame15_Round-RI_f90')))>=14 ):
        # print(f"**This {file_name} already down. skip")
        return True
    else:
        return False

def align_monodepth_with_metric_depth(
    file_name: str,
    metric_depth_dir: str,
    input_monodepth_dir: str,
    output_monodepth_dir: str,
    matching_pattern: str = "*",
):
    print(
        f"Aligning monodepth in {input_monodepth_dir} with metric depth in {metric_depth_dir}"
    )
    mono_paths = sorted(glob(f"{input_monodepth_dir}/{matching_pattern}"))
    img_files = []

    for p in mono_paths:
        if (not "_frame0" in p) and \
            (not "D_frame15_U" in p) and (not "U_frame15_D" in p) \
            and (not "_Round-RI_f90_frame15_Round-RI_90" in p) \
            and (not "_Round-RI_90_frame15_Round-RI_f90" in p) \
            or (file_name + "_frame0" in p):
            img_files.append(os.path.basename(p))

    os.makedirs(output_monodepth_dir, exist_ok=True)
    if len(os.listdir(output_monodepth_dir)) == len(img_files):
        print(f"Founds {len(img_files)} files in {output_monodepth_dir}, skipping")
        return

    for f in tqdm(img_files):
        imname = os.path.splitext(f)[0]
        metric_path = os.path.join(metric_depth_dir, imname + ".npy")
        mono_path = os.path.join(input_monodepth_dir, imname + ".png")

        mono_disp_map = iio.imread(mono_path) / UINT16_MAX
        metric_disp_map = np.load(metric_path)


        metric_disp_map = torch.nn.functional.interpolate(
            torch.tensor(metric_disp_map).unsqueeze(0).unsqueeze(0), size=(mono_disp_map.shape[0],mono_disp_map.shape[1]), mode="bicubic", align_corners=False
        )
        # print("metric_disp_map.shape ", metric_disp_map.shape)
        # print("mono_disp_map.shape ", mono_disp_map.shape)

        metric_disp_map = metric_disp_map.squeeze(0).squeeze(0).cpu().numpy()

        ms_colmap_disp = metric_disp_map - np.median(metric_disp_map) + 1e-8
        ms_mono_disp = mono_disp_map - np.median(mono_disp_map) + 1e-8

        

        scale = np.median(ms_colmap_disp / ms_mono_disp)
        shift = np.median(metric_disp_map - scale * mono_disp_map)

        aligned_disp = scale * mono_disp_map + shift

        min_thre = min(1e-6, np.quantile(aligned_disp, 0.01))
        # set depth values that are too small to invalid (0)
        aligned_disp[aligned_disp < min_thre] = 0.0
        out_file = os.path.join(output_monodepth_dir, imname + ".npy")
        np.save(out_file, aligned_disp)

        colored_depth = colorize(aligned_disp, cmap='magma_r')
        Image.fromarray(colored_depth).save(out_file.replace(".npy","_visual.png"))


name_ext_all = [
    "_Round-RI_f90",
    "_Round-RI_90",
    "_D",
    "_U",
    "_Round-RI_90_frame15_Round-RI_f90",
    "_Round-RI_90_frame15_Round-RI_90",
    "_Round-RI_90_frame15_D",
    "_Round-RI_90_frame15_U",
    "_Round-RI_f90_frame15_Round-RI_f90",
    "_Round-RI_f90_frame15_Round-RI_90",
    "_Round-RI_f90_frame15_D",
    "_Round-RI_f90_frame15_U",
    "_D_frame15_Round-RI_f90",
    "_D_frame15_Round-RI_90",
    "_D_frame15_D",
    "_D_frame15_U",
    "_U_frame15_Round-RI_f90",
    "_U_frame15_Round-RI_90",
    "_U_frame15_D",
    "_U_frame15_U",
]

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")

    # parser.add_argument('--file_name', type=str, default="flower", help="The input file names")
    parser.add_argument('--gpu', type=int, default=0, help="The gpu")
    parser.add_argument('--input-size', type=int, default=512)
    parser.add_argument('--encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--pred-only', dest='pred_only', default=True, help='only display the prediction')
    parser.add_argument("--matching_pattern", type=str, default="*")
    parser.add_argument('--height', type=int, default=768, help="The height of input images")
    parser.add_argument('--width', type=int, default=1024, help="The width of input images")
    parser.add_argument('--cache_dir', type=str, default="output", help="The cache dictionary")
    parser.add_argument('--input_file', type=str, default='input_images/testimg001.png', help="The extention of input images")
    parser.add_argument('--ext', type=str, default='.png', help="The extention of input images")
    parser.add_argument('--ckpt_mast3r', type=str, default='checkpoints/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric.pth', help="The mast3r ckpt")
    parser.add_argument('--ckpt_depthanythingv2', type=str, default='checkpoints/depth_anything_v2_vitl.pth', help="The depth anything v2 ckpt")
    
    args = parser.parse_args()

    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


    loaded_scene = []

    model_path = args.ckpt_mast3r
    device = 'cuda'
    batch_size = 1
    schedule = 'cosine'
    lr = 0.01
    niter = 400
    scene_graph_type = "oneref"
    scene_graph_type = scene_graph_type + "-" + str(0)
    # scene_graph_type='motionctrl'
    rescale = 10.
    
    extention_type = args.ext
    root_dir = args.cache_dir

    load_size = 448 #load_size must can be divided by 64. If you meet a bug "Killed" reduce load_size to save RAM
    raw_height = args.height
    raw_width = args.width
    model = AsymmetricMASt3R.from_pretrained(model_path).to(f'cuda:{args.gpu}')

    image_input = [os.path.join(args.input_file.split('/')[-1].split('.')[0])]
    
    for cur_file_name in image_input:

        if not check_already(cur_file_name, os.path.join(root_dir, cur_file_name, 'motionctrl')):
            print("**Not complete motionctrl generation", os.path.join(root_dir, cur_file_name, 'motionctrl'))
            continue

        if  (os.path.exists(os.path.join(root_dir, cur_file_name, "depth_anything_v2_align")) and \
            len(os.listdir(os.path.join(root_dir, cur_file_name, "depth_anything_v2_align")))>=240)  :
            print("**Already have down calc pose", os.path.join(root_dir, cur_file_name))
            continue

        if os.path.exists(os.path.join(root_dir, cur_file_name, f'mast3r_allpoint.ply')) and \
            os.path.exists(os.path.join(root_dir, cur_file_name,"depth_dust3r")):
            print("**Already have down mast3r, directly_calc_dptv2 ", cur_file_name)
            directly_calc_dptv2 = True
        else:
            directly_calc_dptv2 = False
            
        
        file_name_all = [cur_file_name+name_ext for name_ext in name_ext_all]
        loaded_images = []
        real_img_names = []
        key_img_dict = {cur_file_name+"_frame0.png": 0}
        real_img_dict = {}
        
        input_image = os.path.join(root_dir, cur_file_name, 'motionctrl', cur_file_name+extention_type)

        img = Image.open(input_image).convert('RGB')
        W1, H1 = img.size

        img = img.resize((raw_width,raw_height))
        img.save(os.path.join(root_dir, cur_file_name, 'motionctrl', cur_file_name + "_frame0.png"))

        loaded_images.append(os.path.join(root_dir, cur_file_name, 'motionctrl', cur_file_name + "_frame0.png"))
        real_img_names.append(cur_file_name+"_frame0.png")

        cnt = 1
        if not directly_calc_dptv2:
            for file_name in file_name_all:
                image_path = os.path.join(root_dir, cur_file_name, 'motionctrl', file_name)
                image_names = os.listdir(image_path)

                for idx, image_name in enumerate(image_names):
                    if ("_frame0" in image_name) or not ".png" in image_name:
                        continue

                    if ("_Round-RI_f90_frame15_Round-RI_90_" in image_name) or ("_Round-RI_90_frame15_Round-RI_f90_" in image_name) or ("_U_frame15_D_" in image_name) or ("_D_frame15_U_" in image_name) \
                       or ("_L_frame15_R_" in image_name) or ("_R_frame15_L_" in image_name) or ("_U_frame15_D_" in image_name) or ("_D_frame15_U_" in image_name):
                        continue

                    loaded_images.append(os.path.join(image_path, image_name))
                    real_img_names.append(file_name + "/" + image_name)


            loaded_images = load_images(loaded_images, size=load_size)
            image_names = real_img_names
            sparse_num = len(real_img_names)

            pairs = make_pairs(loaded_images, scene_graph=scene_graph_type, prefilter=None, symmetrize=True)
            # pairs = make_pairs(loaded_images, scene_graph=scene_graph_type, prefilter=None, symmetrize=True, real_img_dict=None)
            output = inference(pairs, model, device, batch_size=batch_size)
            scene = global_aligner(output, device=device, mode=GlobalAlignerMode.PointCloudOptimizer)

            loss = scene.compute_global_alignment(init="mst", niter=niter, schedule=schedule, lr=lr)

            imgs = scene.imgs
            focals = scene.get_focals()
            poses = scene.get_im_poses()
            pts3d = scene.get_pts3d()

            mask_mesh = to_numpy(scene.get_masks())
            pts3d_mesh = to_numpy(scene.get_pts3d())
            
            #import pdb;pdb.set_trace()

            vertices = np.concatenate([p[m] for p, m in zip(pts3d_mesh, mask_mesh)])
            colors = np.concatenate([p[m] for p, m in zip(imgs, mask_mesh)])
            center = np.mean(vertices, axis=0)
            vertices -= center
            max_bbox = np.abs(vertices).max()
            vertices = vertices / max_bbox * rescale    #被rescale了。

            #center 和 max_bbox不能变。
            vertices_input = pts3d_mesh[0][mask_mesh[0]]
            colors_input = imgs[0][mask_mesh[0]]
            vertices_input -= center
            vertices_input = vertices_input / max_bbox * rescale    #被rescale了。

            cloud = trimesh.PointCloud(vertices.reshape(-1, 3), colors.reshape(-1, 3))
            cloud.export(os.path.join(root_dir, cur_file_name, f'mast3r_allpoint.ply'))

            cloud_input = trimesh.PointCloud(vertices_input.reshape(-1, 3), colors_input.reshape(-1, 3))
            cloud_input.export(os.path.join(root_dir, cur_file_name, f'input_view0.ply'))

            poses[:, :3, 3] = (poses[:, :3, 3] - torch.tensor(center).to(device)) / max_bbox * rescale


            cameras = []
            height, width = imgs[0].shape[:2]

            for i in range(sparse_num):
                cameras.append({
                    'id': i,
                    'file_path': 'motionctrl/'+image_names[i].split(".")[0],
                    'width': width,
                    'height': height,
                    'transform_matrix': poses[i].tolist(),
                    'position': poses[i, :3, 3].tolist(),
                    'rotation': poses[i, :3, :3].tolist(),
                    'fy': focals[i].item() ,
                    'fx': focals[i].item() ,
                })

            with open(os.path.join(root_dir, cur_file_name, f'mast3r_camerapose.json'), 'w') as f:
                json.dump(cameras, f, indent=4)
            
            silent = False
            mask_sky = False
            cam_size=0.01
            transparent_cams=False
            as_pointcloud=False
                          
            depths = to_numpy(scene.get_depthmaps())
            confs = to_numpy([c for c in scene.im_conf])

            depths = depths / max_bbox * rescale

            confs_max = max([d.max() for d in confs])
            confs = [rgb(d/confs_max) for d in confs]

            depths = np.array(depths)
            confs = np.array(confs)

            if not os.path.exists(os.path.join(root_dir, cur_file_name,"depth_dust3r")):
                os.mkdir(os.path.join(root_dir, cur_file_name,"depth_dust3r"))
            if not os.path.exists(os.path.join(root_dir, cur_file_name,"confs_dust3r")):
                os.mkdir(os.path.join(root_dir, cur_file_name,"confs_dust3r"))
        

            for idx, image_name in enumerate(image_names):
                image_name_cur = image_name.split("/")[-1]
                np.save(os.path.join(root_dir, cur_file_name,"depth_dust3r",image_name_cur).replace(".png", ".npy"), depths[idx])

                colored_depth = colorize(depths[idx], cmap='magma_r')
                colored_cmap = colorize(confs[idx], cmap='magma_r')
                Image.fromarray(colored_depth).save(os.path.join(root_dir, cur_file_name,"depth_dust3r",image_name_cur).replace(".png","_color.png"))
                Image.fromarray(colored_cmap).save(os.path.join(root_dir, cur_file_name,"confs_dust3r",image_name_cur).replace(".png","_color.png"))

        
        depth_anything = DepthAnythingV2(**model_configs[args.encoder])
        
        depth_anything.load_state_dict(torch.load(args.ckpt_depthanythingv2, map_location='cuda'))
        depth_anything = depth_anything.to(DEVICE).eval()
        cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        print(f"Start Mono Depth Estimation. Load to {DEVICE} complete")

        output_dir = os.path.join(root_dir, cur_file_name, "depth_v2")
        json_file = os.path.join(root_dir, cur_file_name, "mast3r_camerapose.json")

        k = 0
                
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        for folder_name in os.listdir(os.path.join(root_dir,cur_file_name, 'motionctrl')):
            if "_frame0.png" in folder_name:
                print(f"Progress {k+1} : {os.path.join(root_dir, cur_file_name, 'motionctrl', folder_name)}")
                raw_image = cv2.imread(os.path.join(root_dir, cur_file_name, 'motionctrl', folder_name))
                
                depth = depth_anything.infer_image(raw_image, args.input_size)
                
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                
                
                if args.pred_only:
                    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(folder_name))[0] + '.png'), depth)
                else:
                    split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                    combined_result = cv2.hconcat([raw_image, split_region, depth])
                    
                    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(folder_name))[0] + '.png'), combined_result)
                
                k += 1

            if not os.path.isdir(os.path.join(root_dir, cur_file_name, 'motionctrl', folder_name)) or not cur_file_name in folder_name:
                continue

            for item in os.listdir(os.path.join(root_dir, cur_file_name, 'motionctrl', folder_name)):
                if not folder_name in item:
                    continue
                
                print(f"Progress {k+1} : {os.path.join(root_dir, cur_file_name, 'motionctrl', folder_name, item)}")
                # print(f'Progress {k+1}/{len(filenames)}: {filename}')
            
                raw_image = cv2.imread(os.path.join(root_dir, cur_file_name, 'motionctrl', folder_name, item))
                
                depth = depth_anything.infer_image(raw_image, args.input_size)
                
                depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
                depth = depth.astype(np.uint8)
                
                
                if args.pred_only:
                    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(item))[0] + '.png'), depth)
                else:
                    split_region = np.ones((raw_image.shape[0], 50, 3), dtype=np.uint8) * 255
                    combined_result = cv2.hconcat([raw_image, split_region, depth])
                    
                    cv2.imwrite(os.path.join(output_dir, os.path.splitext(os.path.basename(item))[0] + '.png'), combined_result)
                
                k += 1
        
        align_monodepth_with_metric_depth(
            cur_file_name,
            metric_depth_dir=os.path.join(root_dir, cur_file_name,"depth_dust3r"),
            input_monodepth_dir=os.path.join(root_dir, cur_file_name, "depth_v2"),
            output_monodepth_dir=os.path.join(root_dir, cur_file_name, "depth_anything_v2_align"),
            matching_pattern=args.matching_pattern,
        )

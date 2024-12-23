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

from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1024:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1024 pixels width), rescaling to 1024.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1024
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    # print("Camera_utils ", cam_info.mono_depth)
    
    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device, 
                  time = cam_info.time, depth=cam_info.mono_depth
)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry


def convert_opencv_to_opengl(opencv_R, opencv_t):
    # Step 1: Create the transformation matrix in OpenCV's format
    transformation_cv = np.eye(4)
    transformation_cv[:3, :3] = opencv_R
    transformation_cv[:3, 3] = opencv_t.flatten()

    # Step 2: Invert to convert to camera-to-world matrix
    transformation_cv = np.linalg.inv(transformation_cv) 

    # Step 3: Adjust for coordinate system differences between OpenCV and OpenGL
    # Flip Y and Z axes
    convert_cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],  # Flip Y
        [0,  0, -1, 0],  # Flip Z
        [0,  0,  0, 1]
    ])

    # Step 4: Apply the coordinate system conversion
    transformation_gl = transformation_cv @ convert_cv_to_gl
    return transformation_gl


def opencv_c2w_to_opengl_c2w(transformation_cv):
    # Step 3: Adjust for coordinate system differences between OpenCV and OpenGL
    # Flip Y and Z axes
    convert_cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],  # Flip Y
        [0,  0, -1, 0],  # Flip Z
        [0,  0,  0, 1]
    ])

    # Step 4: Apply the coordinate system conversion
    transformation_gl = transformation_cv @ convert_cv_to_gl
    return transformation_gl

def pts3d_to_trimesh(img, pts3d, valid=None):
    H, W, THREE = img.shape
    assert THREE == 3
    assert img.shape == pts3d.shape

    vertices = pts3d.reshape(-1, 3)

    # make squares: each pixel == 2 triangles
    idx = np.arange(len(vertices)).reshape(H, W)
    idx1 = idx[:-1, :-1].ravel()  # top-left corner
    idx2 = idx[:-1, +1:].ravel()  # right-left corner
    idx3 = idx[+1:, :-1].ravel()  # bottom-left corner
    idx4 = idx[+1:, +1:].ravel()  # bottom-right corner
    faces = np.concatenate((
        np.c_[idx1, idx2, idx3],
        np.c_[idx3, idx2, idx1],  # same triangle, but backward (cheap solution to cancel face culling)
        np.c_[idx2, idx3, idx4],
        np.c_[idx4, idx3, idx2],  # same triangle, but backward (cheap solution to cancel face culling)
    ), axis=0)

    # prepare triangle colors
    face_colors = np.concatenate((
        img[:-1, :-1].reshape(-1, 3),
        img[:-1, :-1].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3),
        img[+1:, +1:].reshape(-1, 3)
    ), axis=0)

    # remove invalid faces
    if valid is not None:
        assert valid.shape == (H, W)
        valid_idxs = valid.ravel()
        valid_faces = valid_idxs[faces].all(axis=-1)
        faces = faces[valid_faces]
        face_colors = face_colors[valid_faces]

    assert len(faces) == len(face_colors)
    return dict(vertices=vertices, face_colors=face_colors, faces=faces)


def render_depth_from_scene(trimesh_mesh, train_cameras):
    import trimesh
    import pyrender
    print("re-rendering depths ...")
    pr_mesh = pyrender.Mesh.from_trimesh(trimesh_mesh)
    scene = pyrender.Scene()
    scene.add(pr_mesh)

    for cam in train_cameras:
        print(cam)
        fx, fy, cx, cy = fov2focal(cam.FoVx, cam.image_width), fov2focal(cam.FoV, cam.image_height), cam.image_width/2, cam.image_height/2

        # Setup camera
        camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.01, zfar=100.0)
        c2w_opencv = cam.world_view_transform.transpose(0, 1).inverse().detach().cpu().numpy()
        cam_pose =  opencv_c2w_to_opengl_c2w(c2w_opencv)
        
        scene.add(camera, pose=cam_pose)

        # Create an offscreen renderer
        renderer = pyrender.OffscreenRenderer(cam.image_width, cam.image_height)

        # Render the scene
        color, depth = renderer.render(scene)
    
        from PIL import Image
        color = Image.fromarray(color)
        color.save('color.png')
        depth = Image.fromarray(((depth - depth.min())/ (depth.max()-depth.min()) * 255.).astype(np.uint8))
        depth.save("depth.png")





# # Load a mesh
# mesh = trimesh.load('pancreas_001.ply')

# # Create a Pyrender mesh from the Trimesh mesh
# pr_mesh = pyrender.Mesh.from_trimesh(mesh)

# # Create a scene
# scene = pyrender.Scene()

# # Add the Pyrender mesh to the scene
# scene.add(pr_mesh)

# fx, fy, cx, cy = 525.0, 525.0, 319.5, 239.5

# # Setup camera
# camera = pyrender.IntrinsicsCamera(fx=fx, fy=fy, cx=cx, cy=cy, znear=0.1, zfar=100.0)
# cam_pose = np.array([
#     [1, 0, 0, 0],
#     [0, 1, 0, -1],
#     [0, 0, 1, 2],
#     [0, 0, 0, 1]
# ])
# scene.add(camera, pose=cam_pose)

# # Create an offscreen renderer
# renderer = pyrender.OffscreenRenderer(640, 480)

# # Render the scene
# color, depth = renderer.render(scene)




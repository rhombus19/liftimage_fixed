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
import math
import numpy as np
from typing import NamedTuple
# import camtools as ct
# import open3d as o3d

def gen_swing_path(init_pose=torch.eye(4), num_frames=10, r_x=0.8, r_y=0, r_z=-0.8):
    """
    Generates a sequence of transformation matrices simulating a swing-like motion in 3D space.

    Parameters:
    init_pose (torch.Tensor): Initial 4x4 transformation matrix representing the starting pose of the object.
                              Default is the identity matrix, indicating no initial translation or rotation.
    num_frames (int): The number of frames for which the path is generated, each frame having its own transformation matrix.
    r_x (float): Amplitude of the swing along the x-axis. Defines the extent of left-right motion.
    r_y (float): Amplitude of the swing along the y-axis. Defines the extent of up-down motion.
    r_z (float): Amplitude of the swing along the z-axis, with an offset that affects forward-backward motion and vertical position.

    Returns:
    List of torch.Tensor: A list of 4x4 transformation matrices for each frame, simulating the swing motion.
    """
    # Create a time vector from 0 to 1, evenly spaced over the number of frames
    t = torch.arange(num_frames) / (num_frames - 1)
    
    # Initialize the poses array by repeating the initial pose for each frame
    poses = init_pose.repeat(num_frames, 1, 1)

    # Initialize the swing transformation matrix, repeated across all frames
    swing = torch.eye(4).repeat(num_frames, 1, 1)
    
    # Set the translations for each axis based on the swing parameters and time vector
    swing[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)  # Sine wave for x-axis
    swing[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)  # Cosine wave for y-axis
    swing[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1)  # Cosine wave with offset for z-axis

    # Combine the initial pose with the swing transformations for each frame
    for i in range(num_frames):
        poses[i, :, :] = poses[i, :, :] @ swing[i, :, :]  # Matrix multiplication to apply transformations

    # Return the list of transformation matrices
    return list(poses.unbind())

def create_spheric_poses_along_y(n_poses=10, radius=10, factor=1):
    """
    Generates a list of camera poses arranged spherically, focusing on rotation along the y and x axes.

    Parameters:
    n_poses (int): Total number of poses to generate.

    Returns:
    list of torch.Tensor: A list of camera poses, each represented as a 4x4 transformation matrix.
    """

    # Nested function to calculate a spherical pose rotating along the y-axis
    def spheric_pose_y(phi, radius=radius):
        """
        Generates a transformation matrix for a pose on a sphere at a specific angle phi along the y-axis.

        Parameters:
        phi (float): Angle in radians to rotate around the y-axis.
        radius (float): Radius of the sphere from the center to the camera.

        Returns:
        torch.Tensor: A 4x4 transformation matrix representing the camera pose.
        """
        # Translation matrix that moves the camera along the y-axis
        trans_t = lambda t: np.array([
            [1, 0, 0, math.sin(2. * math.pi * t) * radius],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotation matrix about the y-axis
        rot_phi = lambda phi: np.array([
            [np.cos(phi), 0, np.sin(phi), 0],
            [0, 1, 0, 0],
            [-np.sin(phi), 0, np.cos(phi), 0],
            [0, 0, 0, 1]
        ])

        # Combine translation and rotation into a single camera-to-world matrix
        c2w = rot_phi(phi) @ trans_t(phi)
        c2w = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]) @ c2w
        c2w = torch.tensor(c2w).float()
        return c2w

    # Similar nested function for rotation along the x-axis
    def spheric_pose_x(phi, radius=radius, factor = factor):
        """
        Generates a transformation matrix for a pose on a sphere at a specific angle phi along the x-axis.

        Parameters:
        phi (float): Angle in radians to rotate around the x-axis.
        radius (float): Radius of the sphere from the center to the camera.

        Returns:
        torch.Tensor: A 4x4 transformation matrix representing the camera pose.
        """
        # Translation matrix that moves the camera along the x-axis
        trans_t = lambda t: np.array([
            [1, 0, 0, 0],
            [0, 1, 0, math.sin(2. * math.pi * t) * radius * -1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Rotation matrix about the x-axis
        rot_theta = lambda th: np.array([
            [1, 0, 0, 0],
            [0, np.cos(th), -np.sin(th), 0],
            [0, np.sin(th), np.cos(th), 0],
            [0, 0, 0, 1]
        ])

        # Combine translation and rotation into a single camera-to-world matrix
        c2w = rot_theta(phi) @ trans_t(phi)
        c2w = np.array([[1, 0, 0, 0],
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]]) @ c2w
        c2w = torch.tensor(c2w).float()
        return c2w

    spheric_poses = []

    # Initial swing path generation
    poses = gen_swing_path(num_frames=n_poses)
    spheric_poses += poses

    # Set parameters for spherical rotation
    y_angle = (1/16) * np.pi * factor
    x_angle = (1/16) * np.pi * factor
    x_radius = 0.1 * factor
    y_radius = 0.1 * factor

    # Generate poses by rotating left and right around the y-axis
    for th in np.linspace(0, -1 * y_angle, n_poses//2):
        spheric_poses += [spheric_pose_y(th, y_radius)]

    # Continue generating poses based on the last pose in the spheric_poses list
    poses = gen_swing_path(spheric_poses[-1], num_frames=n_poses)
    spheric_poses += poses

    for th in np.linspace(-1 * y_angle, y_angle, n_poses)[:-1]:
        spheric_poses += [spheric_pose_y(th, y_radius)] 
    
    poses = gen_swing_path(spheric_poses[-1], num_frames=n_poses)
    spheric_poses += poses

    for th in np.linspace(y_angle, 0, n_poses//2)[:-1]:
        spheric_poses += [spheric_pose_y(th, y_radius)] 

    # Generate poses by rotating up and down around the x-axis
    for th in np.linspace(0, -1 * x_angle, n_poses//2):
        spheric_poses += [spheric_pose_x(th, x_radius)]

    poses = gen_swing_path(spheric_poses[-1], num_frames=n_poses)
    spheric_poses += poses

    for th in np.linspace(-1 * x_angle, x_angle, n_poses)[:-1]:
        spheric_poses += [spheric_pose_x(th, x_radius)] 
    
    poses = gen_swing_path(spheric_poses[-1], num_frames=n_poses)
    spheric_poses += poses

    for th in np.linspace(x_angle, 0, n_poses//2)[:-1]:
        spheric_poses += [spheric_pose_x(th, x_radius)] 

    return spheric_poses

def convert(c2w, phi=0):
    """
    Applies a rotation along the y-axis to a given camera-to-world transformation matrix.

    Parameters:
    c2w (np.array): A 3x4 camera-to-world transformation matrix.
    phi (float): The rotation angle in radians around the y-axis.

    Returns:
    np.array: The updated 4x4 camera-to-world transformation matrix after applying the rotation.
    """

    # Extend the 3x4 matrix to a 4x4 matrix by appending a row [0, 0, 0, 1]
    # This row effectively transforms the 3x4 matrix into a homogeneous transformation matrix.
    c2w = np.concatenate((c2w, np.array([[0, 0, 0, 1]])), axis=0)

    # Define the rotation matrix for rotating around the y-axis.
    # The cosine and sine functions compute the components of the matrix based on the angle phi.
    # The matrix layout is:
    # [ cos(phi)  0  sin(phi)  0 ]
    # [     0     1     0      0 ]
    # [-sin(phi)  0  cos(phi)  0 ]
    # [     0     0     0      1 ]
    # This matrix allows for rotation around the y-axis while keeping the z and x positions adjusted
    # according to the sine and cosine of the angle phi.
    rot = np.array([
        [np.cos(phi), 0, np.sin(phi), 0],
        [0, 1, 0, 0],
        [-np.sin(phi), 0, np.cos(phi), 0],
        [0, 0, 0, 1],
    ])
    
    # Multiply the rotation matrix by the original camera-to-world matrix to apply the rotation.
    # The multiplication is done in such an order that the rotation effects are applied directly
    # to the camera-to-world matrix.
    return rot @ c2w

def normalize(x):
    """
    Normalizes a vector to have unit length.
    
    Parameters:
    x (np.array): A numpy array representing a vector in 3D space.
    
    Returns:
    np.array: The normalized vector with unit length.
    """
    return x / np.linalg.norm(x)


def viewmatrix(z, up, pos):
    """
    Creates a camera view matrix from camera forward direction, up vector, and position.
    
    Parameters:
    z (np.array): The forward direction vector of the camera.
    up (np.array): The up direction vector of the camera.
    pos (np.array): The position of the camera in 3D space.
    
    Returns:
    np.array: A 4x4 view matrix representing the camera's orientation and position.
    """
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m


def ptstocam(pts, c2w):
    """
    Transforms points from world coordinates to camera coordinates.
    
    Parameters:
    pts (np.array): Points in world coordinates.
    c2w (np.array): Camera-to-world transformation matrix.
    
    Returns:
    np.array: Points in camera coordinates.
    """
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt


def poses_avg(poses):
    """
    Computes an average camera pose from a list of poses.
    
    Parameters:
    poses (np.array): An array of camera poses.
    
    Returns:
    np.array: The average camera pose.
    """
    hwf = poses[0, :3, -1:]
    center = poses[:, :3, 3].mean(0)
    vec2 = normalize(poses[:, :3, 2].sum(0))
    up = poses[:, :3, 1].sum(0)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w

def pad_0001(array):
    """
    Pad [0, 0, 0, 1] to the bottom row.

    Args:
        array: (3, 4) or (N, 3, 4).

    Returns:
        Array of shape (4, 4) or (N, 4, 4).
    """
    if array.ndim == 2:
        if not array.shape == (3, 4):
            raise ValueError(f"Expected array of shape (3, 4), but got {array.shape}.")
    elif array.ndim == 3:
        if not array.shape[-2:] == (3, 4):
            raise ValueError(
                f"Expected array of shape (N, 3, 4), but got {array.shape}."
            )
    else:
        raise ValueError(
            f"Expected array of shape (3, 4) or (N, 3, 4), but got {array.shape}."
        )

    if array.ndim == 2:
        bottom = np.array([0, 0, 0, 1], dtype=array.dtype)
        return np.concatenate([array, bottom[None, :]], axis=0)
    elif array.ndim == 3:
        bottom_single = np.array([0, 0, 0, 1], dtype=array.dtype)
        bottom = np.broadcast_to(bottom_single, (array.shape[0], 1, 4))
        return np.concatenate([array, bottom], axis=-2)
    else:
        raise ValueError("Should not reach here.")


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N, z_min):
    """
    Generates a set of camera poses in a spiral path for rendering purposes.
    
    Parameters:
    c2w (np.array): Initial camera-to-world transformation matrix.
    up (np.array): Global up vector for the scene.
    rads (tuple): Radial distances for x, y, and z coordinates.
    focal (float): Focal length to adjust the spiral's tightness.
    zdelta (float): Vertical offset (unused in this specific function but often used in spiral calculations).
    zrate (float): Rate of change of the z-coordinate in the spiral.
    rots (float): Number of full rotations around the center.
    N (int): Number of camera poses to generate.
    
    Returns:
    list of np.array: A list of camera poses in the form of 4x4 matrices.
    """
    render_poses = []
    rads = np.array(list(rads) + [1.])  # Appends 1 to maintain homogeneous coordinates
    # hwf = c2w[:,4:5]  # Extracts the last column for perspective projection settings
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:  # Iterates over the angle to create the spiral
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))  # Compute forward vector

        render_poses.append(pad_0001(viewmatrix(z, up, c)))

    # we need make the camera roughly lie +z
    # z_min = 10
    print("z_min ", z_min)
    for rp in render_poses:
        if rp[2,3] < z_min:
            z_min = rp[2,3]
    
    print("z_min ", z_min)
    
    new_render_poses = []
    for rp in render_poses:
        rp[2, 3] -= 0.95 * z_min
        new_render_poses.append(rp)

    return new_render_poses


def render_path_spiral_default(focal=5, N=180, z_min=-25):
    # if not satisfy, can adjust to minor, do not larger miner than 0.5
    # # test render_path_spiral
    c2w = np.eye(4) # identity matrix for initial pose
    # c2w[1, 3] -= 3000.0 
    up = np.array([0, 1, 0])  # global up vector
    # rads = (0.2, 0.2, 0.2)  # radial distances for x, y, z, maybe need recheck for the meaning of these values
    rads = (0.4, 0.4, 0.4)  # radial distances for x, y, z, maybe need recheck for the meaning of these values
    # rads = (1.5, 1.5, 1.5)  # radial distances for x, y, z, maybe need recheck for the meaning of these values
    # focal = 5  # the bigger the focal, the tighter the spiral, look at the scene from a distance
    zdelta = 0.1  # vertical offset
    zrate = 0.5  # rate of change of z-coordinate, may be should align with the spheric_poses?
    rots = 2  # number of full rotations
    # N = 180 # number of camera poses to generate

    # now use the ct.convert.pose_opengl_to_opencv to convert the poses to opencv style
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N, z_min)
    return render_poses

def render_path_spiral_default_torch(focal=5):
    # if not satisfy, can adjust to minor, do not larger miner than 0.5
    # # test render_path_spiral
    c2w = np.eye(4) # identity matrix for initial pose
    up = np.array([0, 1, 0])  # global up vector
    rads = (0.2, 0.2, 0.2)  # radial distances for x, y, z, maybe need recheck for the meaning of these values
    # focal = 5  # the bigger the focal, the tighter the spiral, look at the scene from a distance
    zdelta = 0.1  # vertical offset
    zrate = 0.5  # rate of change of z-coordinate, may be should align with the spheric_poses?
    rots = 2  # number of full rotations
    N = 180 # number of camera poses to generate

    # now use the ct.convert.pose_opengl_to_opencv to convert the poses to opencv style
    render_poses = render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N)
    return [torch.from_numpy(i).float() for i in render_poses]


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getWorld2View2_tensor(R, t, translate=torch.tensor([.0, .0, .0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0,1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt.float()

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def transform_pcd(pcd, MLMatrix44):
    # transform pointcloud to zero123 coordinate
    # in this case, we just tranfer the pointcloud to the origin
    # designed for BasicPointCloud
    points_xyz = pcd.points
    ones = np.ones((points_xyz.shape[0], 1))
    homo_xyz = np.hstack((points_xyz, ones))
    new_pc_xyz = (MLMatrix44 @ homo_xyz.T).T
    # create a new Basicpcd
    new_pcd = BasicPointCloud(points=new_pc_xyz[:,:3], colors=pcd.colors, normals=pcd.normals)
    return new_pcd

import scipy.stats as stats
def z_score_from_percentage(percentage):
    """
    Returns the z-score for a given percentage in a standard normal distribution.

    :param percentage: The desired percentage of points remaining (e.g., 5 for 5%)
    :return: The corresponding z-score
    """
    # Convert percentage to a proportion
    proportion = percentage / 100

    # Calculate the z-score
    z_score = stats.norm.ppf(1 - proportion)

    return z_score

# def gen_swing_path(init_pose=torch.eye(4), num_frames=10, r_x=0.2, r_y=0, r_z=-0.2):
#     """
#     Generates a sequence of transformation matrices simulating a swing-like motion in 3D space.

#     Parameters:
#     init_pose (torch.Tensor): Initial 4x4 transformation matrix representing the starting pose of the object.
#                               Default is the identity matrix, indicating no initial translation or rotation.
#     num_frames (int): The number of frames for which the path is generated, each frame having its own transformation matrix.
#     r_x (float): Amplitude of the swing along the x-axis. Defines the extent of left-right motion.
#     r_y (float): Amplitude of the swing along the y-axis. Defines the extent of up-down motion.
#     r_z (float): Amplitude of the swing along the z-axis, with an offset that affects forward-backward motion and vertical position.

#     Returns:
#     List of torch.Tensor: A list of 4x4 transformation matrices for each frame, simulating the swing motion.
#     """
#     # Create a time vector from 0 to 1, evenly spaced over the number of frames
#     t = torch.arange(num_frames) / (num_frames - 1)
    
#     # Initialize the poses array by repeating the initial pose for each frame
#     poses = init_pose.repeat(num_frames, 1, 1)

#     # Initialize the swing transformation matrix, repeated across all frames
#     swing = torch.eye(4).repeat(num_frames, 1, 1)
    
#     # Set the translations for each axis based on the swing parameters and time vector
#     swing[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)  # Sine wave for x-axis
#     swing[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)  # Cosine wave for y-axis
#     swing[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1)  # Cosine wave with offset for z-axis

#     # Combine the initial pose with the swing transformations for each frame
#     for i in range(num_frames):
#         poses[i, :, :] = poses[i, :, :] @ swing[i, :, :]  # Matrix multiplication to apply transformations

#     # Return the list of transformation matrices
#     return list(poses.unbind())

# # def gen_swing_path(num_frames=180, r_x=0.5, r_y=0.5, r_z=0.10):
# #     "Return a list of matrix [4, 4]"
# #     t = torch.arange(num_frames) / (num_frames - 1)
# #     poses = torch.eye(4).repeat(num_frames, 1, 1)
# #     poses[:, 0, 3] = r_x * torch.sin(2. * math.pi * t)
# #     poses[:, 1, 3] = r_y * torch.cos(2. * math.pi * t)
# #     poses[:, 2, 3] = r_z * (torch.cos(2. * math.pi * t) - 1.)
# #     return list(poses.unbind())


# def create_spheric_poses_along_y(n_poses=10):
#     """
#     Generates a list of camera poses arranged spherically, focusing on rotation along the y and x axes.

#     Parameters:
#     n_poses (int): Total number of poses to generate.

#     Returns:
#     list of torch.Tensor: A list of camera poses, each represented as a 4x4 transformation matrix.
#     """

#     # Nested function to calculate a spherical pose rotating along the y-axis
#     def spheric_pose_y(phi, radius=10):
#         """
#         Generates a transformation matrix for a pose on a sphere at a specific angle phi along the y-axis.

#         Parameters:
#         phi (float): Angle in radians to rotate around the y-axis.
#         radius (float): Radius of the sphere from the center to the camera.

#         Returns:
#         torch.Tensor: A 4x4 transformation matrix representing the camera pose.
#         """
#         # Translation matrix that moves the camera along the y-axis
#         trans_t = lambda t: np.array([
#             [1, 0, 0, math.sin(2. * math.pi * t) * radius],
#             [0, 1, 0, 0],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]
#         ])

#         # Rotation matrix about the y-axis
#         rot_phi = lambda phi: np.array([
#             [np.cos(phi), 0, np.sin(phi), 0],
#             [0, 1, 0, 0],
#             [-np.sin(phi), 0, np.cos(phi), 0],
#             [0, 0, 0, 1]
#         ])

#         # Combine translation and rotation into a single camera-to-world matrix
#         c2w = rot_phi(phi) @ trans_t(phi)
#         c2w = np.array([[1, 0, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1]]) @ c2w
#         c2w = torch.tensor(c2w).float()
#         return c2w

#     # Similar nested function for rotation along the x-axis
#     def spheric_pose_x(phi, radius=10):
#         """
#         Generates a transformation matrix for a pose on a sphere at a specific angle phi along the x-axis.

#         Parameters:
#         phi (float): Angle in radians to rotate around the x-axis.
#         radius (float): Radius of the sphere from the center to the camera.

#         Returns:
#         torch.Tensor: A 4x4 transformation matrix representing the camera pose.
#         """
#         # Translation matrix that moves the camera along the x-axis
#         trans_t = lambda t: np.array([
#             [1, 0, 0, 0],
#             [0, 1, 0, math.sin(2. * math.pi * t) * radius * -1],
#             [0, 0, 1, 0],
#             [0, 0, 0, 1]
#         ])

#         # Rotation matrix about the x-axis
#         rot_theta = lambda th: np.array([
#             [1, 0, 0, 0],
#             [0, np.cos(th), -np.sin(th), 0],
#             [0, np.sin(th), np.cos(th), 0],
#             [0, 0, 0, 1]
#         ])

#         # Combine translation and rotation into a single camera-to-world matrix
#         c2w = rot_theta(phi) @ trans_t(phi)
#         c2w = np.array([[1, 0, 0, 0],
#                         [0, 1, 0, 0],
#                         [0, 0, 1, 0],
#                         [0, 0, 0, 1]]) @ c2w
#         c2w = torch.tensor(c2w).float()
#         return c2w

#     spheric_poses = []

#     # Initial swing path generation
#     poses = gen_swing_path()
#     spheric_poses += poses

#     # Set parameters for spherical rotation
#     factor = 1
#     y_angle = (1/16) * np.pi * factor
#     x_angle = (1/16) * np.pi * factor
#     x_radius = 0.1 * factor
#     y_radius = 0.1 * factor

#     # Generate poses by rotating left and right around the y-axis
#     for th in np.linspace(0, -1 * y_angle, n_poses//2):
#         spheric_poses += [spheric_pose_y(th, y_radius)]

#     # Continue generating poses based on the last pose in the spheric_poses list
#     poses = gen_swing_path(spheric_poses[-1])
#     spheric_poses += poses

#     for th in np.linspace(-1 * y_angle, y_angle, n_poses)[:-1]:
#         spheric_poses += [spheric_pose_y(th, y_radius)] 
    
#     poses = gen_swing_path(spheric_poses[-1])
#     spheric_poses += poses

#     for th in np.linspace(y_angle, 0, n_poses//2)[:-1]:
#         spheric_poses += [spheric_pose_y(th, y_radius)] 

#     # Generate poses by rotating up and down around the x-axis
#     for th in np.linspace(0, -1 * x_angle, n_poses//2):
#         spheric_poses += [spheric_pose_x(th, x_radius)]

#     poses = gen_swing_path(spheric_poses[-1])
#     spheric_poses += poses

#     for th in np.linspace(-1 * x_angle, x_angle, n_poses)[:-1]:
#         spheric_poses += [spheric_pose_x(th, x_radius)] 
    
#     poses = gen_swing_path(spheric_poses[-1])
#     spheric_poses += poses

#     for th in np.linspace(x_angle, 0, n_poses//2)[:-1]:
#         spheric_poses += [spheric_pose_x(th, x_radius)] 

#     return spheric_poses

# def convert(c2w, phi=0):
#     """
#     Applies a rotation along the y-axis to a given camera-to-world transformation matrix.

#     Parameters:
#     c2w (np.array): A 3x4 camera-to-world transformation matrix.
#     phi (float): The rotation angle in radians around the y-axis.

#     Returns:
#     np.array: The updated 4x4 camera-to-world transformation matrix after applying the rotation.
#     """

#     # Extend the 3x4 matrix to a 4x4 matrix by appending a row [0, 0, 0, 1]
#     # This row effectively transforms the 3x4 matrix into a homogeneous transformation matrix.
#     c2w = np.concatenate((c2w, np.array([[0, 0, 0, 1]])), axis=0)

#     # Define the rotation matrix for rotating around the y-axis.
#     # The cosine and sine functions compute the components of the matrix based on the angle phi.
#     # The matrix layout is:
#     # [ cos(phi)  0  sin(phi)  0 ]
#     # [     0     1     0      0 ]
#     # [-sin(phi)  0  cos(phi)  0 ]
#     # [     0     0     0      1 ]
#     # This matrix allows for rotation around the y-axis while keeping the z and x positions adjusted
#     # according to the sine and cosine of the angle phi.
#     rot = np.array([
#         [np.cos(phi), 0, np.sin(phi), 0],
#         [0, 1, 0, 0],
#         [-np.sin(phi), 0, np.cos(phi), 0],
#         [0, 0, 0, 1],
#     ])
    
#     # Multiply the rotation matrix by the original camera-to-world matrix to apply the rotation.
#     # The multiplication is done in such an order that the rotation effects are applied directly
#     # to the camera-to-world matrix.
#     return rot @ c2w

def assert_pose(pose):
    if pose.shape != (4, 4):
        raise ValueError(
            f"pose must has shape (4, 4), but got {pose} of shape {pose.shape}."
        )
    is_valid = np.allclose(pose[3, :], np.array([0, 0, 0, 1]))
    if not is_valid:
        raise ValueError(f"pose must has [0, 0, 0, 1] the bottom row, but got {pose}.")

def opengl_c2w_to_opencv_c2w(transformation_gl):
    # Step 3: Adjust for coordinate system differences between OpenCV and OpenGL
    # Flip Y and Z axes
    convert_cv_to_gl = np.array([
        [1,  0,  0, 0],
        [0, -1,  0, 0],  # Flip Y
        [0,  0, -1, 0],  # Flip Z
        [0,  0,  0, 1]
    ])

    convert_gl_to_cv = convert_cv_to_gl.T

    if isinstance(transformation_gl, list):
        transformation_cv = []
        for transformation_gl_single in transformation_gl:
            transformation_cv_single = transformation_gl_single @ convert_gl_to_cv
            transformation_cv.append(transformation_cv_single)
    else:
        transformation_cv = transformation_gl @ convert_gl_to_cv

    return transformation_cv


def pose_opengl_to_opencv(glpose):
    """
    Convert pose from OpenGL convention to OpenCV convention.

    - OpenCV
        - +X: Right
        - +Y: Down
        - +Z: The view direction, pointing forward and away from the camera
        - Used in: OpenCV, COLMAP, camtools default
    - OpenGL
        - +X: Right
        - +Y: Up
        - +Z: The negative view direction, pointing back and away from the camera
        - -Z: The view direction
        - Used in: OpenGL, Blender, Nerfstudio
          https://docs.nerf.studio/quickstart/data_conventions.html#coordinate-conventions
    """
    if isinstance(glpose, list):
        converted_poses = []
        for pose in glpose:
            assert_pose(pose)
            pose = np.copy(pose)
            pose[2, :] *= -1
            pose = pose[[1, 0, 2, 3], :]
            pose[0:3, 1:3] *= -1
            converted_poses.append(pose)
    else:
        assert_pose(glpose)
        pose = np.copy(glpose)
        pose[2, :] *= -1
        pose = pose[[1, 0, 2, 3], :]
        pose[0:3, 1:3] *= -1
        converted_poses = pose
    return converted_poses

# for more smooth path
def pose_aug(aug_pose_factor, all_poses):
    aug_pose_factor = 0 # set pose augmentation for better results
    cnt = len(all_poses)
    if aug_pose_factor > 0:
        for i in range(cnt):
            cur_pose = torch.FloatTensor(all_poses[i])
            for _ in range(aug_pose_factor):
                cam_ext, cam_ext_inv = self.get_rand_ext()  # [b,4,4]
                cur_aug_pose = torch.matmul(cam_ext, cur_pose)
                all_poses += [cur_aug_pose]
    return all_poses

def apply_rotation(q1, q2):
    """
    Applies a rotation to a quaternion.
    
    Parameters:
    q1 (Tensor): The original quaternion.
    q2 (Tensor): The rotation quaternion to be applied.
    
    Returns:
    Tensor: The resulting quaternion after applying the rotation.
    """
    # Extract components for readability
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    # Compute the product of the two quaternions
    w3 = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x3 = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y3 = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z3 = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    
    # Combine the components into a new quaternion tensor
    q3 = torch.tensor([w3, x3, y3, z3])

    # Normalize the resulting quaternion
    q3_normalized = q3 / torch.norm(q3)
    
    return q3_normalized


def batch_quaternion_multiply(q1, q2):
    """
    Multiply batches of quaternions.
    
    Args:
    - q1 (torch.Tensor): A tensor of shape [N, 4] representing the first batch of quaternions.
    - q2 (torch.Tensor): A tensor of shape [N, 4] representing the second batch of quaternions.
    
    Returns:
    - torch.Tensor: The resulting batch of quaternions after applying the rotation.
    """
    # Calculate the product of each quaternion in the batch
    w = q1[:, 0] * q2[:, 0] - q1[:, 1] * q2[:, 1] - q1[:, 2] * q2[:, 2] - q1[:, 3] * q2[:, 3]
    x = q1[:, 0] * q2[:, 1] + q1[:, 1] * q2[:, 0] + q1[:, 2] * q2[:, 3] - q1[:, 3] * q2[:, 2]
    y = q1[:, 0] * q2[:, 2] - q1[:, 1] * q2[:, 3] + q1[:, 2] * q2[:, 0] + q1[:, 3] * q2[:, 1]
    z = q1[:, 0] * q2[:, 3] + q1[:, 1] * q2[:, 2] - q1[:, 2] * q2[:, 1] + q1[:, 3] * q2[:, 0]

    # Combine into new quaternions
    q3 = torch.stack((w, x, y, z), dim=1)
    
    # Normalize the quaternions
    norm_q3 = q3 / torch.norm(q3, dim=1, keepdim=True)
    
    return norm_q3

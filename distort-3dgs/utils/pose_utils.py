import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from scene.utils import Camera
from copy import deepcopy
def rotation_matrix_to_quaternion(rotation_matrix):
    """将旋转矩阵转换为四元数"""
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_rotation_matrix(quat):
    """将四元数转换为旋转矩阵"""
    return R.from_quat(quat).as_matrix()

def quaternion_slerp(q1, q2, t):
    """在两个四元数之间进行球面线性插值（SLERP）"""
    # 计算两个四元数之间的点积
    dot = np.dot(q1, q2)

    # 如果点积为负，取反一个四元数以保证最短路径插值
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # 防止数值误差导致的问题
    dot = np.clip(dot, -1.0, 1.0)

    # 计算插值参数
    theta = np.arccos(dot) * t
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)

    # 计算插值结果
    return np.cos(theta) * q1 + np.sin(theta) * q3

def bezier_interpolation(p1, p2, t):
    """在两点之间使用贝塞尔曲线进行插值"""
    return (1 - t) * p1 + t * p2
def linear_interpolation(v1, v2, t):
    """线性插值"""
    return (1 - t) * v1 + t * v2
def smooth_camera_poses(cameras, num_interpolations=5):
    """对一系列相机位姿进行平滑处理，通过在每对位姿之间插入额外的位姿"""
    smoothed_cameras = []
    smoothed_times = []
    total_poses = len(cameras) - 1 + (len(cameras) - 1) * num_interpolations
    time_increment = 10 / total_poses

    for i in range(len(cameras) - 1):
        cam1 = cameras[i]
        cam2 = cameras[i + 1]

        # 将旋转矩阵转换为四元数
        quat1 = rotation_matrix_to_quaternion(cam1.orientation)
        quat2 = rotation_matrix_to_quaternion(cam2.orientation)

        for j in range(num_interpolations + 1):
            t = j / (num_interpolations + 1)

            # 插值方向
            interp_orientation_quat = quaternion_slerp(quat1, quat2, t)
            interp_orientation_matrix = quaternion_to_rotation_matrix(interp_orientation_quat)

            # 插值位置
            interp_position = linear_interpolation(cam1.position, cam2.position, t)

            # 计算插值时间戳
            interp_time = i*10 / (len(cameras) - 1) + time_increment * j

            # 添加新的相机位姿和时间戳
            newcam = deepcopy(cam1)
            newcam.orientation = interp_orientation_matrix
            newcam.position = interp_position
            smoothed_cameras.append(newcam)
            smoothed_times.append(interp_time)

    # 添加最后一个原始位姿和时间戳
    smoothed_cameras.append(cameras[-1])
    smoothed_times.append(1.0)
    print(smoothed_times)
    return smoothed_cameras, smoothed_times



def skew_sym_mat(x):
    device = x.device
    dtype = x.dtype
    ssm = torch.zeros(3, 3, device=device, dtype=dtype)
    ssm[0, 1] = -x[2]
    ssm[0, 2] = x[1]
    ssm[1, 0] = x[2]
    ssm[1, 2] = -x[0]
    ssm[2, 0] = -x[1]
    ssm[2, 1] = x[0]
    return ssm

def SO3_exp(theta):
    device = theta.device
    dtype = theta.dtype

    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    I = torch.eye(3, device=device, dtype=dtype)
    if angle < 1e-5:
        return I + W + 0.5 * W2
    else:
        return (
            I
            + (torch.sin(angle) / angle) * W
            + ((1 - torch.cos(angle)) / (angle**2)) * W2
        )

def V(theta):
    dtype = theta.dtype
    device = theta.device
    I = torch.eye(3, device=device, dtype=dtype)
    W = skew_sym_mat(theta)
    W2 = W @ W
    angle = torch.norm(theta)
    if angle < 1e-5:
        V = I + 0.5 * W + (1.0 / 6.0) * W2
    else:
        V = (
            I
            + W * ((1.0 - torch.cos(angle)) / (angle**2))
            + W2 * ((angle - torch.sin(angle)) / (angle**3))
        )
    return V

def SE3_exp(tau):
    dtype = tau.dtype
    device = tau.device

    rho = tau[:3]
    theta = tau[3:]
    R = SO3_exp(theta)
    t = V(theta) @ rho

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def update_pose(camera: Camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], dim=0)
    # print(tau)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = torch.tensor(camera.R.T)
    T_w2c[0:3, 3] = torch.tensor(camera.T)

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3].T
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R.detach().cpu().numpy(), new_T.detach().cpu().numpy())

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

def get_loss_tracking(image, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda() # CHW

    # l1 = opacity * torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask) if opacity is not None else 
    if opacity is not None:
        ssim_loss = 1.0 - ssim(image, gt_image)
        _, h, w = gt_image.shape
        mask_shape = (1, h, w)
        rgb_boundary_threshold = 0.01
        rgb_pixel_mask = (gt_image.sum(dim=0) > rgb_boundary_threshold).view(*mask_shape)
        # rgb_pixel_mask = rgb_pixel_mask * viewpoint.grad_mask
        l1 = opacity * F.smooth_l1_loss(image * rgb_pixel_mask, gt_image * rgb_pixel_mask, reduction='none')
        # torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    else:
        l1 = F.smooth_l1_loss(image, gt_image)
        ssim_loss = 1.0 - ssim(image, gt_image)
    return 0.8 * l1.mean() + 0.2 * ssim_loss


def tracking_loss(image, opacity, viewpoint):
    gt_image = viewpoint.original_image # CHW
    l1 = opacity * F.smooth_l1_loss(image, gt_image, reduction='none')
    # we also use ssim loss for structure
    ssim_loss = 1.0 - ssim(image, gt_image)
    opacity_loss = 1 - opacity
    return 0.8 * l1.mean() + 0.2 * ssim_loss  # + 1. * opacity_loss.mean()


# # 示例：使用两个相机位姿
# cam1 = Camera(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0]))
# cam2 = Camera(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), np.array([1, 1, 1]))

# # 应用平滑处理
# smoothed_cameras = smooth_camera_poses([cam1, cam2], num_interpolations=5)

# # 打印结果
# for cam in smoothed_cameras:
#     print("Orientation:\n", cam.orientation)
#     print("Position:", cam.position)

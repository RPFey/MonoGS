import numpy as np
import torch


def rt2mat(R, T):
    mat = np.eye(4)
    mat[0:3, 0:3] = R
    mat[0:3, 3] = T
    return mat

def batch_rotation_matrix_to_quaternion(rotation_matrices):
    """
    Convert a batch of rotation matrices to quaternions.

    Args:
    - rotation_matrices (torch.Tensor): A tensor of shape (batch_size, 3, 3)
                                        containing batch of rotation matrices.

    Returns:
    - quaternions (torch.Tensor): A tensor of shape (batch_size, 4)
                                  containing quaternions for each rotation matrix.
    """
    # Ensure the input is a batch of 3x3 matrices
    assert rotation_matrices.dim() == 3
    assert rotation_matrices.size(1) == 3
    assert rotation_matrices.size(2) == 3

    # Extract rotation components
    r11, r12, r13 = rotation_matrices[:, 0, 0], rotation_matrices[:, 0, 1], rotation_matrices[:, 0, 2]
    r21, r22, r23 = rotation_matrices[:, 1, 0], rotation_matrices[:, 1, 1], rotation_matrices[:, 1, 2]
    r31, r32, r33 = rotation_matrices[:, 2, 0], rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2]

    # Compute quaternion components
    qw = 0.5 * torch.sqrt(1.0 + r11 + r22 + r33)
    qx = (r32 - r23) / (4.0 * qw)
    qy = (r13 - r31) / (4.0 * qw)
    qz = (r21 - r12) / (4.0 * qw)

    quaternions = torch.stack([qx, qy, qz, qw], dim=1)
    return quaternions

def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.

    Args:
    - quaternion (torch.Tensor): A quaternion in the format (w, x, y, z).

    Returns:
    - rotation_matrix (torch.Tensor): A 3x3 rotation matrix.
    """
    # Ensure the input is a quaternion
    assert quaternion.dim() == 1
    assert quaternion.size(0) == 4

    # Extract quaternion components
    qx, qy, qz, qw = quaternion[0], quaternion[1], quaternion[2], quaternion[3]

    # Compute rotation matrix
    rotation_matrix = torch.zeros(3, 3, dtype=quaternion.dtype, device=quaternion.device)
    rotation_matrix[0, 0] = 1 - 2 * qy**2 - 2 * qz**2
    rotation_matrix[0, 1] = 2 * qx * qy - 2 * qz * qw
    rotation_matrix[0, 2] = 2 * qx * qz + 2 * qy * qw
    rotation_matrix[1, 0] = 2 * qx * qy + 2 * qz * qw
    rotation_matrix[1, 1] = 1 - 2 * qx**2 - 2 * qz**2
    rotation_matrix[1, 2] = 2 * qy * qz - 2 * qx * qw
    rotation_matrix[2, 0] = 2 * qx * qz - 2 * qy * qw
    rotation_matrix[2, 1] = 2 * qy * qz + 2 * qx * qw
    rotation_matrix[2, 2] = 1 - 2 * qx**2 - 2 * qy**2

    if torch.linalg.det(rotation_matrix) < 0:
        rotation_matrix *= -1

    return rotation_matrix

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


def update_pose(camera, converged_threshold=1e-4):
    tau = torch.cat([camera.cam_trans_delta, camera.cam_rot_delta], axis=0)

    T_w2c = torch.eye(4, device=tau.device)
    T_w2c[0:3, 0:3] = camera.R
    T_w2c[0:3, 3] = camera.T

    new_w2c = SE3_exp(tau) @ T_w2c

    new_R = new_w2c[0:3, 0:3]
    new_T = new_w2c[0:3, 3]

    converged = tau.norm() < converged_threshold
    camera.update_RT(new_R, new_T)

    camera.cam_rot_delta.data.fill_(0)
    camera.cam_trans_delta.data.fill_(0)
    return converged

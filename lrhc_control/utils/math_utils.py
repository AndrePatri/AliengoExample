import torch
import math

from EigenIPC.PyEigenIPC import VLevel
from EigenIPC.PyEigenIPC import Journal
from EigenIPC.PyEigenIPC import LogType

def get_pitch(quat: torch.Tensor) -> torch.Tensor:
    cos_theta = torch.clamp(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1.0, 1.0)
    theta = torch.acos(cos_theta)
    return theta.view(-1, 1)

def check_capsize(quat: torch.Tensor, 
        max_angle: float, 
        output_t: torch.Tensor=None):
    # Populate the output tensor with True where cos(theta) exceeds cos(max_angle)
    if output_t is not None:
        output_t[:, 0] = torch.clamp(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1.0, 1.0) < math.cos(max_angle)
        return None
    else:
        return torch.clamp(1-2*(quat[:, 1]**2+quat[:, 2]**2), -1.0, 1.0) < math.cos(max_angle)

def normalize_quaternion(q):
    # Normalizes the quaternion
    return q / torch.norm(q, dim=-1, keepdim=True)

def quaternion_difference(q1, q2):
    """ Compute the quaternion difference needed to rotate from q1 to q2 """
    def quat_conjugate(q):
        # Computes the conjugate of a quaternion
        w, x, y, z = q.unbind(-1)
        return torch.stack([w, -x, -y, -z], dim=-1)

    q1_conj = quat_conjugate(q1)

    return quaternion_multiply(q2, q1_conj)

def quaternion_multiply(q1, q2):
    """ Multiply two quaternions. """
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ], dim=-1)

def quaternion_to_angular_velocity(q_diff, dt):
    """ Convert a quaternion difference to an angular velocity vector. """
    angle = 2 * torch.arccos(q_diff[..., 0].clamp(-1.0, 1.0))  # Clamping for numerical stability
    axis = q_diff[..., 1:]
    norm = axis.norm(dim=-1, keepdim=True)
    norm = torch.where(norm > 0, norm, torch.ones_like(norm))
    axis = axis / norm
    angle = angle.unsqueeze(-1)  # Add an extra dimension for broadcastingù
    angular_velocity=(angle / dt) * axis
    return angular_velocity

def quat_to_omega(q0, q1, dt):
    """ Convert quaternion pairs to angular velocities """
    if q0.shape != q1.shape:
        raise ValueError("Tensor shapes do not match in quat_to_omega.")

    # Normalize quaternions and compute differences
    q0_normalized = normalize_quaternion(q0)
    q1_normalized = normalize_quaternion(q1)
    q_diff = quaternion_difference(q0_normalized, q1_normalized)

    return quaternion_to_angular_velocity(q_diff, dt)

def rel_vel(offset_q0_q1, 
        v0):
    
    # Calculate relative linear velocity in frame q1 from linear velocity in frame q0 using quaternions.

    # Ensure the quaternion is normalized
    offset_q0_q1 = F.normalize(offset_q0_q1, p=2, dim=0)

    # Convert the linear velocity vector to a quaternion
    v0_q = torch.cat([torch.tensor([0]), v0])

    # Rotate the linear velocity quaternion using the orientation offset quaternion
    rotated_velocity_quaternion = quaternion_multiply(offset_q0_q1, v0_q)
    offset_q0_q1_inverse = torch.cat([offset_q0_q1[0:1], -offset_q0_q1[1:]])

    # Multiply by the conjugate of the orientation offset quaternion to obtain the result in frame f1
    v1_q = quaternion_multiply(rotated_velocity_quaternion, offset_q0_q1_inverse)

    # Extract the linear velocity vector from the quaternion result
    v1 = v1_q[1:]

    return v1

if __name__ == "__main__":  
    n_envs = 2

    quats= torch.zeros((n_envs,4), dtype=torch.float32, device="cpu")
    is_capsized=torch.zeros((n_envs,1), dtype=torch.bool, device="cpu")
    # 45 degs pich
    quats[:, 0] = 0
    quats[:, 1] = 0.3826834
    quats[:, 2] = 0
    quats[:, 3] = 0.9238795
    theta_check=45*math.pi/180.0 # degs 

    print("###############")
    print(get_pitch(quats)*180/math.pi)
    check_capsize(quat=quats, max_angle=theta_check, output_t=is_capsized)
    print(is_capsized)
    
    
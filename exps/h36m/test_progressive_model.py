import argparse
import sys

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from hmpdata import Human36MDatasetConfig, H36MEval

from model.model import PGCAGC_V2
from model.velocity import GRUAutoencoder
from utils.skeletons import SkeletonH36M as Skeleton

from utils.functional import gen_velocity, read_config

sys.path.append(".")


def calculate_future_poses(next_pose, future_velocities):
    """
    Calculate future poses given the next pose and future velocities.

    Args:
    - next_pose: Tensor of shape [batch_size, 1, n_joints * 3]
    - future_velocities: Tensor of shape [batch_size, future_steps, n_joints * 3]

    Returns:
    - future_poses: Tensor of shape [batch_size, future_steps + 1, n_joints * 3]
    """
    batch_size, _, n_features = next_pose.shape
    future_steps = future_velocities.shape[1]

    # Initialize future_poses with the next_pose
    future_poses = torch.zeros(
        batch_size, future_steps + 1, n_features, device=next_pose.device
    )
    future_poses[:, 0, :] = next_pose.squeeze(1)

    # Calculate cumulative sum of velocities
    cumulative_velocities = torch.cumsum(future_velocities, dim=1)

    # Add cumulative velocities to the next_pose to get future poses
    future_poses[:, 1:, :] = next_pose + cumulative_velocities

    return future_poses


def predict_next_pose(x: torch.Tensor, model: nn.Module):
    x_permuted = x.permute(0, 2, 1)
    offset = x_permuted[:, :, -1:]
    with torch.no_grad():
        out = model(x, iter=999999)["out_step_3"]
        next_pose = out[:, :, :1] + offset
    return next_pose.permute(0, 2, 1)


def predict_poses(
    model: nn.Module,
    sequence: torch.Tensor,
    velocity_ae: nn.Module,
):
    out = predict_next_pose(x=sequence, model=model)
    z = model(sequence, iter=99999)["z"]
    dx = gen_velocity(sequence.clone())
    dy_hat = velocity_ae.decode(dx, z)
    poses = calculate_future_poses(out, dy_hat)
    # print(f"future poses: {poses.shape}")
    return poses


def evaluate(model: nn.Module, loader: DataLoader, velocity_ae: nn.Module):
    joint_to_ignore = np.array([16, 20, 23, 24, 28, 31]).astype(np.int64)
    joint_equal = np.array([13, 19, 22, 13, 27, 30]).astype(np.int64)
    joint_used_xyz = np.array(
        [
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
            25,
            26,
            27,
            29,
            30,
        ]
    ).astype(np.int64)
    num_samples = 0
    m_p3d_h36 = np.zeros([25])

    for x, y in loader:

        batch_size, seq_len, _, _ = x.shape
        num_samples += batch_size
        motion_input = x[:, :, joint_used_xyz]
        motion_input = motion_input.cuda()
        motion_input = motion_input.reshape(batch_size, seq_len, -1)
        # y_ = y.clone()[:, :, joint_used_xyz, :].reshape(batch_size, 25, 66)
        motion_pred = predict_poses(
            model=model,
            sequence=motion_input,
            velocity_ae=velocity_ae,
        )

        motion_target = y.detach()
        b, n, c, _ = motion_target.shape
        motion_gt = motion_target.clone()
        motion_pred = motion_pred.detach().cpu()
        pred_rot = motion_pred.clone().reshape(b, n, 22, 3)
        motion_pred = motion_target.clone().reshape(b, n, 32, 3)
        motion_pred[:, :, joint_used_xyz] = pred_rot
        tmp = motion_gt.clone()
        tmp[:, :, joint_used_xyz] = motion_pred[:, :, joint_used_xyz]
        motion_pred = tmp
        motion_pred[:, :, joint_to_ignore] = motion_pred[:, :, joint_equal]
        mjmpe_p3d_h36 = torch.sum(
            torch.mean(torch.norm(motion_pred * 1000 - motion_gt * 1000, dim=3), dim=2),
            dim=0,
        )
        m_p3d_h36 += mjmpe_p3d_h36.cpu().numpy()
    m_p3d_h36 = m_p3d_h36 / num_samples
    return m_p3d_h36


def test(model: nn.Module, loader: DataLoader, velocity_ae):
    result = evaluate(model=model, loader=loader, velocity_ae=velocity_ae)
    frame_time = 1000.0 / 25
    ms_keys = [f"{i * frame_time}" for i in range(1, 26)]
    result_list = result.tolist()
    mpjpe = {k: v for k, v in zip(ms_keys, result_list)}

    titles = [
        "40.0",
        "80.0",
        "160.0",
        "320.0",
        "400.0",
        "560.0",
        "720.0",
        "880.0",
        "1000.0",
    ]
    s = "MPJPE: "
    for title in titles:
        s += f"@{title}: {mpjpe[title]:.3f} |"
    print(s)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--main_model_chkpt', required=True)
    parser.add_argument('--ae_ckpt', required=True)
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_cfg', required=True)
    args = parser.parse_args()
    milestones = [3000, 5000]
    skeleton = Skeleton()
    cfg = read_config(args.model_cfg)
    model_cfg = cfg['model']
    model = PGCAGC_V2(
        skeleton=skeleton,
        **model_cfg
    )
    
    prog_model_ckpt_file = args.main_model_chkpt
    ckpt_prog_model = torch.load(prog_model_ckpt_file)
    model.load_state_dict(ckpt_prog_model["model"])

    nx = 66  # n_joints * 3
    ny = 66  # n_joints * 3
    horizon = 24

    specs = {
        "x_birnn": True,
        "e_birnn": True,
        "use_drnn_mlp": False,
        "nh_rnn": 128,
        "nh_mlp": [256, 512],
    }

    velocity_ae = GRUAutoencoder(nx, ny, horizon, specs, num_layers=1)
    ckpt = args.ae_ckpt
    state_dict = torch.load(ckpt)
    velocity_ae.load_state_dict(state_dict=state_dict)

    velocity_ae.eval()
    model.eval()

    dataset = H36MEval(
        data_dir=args.data_dir,
        config=Human36MDatasetConfig(h36m_target_length_eval=25),
        split_name="test",
    )
    val_loader = DataLoader(
        dataset=dataset,
        batch_size=128,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        drop_last=False,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    velocity_ae = velocity_ae.to(device)
    model = model.to(device)

    
    x, y = dataset[66]
    joint_used_xyz = np.array(
        [
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            10,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            21,
            22,
            25,
            26,
            27,
            29,
            30,
        ]
    ).astype(np.int64)
    test(model=model, loader=val_loader, velocity_ae=velocity_ae)

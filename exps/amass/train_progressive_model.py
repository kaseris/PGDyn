import argparse
import os
import sys

import randomname
import yaml
import torch
import torch.nn.functional as F

from datetime import datetime

from torch.utils.data import DataLoader

from hmpdata import AMASSDataset

from model.model import PGCAGC_V2
from model.velocity import GRUAutoencoder
from utils.skeletons import SkeletonAMASS as Skeleton

from utils.functional import gen_velocity, read_config

sys.path.append(".")


def calculate_loss(out_dict: dict, batch_input: torch.Tensor, target: torch.Tensor):
    batch_size, target_len, channels = target.shape
    batch_size, input_len, channels = batch_input.shape

    dx_src = gen_velocity(batch_input)
    dy_target = gen_velocity(target)

    target = target.permute(0, 2, 1)
    batch_input = batch_input.reshape(batch_size, input_len, channels // 3, 3)

    batch_input_small = (
        batch_input.clone()[:, :, skeleton.small, :]
        .reshape(batch_size, input_len, -1)
        .permute(0, 2, 1)
    )
    batch_input_medium = (
        batch_input.clone()[:, :, skeleton.medium, :]
        .reshape(batch_size, input_len, -1)
        .permute(0, 2, 1)
    )
    batch_input_full = (
        batch_input.clone()[:, :, skeleton.full, :]
        .reshape(batch_size, input_len, -1)
        .permute(0, 2, 1)
    )

    batch_target = target.clone().reshape(batch_size, target_len, channels // 3, 3)
    batch_target_small = batch_target[:, :1, skeleton.small, :].reshape(-1, 3)
    batch_target_medium = batch_target[:, :1, skeleton.medium, :].reshape(-1, 3)
    batch_target_full = batch_target[:, :1, skeleton.full, :].reshape(-1, 3)

    offset_small = batch_input_small[:, :, -1:]
    offset_medium = batch_input_medium[:, :, -1:]
    offset_full = batch_input_full[:, :, -1:]

    loss_dict = {}
    loss_dict_items = {
        "loss_small": 0.0,
        "loss_medium": 0.0,
        "loss_full": 0.0,
        "z_loss": 0.0,
    }

    # if "z" in out_dict and out_dict["z"] is not None:
    #     z = velocity_ae.encode(dx_src, dy_target)
    #     z_loss = torch.mean(torch.norm(out_dict["z"] - z, p=2, dim=1))
    #     loss_dict["z_loss"] = z_loss
    #     loss_dict_items["z_loss"] = z_loss.item()

    if "out_step_1" in out_dict and out_dict["out_step_1"] is not None:
        pred_small = out_dict["out_step_1"][:, :, :1] + offset_small
        pred_small = (
            pred_small.permute(0, 2, 1)
            .reshape(batch_size, 1, len(skeleton.small), 3)
            .reshape(-1, 3)
        )
        loss_small = torch.mean(torch.norm(pred_small - batch_target_small, 2, 1))
        loss_dict["loss_small"] = loss_small
        loss_dict_items["loss_small"] = loss_small.item()

    if "out_step_2" in out_dict and out_dict["out_step_2"] is not None:
        pred_medium = out_dict["out_step_2"][:, :, :1] + offset_medium
        pred_medium = (
            pred_medium.permute(0, 2, 1)
            .reshape(batch_size, 1, len(skeleton.medium), 3)
            .reshape(-1, 3)
        )
        loss_medium = torch.mean(torch.norm(pred_medium - batch_target_medium, 2, 1))
        loss_dict["loss_medium"] = loss_medium
        loss_dict_items["loss_medium"] = loss_medium.item()

    if "out_step_3" in out_dict and out_dict["out_step_3"] is not None:
        pred_full = out_dict["out_step_3"][:, :, :1] + offset_full
        pred_full = (
            pred_full.permute(0, 2, 1)
            .reshape(batch_size, 1, len(skeleton.full), 3)
            .reshape(-1, 3)
        )
        loss_full = torch.mean(torch.norm(pred_full - batch_target_full, 2, 1))
        loss_dict["loss_full"] = loss_full
        loss_dict_items["loss_full"] = loss_full.item()

    return loss_dict, loss_dict_items


def get_loss_string(loss_dict: dict):
    s = ""
    for key, value in loss_dict.items():
        s += " "
        s += key.capitalize().replace("_", " ")
        s += f": {value:4f} |"
    return s


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--model_cfg', required=True)
    parser.add_argument('--gru_ae_ckpt', required=True)
    parser.add_argument('--finetune', required=False, default=0, type=int)
    parser.add_argument('--model_ckpt', required=False, type=str)
    args = parser.parse_args()

    milestones = [1000, 2000]
    experiment_name = randomname.get_name()
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f'Creating new experiment: {experiment_name+"_"+now}')
    exp_path = os.path.join(f'runs/amass/{experiment_name+"_"+now}')
    os.mkdir(exp_path)
    cfg = read_config(args.model_cfg)
    with open(os.path.join(exp_path, "config.yaml"), 'w') as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)

    model_cfg = cfg['model']

    skeleton = Skeleton()
    model = PGCAGC_V2(
        skeleton=skeleton,
        **model_cfg
    )
    if args.finetune:
        pretrained_ckpt = torch.load(args.model_ckpt)
        model.load_state_dict(pretrained_ckpt["model"])

    n_params = 0
    for p in model.parameters():
        if p.requires_grad:
            n_params += p.numel()
    print(f'# parameters: {n_params:,}')

    # Define the velocity autoencoder and freeze it
    nx = 54  # n_joints * 3
    ny = 54  # n_joints * 3
    horizon = 24

    specs = {
        "x_birnn": True,
        "e_birnn": True,
        "use_drnn_mlp": False,
        "nh_rnn": 512,
        "nh_mlp": [512, 1024],
    }

    velocity_ae = GRUAutoencoder(nx, ny, horizon, specs, num_layers=3)
    ckpt = args.gru_ae_ckpt
    state_dict = torch.load(ckpt)
    velocity_ae.load_state_dict(state_dict=state_dict)

    velocity_ae.eval()

    for p in velocity_ae.parameters():
        if p.requires_grad:
            p.requires_grad = False

    data_cfg = cfg['data']
    dataset = AMASSDataset(
        data_dir=args.data_dir,
        actions=None,
        split=0,
        input_n=50,
        output_n=25,
        skip_rate=5
    )
    print(f'Length of train split: {len(dataset):,} samples')

    train_loader = DataLoader(
        dataset=dataset,
        batch_size=data_cfg['batch_size'],
        shuffle=True,
        pin_memory=True,
        drop_last=True,
        num_workers=0,
    )
    train_cfg = cfg['training']
    current_lr = train_cfg['current_lr']

    optimizer = torch.optim.Adam(
        params=model.parameters(), lr=current_lr, weight_decay=0
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    velocity_ae = velocity_ae.to(device)
    model = model.to(device)
    model.train()

    iters = 0
    max_iters = train_cfg['max_iters']

    avg_loss_small = 0.0
    avg_loss_medium = 0.0
    avg_loss_full = 0.0
    avg_dloss = 0.0
    avg_zloss = 0.0

    save_every = train_cfg['save_every']
    lr_change_steps = train_cfg['lr_change_steps']
    lrs = train_cfg['lrs']

    while iters < max_iters:
        for batch_input, batch_target, next_pose in train_loader:
            x = batch_input.clone()
            x = x.to(device)
            next_pose = next_pose.to(device)
            batch_target = batch_target.to(device)
            out_dict = model(x, iter=iters + 2000)

            losses, losses_items = calculate_loss(
                out_dict=out_dict, batch_input=x, target=next_pose
            )
            if iters < milestones[0]:
                loss1 = losses["loss_small"]
                loss = loss1
                avg_loss_small += losses_items["loss_small"]
            elif milestones[0] <= iters < milestones[1]:
                loss1, loss2 = losses["loss_small"], losses["loss_medium"]
                loss = loss1 + loss2
                avg_loss_small += losses_items["loss_small"]
                avg_loss_medium += losses_items["loss_medium"]
            elif iters >= milestones[1]:
                loss1, loss2, loss3 = (
                    losses["loss_small"],
                    losses["loss_medium"],
                    losses["loss_full"],
                )

                dx_src = gen_velocity(x.clone())
                dy_target = gen_velocity(batch_target.clone())
                z = velocity_ae.encode(dx_src, dy_target)
                z_loss = torch.mean(torch.norm(out_dict["z"] - z, p=2, dim=1))
                # z_loss = (1.0 - F.cosine_similarity(z, out_dict["z"], dim=1)).mean()
                dy_hat = out_dict["dy_hat"]
                batch_size, target_len, channels = dy_target.shape
                dy_target = dy_target.reshape(batch_size, target_len, channels // 3, 3).reshape(-1, 3)
                dy_hat = dy_hat.reshape(batch_size, target_len, channels // 3, 3).reshape(-1, 3)
                loss_vel = torch.mean(torch.norm(dy_hat * 1000. - dy_target * 1000., p=2, dim=1))
                alpha = 0.2
                loss = loss3 + z_loss + loss_vel

                avg_loss_small += losses_items["loss_small"]
                avg_loss_medium += losses_items["loss_medium"]
                avg_loss_full += losses_items["loss_full"]
                avg_dloss += loss_vel.item()
                avg_zloss += z_loss.item()
            iters += 1
            optimizer.zero_grad()

            loss.backward()
            optimizer.step()

            if iters % train_cfg['print_every'] == 0:
                avg_loss_small = avg_loss_small / train_cfg['print_every']
                avg_loss_medium = avg_loss_medium / train_cfg['print_every']
                avg_loss_full = avg_loss_full / train_cfg['print_every']
                avg_dloss = avg_dloss / train_cfg["print_every"]
                avg_zloss = avg_zloss / train_cfg['print_every']
                loss_str = ""
                loss_str += f"Loss small: {avg_loss_small:.4f} | "
                loss_str += f"Loss medium: {avg_loss_medium:.4f} | "
                loss_str += f"Loss full: {avg_loss_full:.4f} | "
                loss_str += f"Loss vel: {avg_dloss:.4f} | "
                loss_str += f"Z Loss: {avg_zloss:.4f} | "
                print(f"Iter: {iters} {loss_str} Learning rate: {current_lr}")
                avg_loss_small = 0.0
                avg_loss_medium = 0.0
                avg_loss_full = 0.0
                avg_dloss = 0.0
                avg_zloss = 0.0

            if iters + 1 >= max_iters:
                break

            if lr_change_steps[1] >= iters > lr_change_steps[0]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lrs[0]
                current_lr = lrs[0]
            elif iters > lr_change_steps[1]:
                for param_group in optimizer.param_groups:
                    param_group["lr"] = lrs[1]
                current_lr = lrs[1]

            if iters % save_every == 0:
                model_state_dict = model.state_dict()
                optimizer_state_dict = optimizer.state_dict()
                checkpoint = {
                    "model": model_state_dict,
                    "optimizer": optimizer_state_dict,
                }
                now = datetime.now().strftime("%Y%m%d_%H%M%S")
                checkpoint_name = f"checkpoint_iter_{iters}" + "_" + now + ".pt"
                torch.save(checkpoint, os.path.join(exp_path, checkpoint_name))

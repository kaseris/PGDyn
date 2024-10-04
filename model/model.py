import torch
import torch.nn as nn

from einops.layers.torch import Rearrange

from typing import List, Union


from model.modules import (
    MaskOutput,
    TemporalFC,
    ResolutionLevelV2,
)
from model.velocity import LatentVelocityProjection
from utils.skeletons import SkeletonH36M, SkeletonAMASS
from utils.functional import gen_velocity


class PGCAGC_V2(nn.Module):
    def __init__(
        self,
        skeleton: Union[SkeletonAMASS, SkeletonH36M],
        n_joints: List = [8, 13, 22],
        skeleton_resolution: List = ["small", "medium", "full"],
        normalize_adj_mask: bool = True,
        local_window: int = 5,
        seq_len: int = 50,
        target_len: int = 25,
        dims: int = 3,
        levels=[2, 2, 2],
        is_mask_trainable: bool = False,
        milestones: list = [20000, 40000],
        dyna_layers: list = [2, 4, 8],
        layer_norm_axis: str = "spatial",
    ) -> None:
        super().__init__()
        self.skeleton = skeleton
        self.arrange = Rearrange("b l c -> b c l")
        self.arrange_back = Rearrange("b c l -> b l c")
        self.milestones = milestones
        self.target_len = target_len
        self.mlp = nn.Parameter(torch.empty(seq_len, 4))
        nn.init.xavier_uniform_(self.mlp, gain=1e-8)

        self.small_res = nn.Sequential(
            *[
                ResolutionLevelV2(
                    mlp=self.mlp,
                    skeleton=skeleton,
                    n_joints=n_joints[0],
                    skeleton_resolution=skeleton_resolution[0],
                    normalize_adj_mask=normalize_adj_mask,
                    local_window=local_window,
                    seq_len=seq_len,
                    dims=dims,
                    layer_norm_axis=layer_norm_axis,
                    n_dynamic_layers=dyna_layers[0],
                )
                for _ in range(levels[0])
            ]
        )

        self.medium_res = nn.Sequential(
            *[
                ResolutionLevelV2(
                    mlp=self.mlp,
                    skeleton=skeleton,
                    n_joints=n_joints[1],
                    skeleton_resolution=skeleton_resolution[1],
                    normalize_adj_mask=normalize_adj_mask,
                    local_window=local_window,
                    seq_len=seq_len,
                    dims=dims,
                    layer_norm_axis=layer_norm_axis,
                    n_dynamic_layers=dyna_layers[1],
                )
                for _ in range(levels[1])
            ]
        )

        self.full_res = nn.Sequential(
            *[
                ResolutionLevelV2(
                    mlp=self.mlp,
                    skeleton=skeleton,
                    n_joints=n_joints[2],
                    skeleton_resolution=skeleton_resolution[2],
                    normalize_adj_mask=normalize_adj_mask,
                    local_window=local_window,
                    seq_len=seq_len,
                    dims=dims,
                    n_dynamic_layers=dyna_layers[2],
                )
                for _ in range(levels[2])
            ]
        )

        self.mask_s_m = MaskOutput(
            skeleton=skeleton,
            resolution="medium",
            trainable=is_mask_trainable,
            n_joints=n_joints[1],
            dims=dims,
            seq_len=seq_len,
        )

        self.mask_m_f = MaskOutput(
            skeleton=skeleton,
            resolution="full",
            trainable=is_mask_trainable,
            n_joints=n_joints[2],
            dims=dims,
            seq_len=seq_len,
        )
        self.temporal_fc_in = TemporalFC(seq_len=seq_len, hidden_dim=seq_len)
        self.temporal_fc_out = TemporalFC(seq_len=seq_len, hidden_dim=seq_len)

        self.spatial_fc_in = nn.Linear(
            in_features=n_joints[2] * dims, out_features=n_joints[2] * dims
        )
        self.spatial_fc_out = nn.Linear(
            in_features=n_joints[2] * dims, out_features=n_joints[2] * dims
        )

        self.mlp_v = nn.Parameter(torch.empty(self.target_len - 1, 4))
        nn.init.xavier_uniform_(self.mlp_v, gain=1e-8)

        self.vel_dl = ResolutionLevelV2(
            mlp=self.mlp_v,
            skeleton=skeleton,
            n_joints=22,
            skeleton_resolution="full",
            normalize_adj_mask=False,
            local_window=3,
            seq_len=self.target_len - 1,
            dims=3,
            layer_norm_axis="spatial",
            n_dynamic_layers=4,
        )
        self.lvp = LatentVelocityProjection()

    def forward(self, x: torch.Tensor, iter):
        """The input is supposed to be the raw batch as drawn from the data loader. That is,
        a tensor of shape [batch_size, seq_len, n_joints*dims]. Where:

        Shape
        ------

            - batch_size: The batch's sample size
            - n_joints: The original pose joint size (22 for H36M)
            - dims: The dimensionality that describes the joint (3)
            - seq_len: Input sequence length."""
        out_step_1, out_step_2, out_step_3, dy_hat, z = None, None, None, None, None
        # x = self.temporal_fc_in(x)
        x = self.spatial_fc_in(x)
        out_step_1 = self.arrange(
            self.temporal_fc_out(self.arrange_back(self._step1(x)))
        )

        if iter >= self.milestones[0]:
            out_step_2 = self._step2(x, out_step_1)

        if iter >= self.milestones[1]:
            out_step_3 = self._step3(x, out_step_2)
            out_step_3 = self.arrange(
                self.temporal_fc_out(self.arrange_back(out_step_3))
            )
            out_step_3 = out_step_3.permute(0, 2, 1)
            out_step_3 = self.spatial_fc_out(out_step_3)
            out_step_3 = out_step_3.permute(0, 2, 1)
            # Tensor out_step_3 represents the next pose for t=T+1 (after the observation)
            # We now need to compute the velocity of the input as well as the predicted velocities from the
            # velocity encoder. Afterwards, we will learn the latent variable z using a pretrained autoencoder.

            # Detach just in case
            dx = gen_velocity(x.clone().detach())
            out_clone = out_step_3.clone()
            out_clone = out_clone.detach()
            dy_hat = gen_velocity(out_clone[:, :, : self.target_len].permute(0, 2, 1))
            dy_hat = self.vel_dl(dy_hat.permute(0, 2, 1)).permute(0, 2, 1)
            # Estimate the latent velocity variable
            z = self.lvp(dx, dy_hat)

        return {
            "out_step_1": out_step_1,
            "out_step_2": out_step_2,
            "out_step_3": out_step_3,
            "z": z,
        }

    def _step1(self, x: torch.Tensor):
        input_ = self._prepare_input(
            input_=x.clone(), joints_to_use=self.skeleton.small
        )
        out = self.small_res(input_)
        return out

    def _step2(self, x: torch.Tensor, previous_output: torch.Tensor):
        input_ = self._prepare_input(
            input_=x.clone(), joints_to_use=self.skeleton.medium
        )
        out_current = self.medium_res(input_)
        out = self.mask_s_m(previous_output, out_current)
        return out

    def _step3(self, x: torch.Tensor, previous_output: torch.Tensor) -> torch.Tensor:
        input_ = self._prepare_input(input_=x.clone(), joints_to_use=self.skeleton.full)
        out_current = self.full_res(input_)
        out = self.mask_m_f(previous_output, out_current)
        return out

    def _prepare_input(self, input_: torch.Tensor, joints_to_use: List):
        # [b l c] -> [b c l]
        input_ = self.arrange(input_)
        batch_size, channels, seq_len = input_.shape
        input_ = input_.reshape(batch_size, channels // 3, 3, seq_len)
        input_ = input_[:, joints_to_use, :, :].reshape(
            batch_size, len(joints_to_use) * 3, seq_len
        )
        assert input_.shape == torch.Size([batch_size, len(joints_to_use) * 3, seq_len])
        return input_

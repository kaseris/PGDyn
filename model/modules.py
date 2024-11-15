import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

import math

from typing import List, Tuple, Union

from utils.skeletons import SkeletonH36M, SkeletonAMASS


class GraphConvolution(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 16,
        n_joints: int = 22,
        seq_len: int = 35,
        bias: bool = True,
    ):
        super().__init__()
        self.channels = in_channels
        self.out_channels = out_channels
        self.n_joints = n_joints
        self.seq_len = seq_len

        self.spatial_adjacency_matrix = nn.Parameter(
            torch.FloatTensor(n_joints, n_joints)
        )
        self.temporal_adjacency_matrix = nn.Parameter(
            torch.FloatTensor(seq_len, seq_len)
        )

        self.weight_c = nn.Parameter(torch.FloatTensor(in_channels, out_channels))

        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(seq_len))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.spatial_adjacency_matrix.size(1))
        self.weight_c.data.uniform_(-stdv, stdv)
        self.temporal_adjacency_matrix.data.uniform_(-stdv, stdv)
        self.spatial_adjacency_matrix.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Expect an input of [batch_size, seq_len, n_joints, c]
        """
        # [batch_size, seq_len, n_joints, C] -> [batch_size, seq_len, n_joints, C]
        support = torch.matmul(self.spatial_adjacency_matrix, x)
        # [batch_size, seq_len, n_joints, C] -> [batch_size, seq_len, n_joints, out_channels]
        out = torch.matmul(support, self.weight_c)
        output = (
            torch.matmul(out.permute(0, 2, 3, 1), self.temporal_adjacency_matrix)
            .permute(0, 2, 1, 3)
            .contiguous()
        )

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class GCBlock(nn.Module):
    def __init__(
        self,
        channels: int,
        dropout: float = 0.3,
        bias: bool = True,
        n_joints: int = 22,
        seq_len: int = 20,
    ):
        super().__init__()
        self.in_channels = channels
        self.out_channels = channels

        self.gc1 = GraphConvolution(
            in_channels=channels,
            out_channels=channels,
            n_joints=n_joints,
            seq_len=seq_len,
            bias=bias,
        )
        self.bn1 = nn.BatchNorm1d(channels * n_joints * seq_len)

        self.gc2 = GraphConvolution(
            in_channels=channels,
            out_channels=channels,
            n_joints=n_joints,
            seq_len=seq_len,
            bias=bias,
        )
        self.bn2 = nn.BatchNorm1d(channels * n_joints * seq_len)

        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        The block expects a sequence of shape [batch_size, seq_len, n_joints, dim]

        Returns a tensor of shape [batch_size, seq_len, n_joints, dim]
        """
        out = self.gc1(x)
        batch_size, channels, n_joints, seq_len = out.shape
        out = out.view(batch_size, -1).contiguous()
        out = self.bn1(out).view(batch_size, seq_len, n_joints, channels).contiguous()
        out = self.act(out)
        out = self.dropout(out)

        out = self.gc2(out)
        batch_size, channels, n_joints, seq_len = out.shape
        out = out.view(batch_size, -1).contiguous()
        out = self.bn2(out).view(batch_size, seq_len, n_joints, channels).contiguous()
        out = self.act(out)
        out = self.dropout(out)
        return x + out


class LN(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, dim, 1]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, dim, 1]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class LN_v2(nn.Module):
    def __init__(self, dim, epsilon=1e-5):
        super().__init__()
        self.epsilon = epsilon

        self.alpha = nn.Parameter(torch.ones([1, 1, dim]), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros([1, 1, dim]), requires_grad=True)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        y = y * self.alpha + self.beta
        return y


class TemporalFC(nn.Module):
    """
    Expects a tensor of shape [batch_size, seq_len, n_joints * dims]
    """

    def __init__(self, seq_len: int = 30, hidden_dim: int = 30):
        super().__init__()
        self.fc0 = nn.Linear(seq_len, seq_len, bias=False)
        # self.act = nn.Tanh()
        # self.fc1 = nn.Linear(hidden_dim, seq_len)
        self.rearr0 = Rearrange("b l n -> b n l")
        self.rearr1 = Rearrange("b n l -> b l n")
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.weight, gain=1e-7)
        # nn.init.xavier_uniform_(self.fc1.weight, gain=1e-9)

        # nn.init.constant_(self.fc0.bias, 0)
        # nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rearr0(x)
        x = self.fc0(x)
        # x = self.act(x)
        # x = self.fc1(x)
        x = self.rearr1(x)
        return x


class SpatialFC(nn.Module):
    """
    Expects a tensor of shape [batch_size, seq_len, n_joints * dims]
    """

    def __init__(self, dims: int = 66, hidden_dim: int = 66):
        super().__init__()
        self.fc0 = nn.Linear(in_features=dims, out_features=hidden_dim)
        self.act = nn.Tanh()
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=dims)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.fc0.weight, gain=1e-9)
        nn.init.xavier_uniform_(self.fc1.weight, gain=1e-9)

        nn.init.constant_(self.fc0.bias, 0)
        nn.init.constant_(self.fc1.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc0(x)
        x = self.act(x)
        x = self.fc1(x)
        return x


class GatingMechanism(nn.Module):
    def __init__(self, dim, mlp):
        super().__init__()
        self.gate = mlp
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.gate.weight, gain=1.0)
        nn.init.zeros_(self.gate.bias)

    def forward(
        self,
        x: torch.Tensor,
        temporal_out: torch.Tensor,
        spatial_out: torch.Tensor,
        tau: float = 1.01,
    ):
        x = x.mean(1)
        logits = self.gate(x)
        probs = F.gumbel_softmax(logits=logits, tau=tau, hard=True, dim=-1)
        spatial_out = spatial_out.unsqueeze(1)
        temporal_out = temporal_out.unsqueeze(1)
        options = torch.cat([spatial_out, temporal_out], dim=1)
        selected = torch.einsum("bj,bjld->bld", probs, options)
        return selected


class MLP(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        dims: int = 66,
        seq_len: int = 30,
        use_norm: bool = True,
        layernorm_axis: str = "spatial",
    ):
        super().__init__()
        self.mlp = mlp
        self.spatial_fc = SpatialFC(dims=dims)
        self.temporal_fc = TemporalFC(seq_len=seq_len, hidden_dim=seq_len)
        # TODO: Keep in mind shape consistency. Might be [bs, dims, seq_len] in siMLPe implementation
        norms = {
            "spatial": LN_v2(dim=dims),
            "temporal": LN(dim=seq_len),
            "all": nn.LayerNorm([dims, seq_len]),
        }
        self.norm = (
            norms.get(layernorm_axis, LN(dim=dims)) if use_norm else nn.Identity()
        )

        self.gate = GatingMechanism(dim=dims, mlp=self.mlp)

    def forward(self, x: torch.Tensor, tau: float = 10.0):
        spatial_out = self.spatial_fc(x)
        temporal_out = self.temporal_fc(x)

        selected = self.gate(x, temporal_out, spatial_out, tau)
        out = self.norm(selected)
        out = x + out
        return out


class MultiLayerMLP(nn.Module):
    def __init__(
        self,
        mlp: nn.Module,
        dims: int = 66,
        seq_len: int = 30,
        use_norm: bool = True,
        layernorm_axis: str = "spatial",
        num_layers: int = 50,
    ):
        super().__init__()
        self.mlps = nn.ModuleList(
            [
                MLP(
                    mlp=mlp,
                    dims=dims,
                    seq_len=seq_len,
                    use_norm=use_norm,
                    layernorm_axis=layernorm_axis,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x: torch.Tensor, tau: float = 1.01):
        for idx, block in enumerate(self.mlps):
            if idx == 0:
                out = block(x, tau)
            else:
                out = block(out, tau)
        return out


def generate_adjacency_matrix(connections: List[Tuple], n_joints=22):
    adjacency_matrix = torch.zeros(
        n_joints, n_joints, dtype=torch.float32, requires_grad=False
    )
    for node0, node1 in connections:
        adjacency_matrix[node0, node1] = 1
        adjacency_matrix[node1, node0] = 1
    return adjacency_matrix


class SpatialGC(nn.Module):
    def __init__(
        self,
        skeleton: Union[SkeletonAMASS, SkeletonH36M],
        n_joints: int = 8,
        skeleton_resolution: str = "small",
        normalize_adj_mask: bool = True,
    ):
        super().__init__()
        self.normalize_adj_mask = normalize_adj_mask

        assert skeleton_resolution in ["small", "medium", "full"]
        if skeleton_resolution == "small":
            connections = skeleton.connections_small
        elif skeleton_resolution == "medium":
            connections = skeleton.connections_medium
        else:
            connections = skeleton.connections_full
        adj_mask = generate_adjacency_matrix(connections=connections, n_joints=n_joints)
        self.register_buffer("adj_mask", adj_mask)
        self.adj = nn.Parameter(torch.eye(n_joints), requires_grad=True)

    def forward(self, x: torch.Tensor):
        # The input is expected to be of shape [batch_size, n_joints*dims, seq_len]
        batch_size, channels, seq_len = x.shape
        x1 = x.reshape(batch_size, channels // 3, 3, seq_len)
        if self.normalize_adj_mask:
            adj_mask = self.adj_mask
            deg = adj_mask.sum(dim=1)

            # Compute the normalization term
            deg_inv_sqrt = torch.pow(deg, -0.5)
            deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0
            norm = deg_inv_sqrt.unsqueeze(1) * deg_inv_sqrt.unsqueeze(0)

            # Normalize adjacency matrix
            adj_mask = adj_mask * norm
        else:
            adj_mask = self.adj_mask

        x1 = torch.einsum("vj,bjct->bvct", self.adj.mul(adj_mask), x1)
        x1 = x1.reshape(batch_size, channels, seq_len)
        return x1


class TemporalGC(nn.Module):
    def __init__(self, local_window: int = 1, seq_len: int = 50):
        super().__init__()
        traj_mask = torch.tril(torch.ones(seq_len, seq_len), local_window) * torch.triu(
            torch.ones(seq_len, seq_len), -local_window
        )
        self.register_buffer("traj_mask", traj_mask)
        self.adj = nn.Parameter(torch.zeros(seq_len, seq_len))

    def forward(self, x: torch.Tensor):
        # Assume that the input is of shape [batch_size, dims, seq_len]
        return torch.einsum("ft,bnt->bnf", self.adj.mul(self.traj_mask), x)


class JointChannelGC(nn.Module):
    def __init__(self, n_joints: int = 22, dims: int = 3):
        super().__init__()
        self.adj_jc = nn.Parameter(
            torch.zeros(n_joints, dims, dims), requires_grad=True
        )
        self.dims = dims

    def forward(self, x: torch.Tensor):
        """Assume that the input [bs, 66, 50]"""
        batch_size, channels, seq_len = x.shape
        x = x.reshape(batch_size, channels // self.dims, self.dims, seq_len)
        x = torch.einsum("jkc,bjct->bjkt", self.adj_jc, x)
        x = x.reshape(batch_size, channels, seq_len)
        return x


class SpatioTemporalGC(nn.Module):
    def __init__(
        self,
        dims: int = 3,
        n_joints: int = 22,
        seq_len: int = 50,
        local_window: int = 1,
    ):
        super().__init__()
        traj_mask = torch.tril(torch.ones(seq_len, seq_len), local_window) * torch.triu(
            torch.ones(seq_len, seq_len), -local_window
        )
        self.register_buffer("traj_mask", traj_mask)
        self.adj_tj = nn.Parameter(
            torch.zeros(dims * n_joints, seq_len, seq_len), requires_grad=True
        )

    def forward(self, x: torch.Tensor):
        x = torch.einsum(
            "nft,bnt->bnf", self.adj_tj.mul(self.traj_mask.unsqueeze(0)), x
        )
        return x


class Gate(nn.Module):
    def __init__(self, mlp: nn.Parameter, is_dynamic: bool = True, tau: float = 5.0):
        super().__init__()
        self.mlp = mlp
        self.is_dynamic = is_dynamic
        self.tau = 5.0

    def forward(
        self,
        x: torch.Tensor,
        x1: torch.Tensor,
        x2: torch.Tensor,
        x3: torch.Tensor,
        x4: torch.Tensor,
    ):
        prob = torch.einsum("bj,jk->bk", x.mean(1), self.mlp)
        if self.is_dynamic:
            gate = gate = F.gumbel_softmax(prob, tau=self.tau, hard=True)
        else:
            gate = (
                torch.tensor([1.0, 0.0, 0.0, 0.0])
                .unsqueeze(0)
                .expand(x.shape[0], -1)
                .to(x.device)
            )
        return gate


class DynamicLayer(nn.Module):
    def __init__(
        self,
        skeleton: Union[SkeletonAMASS, SkeletonH36M],
        mlp: nn.Parameter,
        n_joints: int = 22,
        seq_len: int = 50,
        skeleton_resolution: str = "small",
        normalize_adj_mask: bool = False,
        local_window: int = 3,
        dims: int = 3,
        is_dynamic: bool = True,
        layer_norm_axis: str = "spatial",
    ):
        super().__init__()
        self.update = nn.Linear(in_features=seq_len, out_features=seq_len)
        self.spatial_gc = SpatialGC(
            skeleton=skeleton,
            n_joints=n_joints,
            skeleton_resolution=skeleton_resolution,
            normalize_adj_mask=normalize_adj_mask,
        )
        self.temporal_gc = TemporalGC(local_window=local_window, seq_len=seq_len)
        self.joint_channel_gc = JointChannelGC(n_joints=n_joints, dims=dims)
        self.spatio_temporal_gc = SpatioTemporalGC(
            dims=dims, n_joints=n_joints, seq_len=seq_len, local_window=local_window
        )
        self.gate = Gate(mlp=mlp, is_dynamic=is_dynamic)

        if layer_norm_axis == "spatial":
            self.norm = LN(dim=dims * n_joints)
        elif layer_norm_axis == "temporal":
            self.norm = LN_v2(dim=seq_len)
        elif layer_norm_axis == "all":
            self.norm = nn.LayerNorm(normalized_shape=[dims * n_joints, seq_len])
        else:
            self.norm = nn.Identity()

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.update.weight, gain=1e-8)
        nn.init.constant_(self.update.bias, 0)

    def forward(self, x: torch.Tensor):
        x1 = self.spatial_gc(x)
        x2 = self.temporal_gc(x)
        x3 = self.joint_channel_gc(x)
        x4 = self.spatio_temporal_gc(x)

        gate = self.gate(x, x1, x2, x3, x4)
        x2, x3, x4 = x2.unsqueeze(1), x3.unsqueeze(1), x4.unsqueeze(1)
        options = torch.cat(
            [torch.zeros_like(x1).to(x.device).unsqueeze(1), x2, x3, x4], dim=1
        )

        x_ = torch.einsum("bj,bjvt->bvt", gate, options)
        x_ = self.update(x1 + x_)
        x_ = self.norm(x_)
        x = x + x_
        return x


class ResolutionLevelV2(nn.Module):
    def __init__(
        self,
        skeleton: Union[SkeletonAMASS, SkeletonH36M],
        mlp: nn.Parameter,
        n_joints: int = 8,
        skeleton_resolution: str = "small",
        normalize_adj_mask: bool = True,
        local_window: int = 5,
        seq_len: int = 50,
        dims: int = 3,
        is_dynamic: bool = True,
        layer_norm_axis: str = "spatial",
        n_dynamic_layers: int = 8,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.dims = dims
        self.layers = nn.Sequential(
            *[
                DynamicLayer(
                    mlp=mlp,
                    skeleton=skeleton,
                    n_joints=n_joints,
                    seq_len=seq_len,
                    skeleton_resolution=skeleton_resolution,
                    normalize_adj_mask=normalize_adj_mask,
                    local_window=local_window,
                    dims=dims,
                    is_dynamic=is_dynamic,
                    layer_norm_axis=layer_norm_axis,
                )
                for _ in range(n_dynamic_layers)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layers(x)
        return x


class CrossAttention(nn.Module):
    def __init__(self, query_dim, key_dim, num_heads=8, head_dim=64):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim

        self.query_proj = nn.Linear(query_dim, head_dim * num_heads)
        self.key_proj = nn.Linear(key_dim, head_dim * num_heads)
        self.value_proj = nn.Linear(key_dim, head_dim * num_heads)

        self.output_proj = nn.Linear(head_dim * num_heads, query_dim)

    def forward(self, query, key, value):
        batch_size = query.size(0)

        # Project and reshape
        query = (
            self.query_proj(query)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = (
            self.key_proj(key)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        value = (
            self.value_proj(value)
            .view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Compute attention
        scores = torch.matmul(query, key.transpose(-2, -1)) / (self.head_dim**0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, value)

        # Reshape and project output
        attn_output = (
            attn_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        output = self.output_proj(attn_output)

        return output


class CrossAttentiveGC(nn.Module):
    def __init__(self, n_joints=8, seq_len=50, dims=3, skeleton_resolution="small"):
        super().__init__()
        self.n_joints = n_joints
        self.seq_len = seq_len
        self.dims = dims

        spatial_dim = n_joints * dims
        temporal_dim = seq_len

        self.spatial_temporal_attn = CrossAttention(spatial_dim, temporal_dim)
        self.temporal_spatial_attn = CrossAttention(temporal_dim, spatial_dim)

    def forward(self, spatial_features: torch.Tensor, temporal_features: torch.Tensor):
        # x shape: [batch_size, n_joints*dims, seq_len]
        batch_size, channels, seq_len = spatial_features.shape
        n_joints = channels // self.dims

        # Reshape for cross attention
        spatial_features_reshaped = spatial_features.transpose(1, 2).reshape(
            batch_size, seq_len, -1
        )
        temporal_features_reshaped = temporal_features.reshape(
            batch_size, n_joints * self.dims, -1
        )

        # Apply cross attention
        enhanced_spatial = self.spatial_temporal_attn(
            spatial_features_reshaped,
            temporal_features_reshaped,
            temporal_features_reshaped,
        )
        enhanced_temporal = self.temporal_spatial_attn(
            temporal_features_reshaped,
            spatial_features_reshaped,
            spatial_features_reshaped,
        )

        # Reshape back to original format
        enhanced_spatial = enhanced_spatial.reshape(
            batch_size, seq_len, n_joints, self.dims
        ).permute(0, 2, 3, 1)
        enhanced_temporal = enhanced_temporal.reshape(
            batch_size, n_joints, self.dims, seq_len
        )

        # Combine enhanced features
        combined = enhanced_spatial + enhanced_temporal
        return combined


class ResolutionLevel(nn.Module):
    def __init__(
        self,
        n_joints: int = 8,
        skeleton_resolution: str = "small",
        normalize_adj_mask: bool = True,
        local_window: int = 5,
        seq_len: int = 50,
        dims: int = 3,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.dims = dims
        self.spatial_gc = SpatialGC(
            n_joints=n_joints,
            skeleton_resolution=skeleton_resolution,
            normalize_adj_mask=normalize_adj_mask,
        )
        self.temporal_gc = TemporalGC(local_window=local_window, seq_len=seq_len)
        self.cross_attention = CrossAttentiveGC(
            n_joints=n_joints,
            seq_len=seq_len,
            dims=dims,
            skeleton_resolution=skeleton_resolution,
        )

        self.norm = LN(dim=n_joints * 3)
        self.temporal_fc = nn.Linear(seq_len, seq_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, seq_len = x.shape
        spatial_features = self.spatial_gc(x)
        temporal_features = self.temporal_gc(x)
        combined = self.cross_attention(spatial_features, temporal_features)
        combined = combined.reshape(batch_size, self.n_joints * self.dims, seq_len)
        out = self.norm(combined)
        out = self.temporal_fc(out + x)
        return out.tanh()


class MaskOutput(nn.Module):
    """Copies the output from the previous level to the output of the current level"""

    def __init__(
        self,
        skeleton: Union[SkeletonH36M, SkeletonAMASS],
        resolution: str = "medium",
        trainable: bool = False,
        n_joints: int = 13,
        dims: int = 3,
        seq_len: int = 50,
    ):
        super().__init__()
        self.n_joints = n_joints
        self.dims = dims
        self.seq_len = seq_len

        if resolution == "medium":
            self.previous_resolution_indices = skeleton.small
            self.correspondences = skeleton.correspondence_small_to_medium
            self.current_resolution_indices = skeleton.medium
        elif resolution == "full":
            self.previous_resolution_indices = skeleton.medium
            self.correspondences = skeleton.correspondence_medium_to_full
            self.current_resolution_indices = [i for i in range(n_joints)]
        mask_indices = torch.tensor(
            [
                self.correspondences[i]
                for i in range(len(self.previous_resolution_indices))
            ]
        )
        self.register_buffer("mask_indices", mask_indices)
        weight_mask = torch.ones(len(self.current_resolution_indices), 1, 1)
        weight_mask[self.mask_indices] = 0.0
        self.weight = nn.Parameter(weight_mask, requires_grad=trainable)

    def forward(
        self,
        previous_resolution_output: torch.Tensor,
        current_resolution_output: torch.Tensor,
    ):
        _maybe_need_reshape = previous_resolution_output.ndim == 3
        batch_size = previous_resolution_output.shape[0]
        if _maybe_need_reshape:
            previous_resolution_output = previous_resolution_output.reshape(
                batch_size,
                len(self.previous_resolution_indices),
                self.dims,
                self.seq_len,
            )
            current_resolution_output = current_resolution_output.reshape(
                batch_size,
                len(self.current_resolution_indices),
                self.dims,
                self.seq_len,
            )

        previous_resolution_output_ = torch.zeros(
            batch_size, self.n_joints, self.dims, self.seq_len
        ).to(previous_resolution_output.device)
        previous_resolution_output_[:, self.mask_indices, :, :] = (
            previous_resolution_output.clone()
        )
        pred_current_ = (
            self.weight.multiply(current_resolution_output)
            + previous_resolution_output_
        )
        if _maybe_need_reshape:
            return pred_current_.reshape(
                batch_size, self.n_joints * self.dims, self.seq_len
            )
        return pred_current_


class MaskInput(nn.Module): ...


if __name__ == "__main__":
    x = torch.randn(1, 30, 66)
    tfc = TemporalFC(seq_len=30, hidden_dim=30)
    tfc_out = tfc(x)
    print(f"after tfc: {tfc_out.shape}")
    sfc = SpatialFC(dims=66, hidden_dim=66)
    sfc_out = sfc(x)
    print(f"after sfc: {sfc_out.shape}")
    gating = GatingMechanism(dim=66)
    out = gating(x, tfc_out, sfc_out, 1.05)
    print(f"after gating: {out.shape}")

    multimlp = MultiLayerMLP(num_layers=48)
    print(
        f"Num of trainable parameters: {sum([p.numel() for p in multimlp.parameters()])}"
    )

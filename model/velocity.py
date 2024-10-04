import torch
import torch.nn as nn

from typing import List


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, bi_dir=False, cell_type="gru", num_layers=1):
        super(RNN, self).__init__()
        self.bi_dir = bi_dir
        self.hidden_size = hidden_size
        if cell_type == "gru":
            self.rnn = nn.GRU(
                input_size, hidden_size, bidirectional=bi_dir, batch_first=True, num_layers=num_layers
            )
        else:
            raise ValueError(f"Unsupported cell type: {cell_type}")

    def forward(self, x):
        output, _ = self.rnn(x)
        if self.bi_dir:
            return output[:, -1, : self.hidden_size] + output[:, 0, self.hidden_size :]
        else:
            return output[:, -1, :]


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, activation="relu"):
        super(MLP, self).__init__()
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            else:
                raise ValueError(f"Unsupported activation: {activation}")
            prev_dim = dim
        self.layers = nn.Sequential(*layers)
        self.out_dim = prev_dim

    def forward(self, x):
        return self.layers(x)


class GRUAutoencoder(nn.Module):
    def __init__(self, nx, ny, horizon, specs):
        super(GRUAutoencoder, self).__init__()
        self.nx = nx
        self.ny = ny
        self.horizon = horizon
        self.x_birnn = specs.get("x_birnn", True)
        self.e_birnn = specs.get("e_birnn", True)
        self.use_drnn_mlp = specs.get("use_drnn_mlp", False)
        self.nh_rnn = nh_rnn = specs.get("nh_rnn", 128)
        self.nh_mlp = nh_mlp = specs.get("nh_mlp", [300, 200])

        # Encoder
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=self.x_birnn, cell_type="gru")
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=self.e_birnn, cell_type="gru")
        self.e_mlp = MLP(2 * nh_rnn, nh_mlp)
        self.e_out = nn.Linear(self.e_mlp.out_dim, nh_rnn)

        # Decoder
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation="tanh")
        self.d_rnn = nn.GRU(ny + nh_rnn + nh_rnn, nh_rnn, batch_first=True)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)

    def encode_x(self, x):
        return self.x_rnn(x)

    def encode_y(self, y):
        return self.e_rnn(y)

    def encode(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)

        h = torch.cat((h_x, h_y), dim=1)

        h = self.e_mlp(h)
        return self.e_out(h)

    def decode(self, x, z):
        h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            h_0 = h_d.unsqueeze(0)
        else:
            h_0 = torch.zeros(1, z.shape[0], self.nh_rnn, device=z.device)

        y = []
        h = h_0
        for i in range(self.horizon):
            y_p = x[:, -1, :] if i == 0 else y_i
            rnn_in = torch.cat([h_x, z, y_p], dim=1).unsqueeze(1)
            _, h = self.d_rnn(rnn_in, h)
            h_out = self.d_mlp(h.squeeze(0))
            y_i = self.d_out(h_out)
            y.append(y_i)
        y = torch.stack(y, dim=1)
        return y

    def forward(self, x, y):
        z = self.encode(x, y)
        return self.decode(x, z)


class LatentVelocityProjection(nn.Module):
    def __init__(
        self,
        nx: int = 66,
        ny: int = 66,
        horizon: int = 24,
        nh_rnn: int = 128,
        bi_dir_e_rnn: bool = True,
        bi_dir_x_rnn: bool = True,
        nh_mlp: List = [512, 512],
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.horizon = horizon

        self.x_rnn = RNN(nx, nh_rnn, bi_dir=bi_dir_x_rnn, cell_type="gru", num_layers=2)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=bi_dir_e_rnn, cell_type="gru", num_layers=2)

        self.e_mlp = MLP(2 * nh_rnn, nh_mlp)
        self.e_out = nn.Linear(self.e_mlp.out_dim, nh_rnn)

    def encode_x(self, x):
        return self.x_rnn(x)

    def encode_y(self, y):
        return self.e_rnn(y)

    def forward(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)
        h = torch.cat((h_x, h_y), dim=1)
        h = self.e_mlp(h)
        return self.e_out(h)
    

class EnhancedLatentVelocityProjection(nn.Module):
    def __init__(
        self,
        nx: int = 66,
        ny: int = 66,
        horizon: int = 24,
        nh_rnn: int = 256,
        bi_dir_e_rnn: bool = True,
        bi_dir_x_rnn: bool = True,
        nh_mlp: List[int] = [1024, 1024, 512],
        dropout: float = 0.1,
        use_attention: bool = True,
        use_residual: bool = True,
        num_layers: int = 3,
    ):
        super().__init__()
        self.nx = nx
        self.ny = ny
        self.horizon = horizon
        self.use_attention = use_attention
        self.use_residual = use_residual

        # Enhanced RNNs with LSTM cells and increased hidden size
        self.x_rnn = nn.LSTM(nx, nh_rnn, bidirectional=bi_dir_x_rnn, num_layers=num_layers, dropout=dropout, batch_first=True)
        self.e_rnn = nn.LSTM(ny, nh_rnn, bidirectional=bi_dir_e_rnn, num_layers=num_layers, dropout=dropout, batch_first=True)

        rnn_output_size = nh_rnn * 2 if bi_dir_x_rnn and bi_dir_e_rnn else nh_rnn

        # Attention mechanism
        if use_attention:
            self.attention = nn.MultiheadAttention(rnn_output_size, num_heads=8, dropout=dropout)

        # Enhanced MLP with more layers, larger hidden sizes, and dropout
        mlp_layers = []
        in_features = rnn_output_size * 2
        for out_features in nh_mlp:
            mlp_layers.extend([
                nn.Linear(in_features, out_features),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.BatchNorm1d(out_features)
            ])
            in_features = out_features

        self.mlp = nn.Sequential(*mlp_layers)

        # Output layer
        self.out = nn.Linear(in_features, 128)  # Assuming the latent code size is 128

        # Layer normalization
        self.layer_norm = nn.LayerNorm(128)

    def encode_x(self, x):
        output, _ = self.x_rnn(x)
        return output[:, -1, :]  # Return the last hidden state

    def encode_y(self, y):
        output, _ = self.e_rnn(y)
        return output[:, -1, :]  # Return the last hidden state

    def forward(self, x, y):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y)

        if self.use_attention:
            # Apply self-attention to the concatenated features
            h_combined = torch.cat((h_x.unsqueeze(0), h_y.unsqueeze(0)), dim=0)
            h_attended, _ = self.attention(h_combined, h_combined, h_combined)
            h = h_attended.transpose(0, 1).reshape(h_x.size(0), -1)
        else:
            h = torch.cat((h_x, h_y), dim=1)

        h_mlp = self.mlp(h)
        
        out = self.out(h_mlp)
        
        if self.use_residual:
            # Add a residual connection
            out = out + h[:, :128]  # Assuming the latent code size is 128
        
        return self.layer_norm(out)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=3, order=2):
        super().__init__()
        self.nconv = nn.Sequential(
            nn.Conv2d(c_in * support_len, c_out, kernel_size=(1, 1)),
            nn.ReLU()
        )
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, x, support):
        out = []
        for a in support:
            out.append(torch.einsum('ncvl,vw->ncwl', (x, a)))
        h = torch.cat(out, dim=1) # Nối các kết quả lại
        h = self.nconv(h)         # Dùng một lớp conv duy nhất để đưa về đúng số kênh c_out
        h = self.dropout_layer(h)
        return h

class GraphWaveNet(nn.Module):
    def __init__(self, num_nodes, dropout=0.3, supports=None, in_dim=1, out_dim=12,
                residual_channels=32, dilation_channels=32, skip_channels=256,
                end_channels=512, kernel_size=2, blocks=4, layers=2):
        super().__init__()
        self.dropout = dropout
        self.blocks = blocks
        self.layers = layers

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=residual_channels, kernel_size=(1, 1))
        self.supports = supports
        
        # Adaptive adjacency matrix
        self.node_embeddings = nn.Parameter(torch.randn(num_nodes, 10), requires_grad=True)
        
        self.tcn_layers = nn.ModuleList()
        self.gcn_layers = nn.ModuleList()
        self.skip_convs = nn.ModuleList()
        self.bn = nn.ModuleList()

        receptive_field = 1
        for b in range(blocks):
            for l in range(layers):
                dilation = 2 ** l
                self.tcn_layers.append(nn.Conv2d(residual_channels, dilation_channels, kernel_size=(1, kernel_size), dilation=dilation))
                gcn_support_len = (len(self.supports) if self.supports is not None else 0) + 1
                self.gcn_layers.append(GCN(dilation_channels, residual_channels, dropout, support_len=gcn_support_len))
                self.skip_convs.append(nn.Conv2d(dilation_channels, skip_channels, kernel_size=(1, 1)))
                self.bn.append(nn.BatchNorm2d(residual_channels))
                receptive_field += dilation

        self.end_conv_1 = nn.Conv2d(skip_channels, end_channels, kernel_size=(1, 1))
        self.end_conv_2 = nn.Conv2d(end_channels, out_dim, kernel_size=(1, 1))

        self.receptive_field = receptive_field

    def forward(self, x):
        # x shape: (batch_size, seq_len, num_nodes, in_dim) -> (batch, in_dim, num_nodes, seq_len)
        x = x.permute(0, 3, 2, 1)
        
        adp_adj = F.softmax(F.relu(torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))), dim=1)
        
        # Nếu self.supports là None, coi nó là list rỗng
        base_supports = self.supports if self.supports is not None else []
        current_supports = base_supports + [adp_adj]
        
        x = self.start_conv(x)
        skip = 0

        for i in range(self.blocks * self.layers):
            residual = x
            
            # TCN (Dilated Causal Convolution)
            filter_ = torch.tanh(self.tcn_layers[i](residual))
            gate = torch.sigmoid(self.tcn_layers[i](residual))
            x = filter_ * gate
            
            # Skip connection
            s = self.skip_convs[i](x)
            if isinstance(skip, torch.Tensor):
                skip = skip[:, :, :, -s.size(3):]
            skip = s + skip

            # GCN
            x = self.gcn_layers[i](x, current_supports)
            
            # Residual connection
            x = x + residual[:, :, :, -x.size(3):]
            x = self.bn[i](x)
            
        assert isinstance(skip, torch.Tensor)
        x = F.relu(skip)
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x) # (batch, out_dim, num_nodes, 1)
        if x.dim() == 4:
            x = x[:, :, :, -1] # (batch, out_dim, num_nodes)
        return x.permute(0, 2, 1) # (batch, num_nodes, out_dim)
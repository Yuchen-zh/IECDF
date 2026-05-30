import torch
import torch.nn as nn


class AM_Layer(nn.Module):
    def __init__(self, self_attention, mamba, d_model, dropout):
        super(AM_Layer, self).__init__()
        self.self_attention = self_attention
        self.mamba = mamba
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, x_=None, x_mask=None):
        if x_ is not None:
            x = x_ + self.dropout(self.self_attention(
                x_, x, x,
                attn_mask=x_mask
            )[0])
        else:
            x = x + self.dropout(self.self_attention(
                x, x, x,
                attn_mask=x_mask
            )[0])
        x = self.norm1(x)

        x_mamba = self.mamba(x)

        x = x + x_mamba
        x = self.norm2(x)

        return x

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(torch.nn.Module):
    def __init__(self, input_dim, embed_dims, dropout):
        super(MLP, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, 2))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_fusion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(MLP_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_gate_fusion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(MLP_gate_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.LayerNorm(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.size()
        x = x.view(batch_size, -1)
        x = self.mlp(x)
        return x


class MLP_share_gate_fusion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(MLP_share_gate_fusion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.LayerNorm(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MLP_trans(torch.nn.Module):
    def __init__(self, input_size, out_size, dropout=0.2):
        super(MLP_trans, self).__init__()
        self.dropout = torch.nn.Dropout(dropout)
        self.activate = torch.nn.SiLU()
        self.mlp_1 = nn.Linear(input_size, out_size)
        self.mlp_2 = nn.Linear(out_size, out_size)

    def forward(self, emb_trans):
        emb_trans = self.dropout(self.activate(self.mlp_1(emb_trans)))
        emb_trans = self.dropout(self.activate(self.mlp_2(emb_trans)))
        return emb_trans


class clip_fuion(torch.nn.Module):
    def __init__(self, input_dim, out_dim, embed_dims, dropout):
        super(clip_fuion, self).__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.GELU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        layers.append(torch.nn.Linear(input_dim, out_dim))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class cnn_extractor(torch.nn.Module):
    def __init__(self, input_size, feature_kernel):
        super(cnn_extractor, self).__init__()
        self.convs = torch.nn.ModuleList(
            [torch.nn.Conv1d(input_size, feature_num, kernel)
             for kernel, feature_num in feature_kernel.items()]
        )

    def forward(self, input_data):
        input_data = input_data.permute(0, 2, 1).to('cuda')
        feature = [F.relu(conv(input_data)) for conv in self.convs]
        feature = [F.max_pool1d(f, f.shape[-1]) for f in feature]
        feature = torch.cat(feature, dim=1)
        feature = F.dropout(feature, p=0.5, training=self.training)
        feature = feature.view(-1, feature.shape[1])
        return feature


class MaskAttention(torch.nn.Module):
    def __init__(self, input_dim):
        super(MaskAttention, self).__init__()
        self.Line = torch.nn.Linear(input_dim, 1)

    def forward(self, input, mask):
        score = self.Line(input).view(-1, input.size(1))
        if mask is not None:
            score = score.masked_fill(mask == 0, float("-inf"))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score, input).squeeze(1)
        return output


class TokenAttention(torch.nn.Module):
    def __init__(self, input_shape):
        super(TokenAttention, self).__init__()
        self.attention_layer = nn.Sequential(
            torch.nn.Linear(input_shape, input_shape),
            nn.SiLU(),
            torch.nn.Linear(input_shape, 1),
        )

    def forward(self, inputs):
        scores = self.attention_layer(inputs).view(-1, inputs.size(1))
        scores = scores.unsqueeze(1)
        outputs = torch.matmul(scores, inputs).squeeze(1)
        return outputs, scores

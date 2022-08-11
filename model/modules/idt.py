import torch
import torch.nn as nn


def feature_map2vec(feature_map, mask):
    batch_size = feature_map.size(0)
    num_channel = feature_map.size(1)
    num_part = mask.size(1)

    feature_map = feature_map.unsqueeze(2)
    sum_of_weight = mask.view(batch_size, num_part, -1).sum(-1) + 1e-12
    mask = mask.unsqueeze(1)

    vec = (feature_map * mask).view(batch_size, num_channel, num_part, -1).sum(-1)
    vec = vec / sum_of_weight.unsqueeze(1)
    vec = vec.view(batch_size, num_channel * num_part)

    return vec


class ConvNet(nn.Module):
    def __init__(self, dim, embed_num, alpha, iters, **kwargs):
        super(ConvNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim, embed_num, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, x):
        b = x.shape[0]
        mask = self.conv(x)
        mask = nn.Sigmoid()(mask)
        n = mask.shape[1]
        feature_vector = feature_map2vec(x, mask)
        return feature_vector.reshape(b, n, -1)


class MetaPartModule(nn.Module):
    def __init__(self, dim, embed_num, iters=3, alpha=0.1):
        super(MetaPartModule, self).__init__()
        self.iters = iters
        self.eps = 1e-12
        self.alpha = alpha

        self.meta_embed = nn.Parameter(torch.randn(embed_num, dim), requires_grad=True)
        self.to_q = nn.Sequential(nn.Linear(dim, dim), nn.BatchNorm1d(dim, track_running_stats=False), nn.ReLU())
        self.to_k = nn.Identity()
        self.mask_norm = nn.Sequential(nn.BatchNorm1d(embed_num, track_running_stats=False), nn.ReLU())

    def forward(self, x):
        b, c, h, w = x.shape
        inputs = x.reshape(b, c, -1).permute(0, 2, 1)
        b, _, d = inputs.shape

        slots = self.meta_embed.unsqueeze(0).expand(b, -1, -1)
        k, v = self.to_k(inputs.reshape(-1, d)).reshape(b, -1, d), inputs
        new_slots = None
        multi_slots = []
        for _ in range(self.iters):
            if new_slots is not None:
                slots = new_slots.reshape(b, -1, d) * self.alpha + slots * (1 - self.alpha)
            q_slots = self.to_q(slots.reshape(-1, d)).reshape(b, -1, d)
            attn = torch.einsum('bid,bjd->bij', q_slots, k)
            attn = self.mask_norm(attn)
            new_slots = torch.einsum('bjd, bij->bid', v, attn) / (attn.sum(-1, keepdim=True) + self.eps)
            multi_slots.append(new_slots)
        return new_slots

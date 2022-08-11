import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


class MultiProtoMetric(nn.Module):
    def __init__(self):
        super(MultiProtoMetric, self).__init__()
        self.proto_metric = ProtoMetric()

    def forward(self, f, elabel, glabel, bs, nw, ns, nq):
        _, emb_num, f_dim = f.shape
        f = f.reshape(bs, nw, ns + nq, emb_num, f_dim).permute(0, 3, 1, 2, 4)

        pred, _ = self.proto_metric(
            f.reshape(bs * emb_num, nw * (ns + nq), f_dim), elabel, glabel, bs * emb_num, nw, ns, nq
        )
        pred = pred.reshape(bs, emb_num, nw * (ns + nq), nw).mean(1)
        return pred, elabel


class ProtoMetric(nn.Module):
    def __init__(self):
        super(ProtoMetric, self).__init__()

    def forward(self, emb_all, elabel, glabel, bs, nw, ns, nq):
        if emb_all.dim() == 4:
            emb_all = func.adaptive_avg_pool2d(emb_all, output_size=(1, 1)).reshape(emb_all.shape[0], -1)
            emb_all = emb_all.reshape(bs, nw * (ns + nq), -1)
        bs, _, f_dim = emb_all.shape
        sup, _ = torch.split(emb_all.reshape(bs, nw, ns + nq, f_dim), dim=2, split_size_or_sections=[ns, nq])
        sup = sup.mean(2).reshape(bs, 1, nw, f_dim)
        emb_all = emb_all.reshape(bs, nw * (ns + nq), 1, f_dim)
        pred = - torch.sum((emb_all - sup) ** 2, dim=-1)
        return pred, elabel


class DN4Metric(nn.Module):
    def __init__(self, top_k=3):
        super(DN4Metric, self).__init__()
        self.top_k = top_k

    def forward(self, fm, elabel, glabel, bs, nw, ns, nq):
        _, f_dim, f_h, f_w = fm.shape
        # split
        fm = fm / (torch.norm(fm, p=2, dim=1, keepdim=True) + 1e-12)
        sup_fm, que_fm = torch.split(
            fm.reshape(bs, nw, ns + nq, f_dim, f_h, f_w), dim=2, split_size_or_sections=[ns, nq]
        )
        que_pred = torch.einsum(
            'b q x c, b w y c -> b q w xy', [
                que_fm.reshape(bs, nw * nq, f_dim, f_h * f_w).permute(0, 1, 3, 2),
                sup_fm.permute(0, 1, 2, 4, 5, 3).reshape(bs, nw, ns * f_h * f_w, f_dim)
            ]
        )
        pred = torch.topk(que_pred, self.top_k, dim=-1).values.sum([-2, -1])
        elabel = elabel.reshape(bs, nw, ns + nq)[:, :, ns:].reshape(-1)
        return pred, elabel


class TPNMetric(nn.Module):
    def __init__(self, fd, top_k=20):
        super(TPNMetric, self).__init__()

        class RelationNetwork(nn.Module):
            """Graph Construction Module"""

            def __init__(self, fd):
                super(RelationNetwork, self).__init__()

                self.layer1 = nn.Sequential(
                    nn.Linear(fd, fd),
                    nn.BatchNorm1d(fd),
                    nn.ReLU(),
                    nn.Linear(fd, 1),
                    nn.Sigmoid()
                )

            def forward(self, x):
                out = self.layer1(x)
                return out

        self.relation = RelationNetwork(fd)
        self.top_k = top_k
        self.alpha = nn.Parameter(torch.tensor([0.99]), requires_grad=False)

    def forward(self, emb_all, elabel, glabel, bs, nw, ns, nq):
        # init
        eps = np.finfo(float).eps
        bs, C, fh, fw = emb_all.shape

        # Step2: Graph Construction
        emb_all = func.adaptive_avg_pool2d(emb_all, output_size=(1, 1)).reshape(bs, C)
        sigma = self.relation(emb_all).reshape(bs, 1)
        N = bs
        bs = 1

        emb_all = emb_all / (sigma + eps)  # BxN*d
        emb_all = emb_all.reshape(1, N, -1)
        emb1 = torch.unsqueeze(emb_all, 2)  # BxN*1*d
        emb2 = torch.unsqueeze(emb_all, 1)  # Bx1*N*d
        W = ((emb1 - emb2) ** 2).mean(-1)  # BxN*N*d -> BxN*N
        W = torch.exp(-W / 2)

        ## keep top-k values
        if self.top_k > 0:
            topk, indices = torch.topk(W, self.top_k)  # BxNxN
            mask = torch.zeros_like(W)  # BxNxN
            mask = mask.scatter(2, indices, 1)
            mask = ((mask + mask.transpose(1, 2)) > 0).type(torch.float32)  # union, kNN graph
            W = W * mask  # BxNxN

        ## normalize
        D = W.sum(1)  # B x N
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))  # B x N
        D1 = torch.unsqueeze(D_sqrt_inv, 2).repeat(1, 1, N)  # B x N -> B x N x 1 -> B x N x N
        D2 = torch.unsqueeze(D_sqrt_inv, 1).repeat(1, N, 1)  # B x N -> B x 1 x N -> B x N x N
        S = D1 * W * D2  # B x N x N

        # Step3: Label Propagation, F = (I-\alpha S)^{-1}Y
        graph_elabel = elabel.reshape(bs, nw, ns + nq)[:, :, :ns]
        # print(elabel.shape)
        ys = func.one_hot(graph_elabel.reshape(bs, nw, ns))
        yu = torch.zeros(bs, nw, nq, nw).to(ys.device)
        y = torch.cat((ys, yu), 2).reshape(bs, nw * (ns + nq), nw)  # bs, nw * ns + nq, nw
        F = torch.matmul(torch.inverse(torch.eye(N).to(ys.device).unsqueeze(0) - self.alpha * S + eps), y)
        return F, elabel


def cosine_fully_connected_layer(x_in, weight, scale=None, bias=None, normalize_x=True, normalize_w=True):
    # x_in: a 2D tensor with shape [batch_size x num_features_in]
    # weight: a 2D tensor with shape [num_features_in x num_features_out]
    # scale: (optional) a scalar value
    # bias: (optional) a 1D tensor with shape [num_features_out]

    assert x_in.dim() == 2
    assert weight.dim() == 2
    assert x_in.size(1) == weight.size(0)

    if normalize_x:
        x_in = func.normalize(x_in, p=2, dim=1, eps=1e-12)

    if normalize_w:
        weight = func.normalize(weight, p=2, dim=0, eps=1e-12)

    x_out = torch.mm(x_in, weight)

    if scale is not None:
        x_out = x_out * scale.view(1, -1)

    if bias is not None:
        x_out = x_out + bias.view(1, -1)

    return x_out


class CosineClassifier(nn.Module):
    def __init__(
            self,
            num_channels,
            num_classes,
            scale=10.0,
            learn_scale=True,
            bias=False,
            normalize_x=True,
            normalize_w=True,
    ):
        super().__init__()

        self.num_channels = num_channels
        self.num_classes = num_classes
        self.normalize_x = normalize_x
        self.normalize_w = normalize_w

        weight = torch.FloatTensor(num_classes, num_channels).normal_(
            0.0, np.sqrt(2.0 / num_channels)
        )
        self.weight = nn.Parameter(weight, requires_grad=True)

        if bias:
            bias = torch.FloatTensor(num_classes).fill_(0.0)
            self.bias = nn.Parameter(bias, requires_grad=True)
        else:
            self.bias = None

        scale_cls = torch.FloatTensor(1).fill_(scale)
        self.scale_cls = nn.Parameter(scale_cls, requires_grad=learn_scale)

    def forward(self, x_in):
        assert x_in.dim() == 2
        return cosine_fully_connected_layer(
            x_in,
            self.weight.t(),
            scale=self.scale_cls,
            bias=self.bias,
            normalize_x=self.normalize_x,
            normalize_w=self.normalize_w,
        )


class RotClassifier(nn.Module):
    def __init__(self, dim, cls_num=4):
        super(RotClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.fc = CosineClassifier(dim, cls_num, 10.0, True)

    def forward(self, x):
        x = self.conv(x)
        x = func.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.shape[0], -1)
        x = self.fc(x)
        return x


class FxCosineClassifier(nn.Module):
    def __init__(self, dim, cls_num=200):
        super(FxCosineClassifier, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
            nn.Conv2d(dim, dim, 3, 2, 1),
            nn.BatchNorm2d(dim),
            nn.ReLU(),
        )
        self.fc = CosineClassifier(dim, cls_num, 10.0, True)
        self.scale = nn.Parameter(torch.tensor([10]), requires_grad=False)

    def forward(self, x, elabel, glabel, bs, nw, ns, nq):
        x = self.conv(x)
        x = func.adaptive_avg_pool2d(x, output_size=(1, 1)).view(x.shape[0], -1)
        if self.training:
            pred = self.fc(x)
            return pred, glabel
        else:
            fd = x.shape[-1]
            x = x.reshape(bs, nw, ns + nq, fd)
            weight = torch.mean(x[:, :, :ns], dim=2).reshape(nw, fd).t()
            pred = cosine_fully_connected_layer(x.reshape(nw * (ns + nq), fd), weight=weight, scale=self.scale)
            return pred, elabel

import torch
import torch.nn as nn
import torch.nn.functional as func


class PaCLModule(nn.Module):
    def __init__(self, dim, pacl_mode='cls_pro', simsiam=False):
        super(PaCLModule, self).__init__()
        self.pacl_mode = pacl_mode

        if simsiam or pacl_mode == 'ind':
            self.trans_mlp1 = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),  # first layer
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),  # second layer
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim, affine=False)
            )  # output layer

            # build a 2-layer predictor
            self.trans_mlp2 = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.ReLU(inplace=True),  # hidden layer
                nn.Linear(dim, dim)
            )  # output layer

        else:

            self.trans_mlp1 = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
                                            nn.Dropout(p=0.3))
            self.trans_mlp2 = nn.Sequential(nn.Linear(dim, dim, bias=False), nn.BatchNorm1d(dim), nn.ReLU(inplace=True),
                                            nn.Linear(dim, dim))

            # self.trans_mlp1 = nn.Identity()
            # self.trans_mlp2 = nn.Identity()

    def forward(self, prt_embeds, bs, nw, ns, nq):
        # pProRe Learning
        img_num, embed_num, f_dim = prt_embeds.shape

        if self.pacl_mode == 'cls_pro':
            prt_embeds = self.trans_mlp1(prt_embeds.reshape(-1, f_dim)).reshape(img_num, embed_num, f_dim)

            prt_embeds = prt_embeds.reshape(bs * nw, ns + nq, embed_num, f_dim)
            part_proto = prt_embeds.mean(1).detach()
            prt_embeds = self.trans_mlp2(prt_embeds.reshape(-1, f_dim)).reshape(img_num, embed_num, f_dim)
            pred = torch.einsum(
                'n x c, b c -> b n x',
                [
                    func.normalize(part_proto, dim=-1, p=2),
                    func.normalize(prt_embeds.reshape(img_num * embed_num, f_dim), dim=-1, p=2)
                ]
            ).mean(-2) * 10.

            label = torch.repeat_interleave(torch.arange(embed_num), repeats=img_num)
            label = label.reshape(embed_num, img_num).permute(1, 0).reshape(-1).to(prt_embeds.device)
            loss = func.cross_entropy(pred, label)
        elif self.pacl_mode == 'pro':
            prt_embeds = self.trans_mlp1(prt_embeds.reshape(-1, f_dim)).reshape(img_num, embed_num, f_dim)
            part_proto = prt_embeds.mean(0).detach()
            prt_embeds = self.trans_mlp2(prt_embeds.reshape(-1, f_dim))
            pred = torch.einsum(
                'n c, b c -> b n',
                [
                    func.normalize(part_proto, dim=-1, p=2),
                    func.normalize(prt_embeds, dim=-1, p=2)
                ]
            ) * 10.

            label = torch.repeat_interleave(torch.arange(embed_num), repeats=img_num)
            label = label.reshape(embed_num, img_num).permute(1, 0).reshape(-1).to(prt_embeds.device)
            loss = func.cross_entropy(pred, label)
        elif self.pacl_mode == 'ind':
            prt_embeds1 = self.trans_mlp1(prt_embeds.reshape(-1, f_dim)).reshape(img_num, embed_num, f_dim)
            prt_embeds2 = self.trans_mlp2(prt_embeds1.reshape(-1, f_dim))
            prt_embeds1 = prt_embeds1.detach()
            loss = -func.cosine_similarity(prt_embeds1.reshape(-1, f_dim), prt_embeds2.reshape(-1, f_dim)).mean()
        else:
            raise KeyError('No Valid Mode for PaCL.')

        return loss

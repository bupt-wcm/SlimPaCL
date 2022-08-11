import torch
import torch.nn as nn
import torch.nn.functional as func

from model.backbone import init_backbone
from model.modules.classifiers import MultiProtoMetric, DN4Metric, RotClassifier, ProtoMetric, TPNMetric, \
    FxCosineClassifier
from model.modules.idt import MetaPartModule
from model.modules.pacl import PaCLModule
from model.utils import create_4rotations_images
from model.utils import to_float, accuracy


class PaCLNet(nn.Module):
    def __init__(self, net_params):
        super(PaCLNet, self).__init__()
        self.backbone = init_backbone(net_params.backbone)
        fd = self.backbone.out_dim
        self.idt, self.pacl, self.rot, self.x_classifier = \
            net_params.idt, net_params.pacl, net_params.rot, net_params.x_classifier
        assert self.idt or self.x_classifier

        if self.idt:
            self.idt_module = MetaPartModule(fd, **net_params.idt_params)
            # few-shot classifier for the part features
            self.f_fs_classifier = MultiProtoMetric()

        # ssl-technology, [PaCL, Rot]
        if self.pacl:
            self.pacl_module = PaCLModule(dim=fd, pacl_mode=net_params.pacl_mode)
        self.pacl_loss_weight = net_params.pacl_loss_weight
        if self.rot:
            self.rot_classifier = RotClassifier(fd)
        self.rot_loss_weight = net_params.rot_loss_weight

        # few-shot classifier for the feature maps
        self.x_classifier_type = net_params.x_classifier_type.lower()
        if self.x_classifier:
            if net_params.x_classifier_type.lower() == 'dn4':
                self.x_fs_classifier = DN4Metric()
            elif net_params.x_classifier_type.lower() == 'tpn':
                self.x_fs_classifier = TPNMetric(fd)
            elif net_params.x_classifier_type.lower() == 'cc':
                self.x_fs_classifier = FxCosineClassifier(fd, cls_num=5)
            else:
                self.x_fs_classifier = ProtoMetric()

    def forward(self, batch_data):
        im, glabel, elabel, bs, nw, ns, nq = batch_data

        ret_info = {'loss': 0.0}

        rot_loss = 0.0
        img_num = im.shape[0]
        if self.rot:
            rot_ims, rot_labels = create_4rotations_images(im)
            fm = self.backbone(rot_ims)
            rot_pred = self.rot_classifier(fm)
            rot_loss = func.cross_entropy(rot_pred, rot_labels)
            fm = fm[:img_num]
            ret_info['rot_loss'] = to_float(rot_loss)
        else:
            fm = self.backbone(im)

        pacl_loss, f_fs_loss, x_fs_loss = 0.0, 0.0, 0.0
        if self.idt:
            all_embeds = self.idt_module(fm)
            if self.pacl:
                pacl_loss = self.pacl_module(all_embeds, bs, nw, ns, nq)
                ret_info['p_loss'] = to_float(pacl_loss)
            f_fs_pred, _ = self.f_fs_classifier(all_embeds, elabel, glabel, bs, nw, ns, nq)
            f_fs_loss = func.cross_entropy(f_fs_pred.reshape(-1, nw), elabel.reshape(-1))

            f_fs_pred = f_fs_pred.reshape(bs, nw, ns + nq, nw)[:, :, ns:]
            f_fs_label = elabel.reshape(bs, nw, ns + nq)[:, :, ns:]
            f_fs_acc = accuracy(f_fs_pred, f_fs_label, bs, nw, ns, nq)
            ret_info['f_closs'] = to_float(f_fs_loss)
            ret_info['f_acc.'] = to_float(f_fs_acc)

        if self.x_classifier:
            x_fs_pred, x_fs_label = self.x_fs_classifier(fm, elabel, glabel, bs, nw, ns, nq)
            if 'dn4' != self.x_classifier_type:
                x_fs_pred = x_fs_pred.reshape(bs, nw, ns + nq, -1)[:, :, ns:]
                x_fs_label = x_fs_label.reshape(bs, nw, ns + nq)[:, :, ns:]
            if 'cc' == self.x_classifier_type:
                if self.training:
                    x_fs_loss = func.cross_entropy(x_fs_pred.reshape(bs * nw * nq, -1), x_fs_label.reshape(-1))
                    x_fs_acc = torch.argmax(
                        x_fs_pred.reshape(bs * nw * nq, -1), dim=-1
                    ).view(-1).eq(x_fs_label.reshape(-1)).sum() / (bs * nw * nq) * 100.
                else:
                    x_fs_loss = func.cross_entropy(x_fs_pred.reshape(bs * nw * nq, -1), x_fs_label.reshape(-1))
                    x_fs_acc = accuracy(x_fs_pred, x_fs_label, bs, nw, ns, nq)
            else:
                x_fs_loss = func.cross_entropy(x_fs_pred.reshape(bs * nw * nq, -1), x_fs_label.reshape(-1))
                x_fs_acc = accuracy(x_fs_pred, x_fs_label, bs, nw, ns, nq)

            ret_info['x_closs'] = to_float(x_fs_loss)
            ret_info['x_acc.'] = to_float(x_fs_acc)

        # print(rot_loss, self.rot_loss_weight, f_fs_loss, self.pacl_loss_weight[0], pacl_loss, self.pacl_loss_weight[1], x_fs_loss)

        loss = rot_loss * self.rot_loss_weight + \
               (f_fs_loss * self.pacl_loss_weight[0] + pacl_loss * self.pacl_loss_weight[1]) + \
               x_fs_loss

        ret_info['loss'] = to_float(loss)
        return loss, ret_info

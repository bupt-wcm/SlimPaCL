from .cov_net import CovNet
from .res_net12 import resnet12


def init_backbone(backbone_params):
    name = backbone_params.name
    if name == 'cov_net':
        backbone = CovNet(
            **backbone_params.params
        )
    elif name == 'res_net12':
        backbone = resnet12(
            **backbone_params.params
        )
    else:
        raise KeyError('Type %s backbone is not supported.' % name)
    return backbone

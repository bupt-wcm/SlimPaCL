import json
import pprint
import random
import warnings
from pprint import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

from data import init_loader
from libs.init_exp import init_exp
from model.pacl_model import PaCLModel

warnings.filterwarnings("ignore")

DATA_INFO = {
    'cub': './data/source_data/pickle_data/cub_cls_info-100-50-50.pkl',
    'dog': './data/source_data/pickle_data/dog_cls_info-70-20-30.pkl',
    'car': './data/source_data/pickle_data/car_cls_info-130-17-49.pkl'
}


def main(hparams):
    # init experiment information and Logger
    exp_info = init_exp(
        data_name=hparams.data.train.name,
        bs=hparams.data.train.bs,
        nw=hparams.data.train.nw,
        ns=hparams.data.train.ns,
        nq=hparams.data.train.nq,
        gpu=hparams.environ.gpu,
        prefix=hparams.environ.save_prefix,
        prefix_describe=hparams.environ.prefix,
        postfix_describe=hparams.environ.postfix,
    )
    with open(exp_info['saved_path'] + '/train_cfg.yml', 'w') as f:
        yaml.dump(json.loads(json.dumps(hparams)), f)

    pprint(exp_info)

    # init train/valid data and dataloaders
    dataloader_params = {
        'pin_memory': False,
        'num_workers': 16,
        'persistent_workers': True,
    }

    if hparams.environ.deterministic:
        # torch.use_deterministic_algorithms(True)
        np.random.seed(hparams.environ.seed)
        random.seed(hparams.environ.seed)
        torch.manual_seed(hparams.environ.seed)

        # for efficient
        torch.backends.cudnn.benchmark = True

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(hparams.environ.seed)

        dataloader_params['worker_init_fn'] = seed_worker
        dataloader_params['generator'] = g

    # ----------------- Create Data Loaders --------------------------------------------------
    nw, ns, nq, episode, bs = \
        hparams.data.train.nw, hparams.data.train.ns, hparams.data.train.nq, hparams.data.train.episode, hparams.data.train.bs
    train_loader = init_loader(
        DATA_INFO[hparams.data.train.name], 'train', nw, ns, nq, episode, bs, dataloader_params
    )

    nw, ns, nq, episode, bs = \
        hparams.data.valid.nw, hparams.data.valid.ns, hparams.data.valid.nq, hparams.data.valid.episode, hparams.data.valid.bs
    valid_loader = init_loader(
        DATA_INFO[hparams.data.valid.name], 'valid', nw, ns, nq, episode, bs, dataloader_params
    )

    nw, ns, nq, episode, bs = \
        hparams.data.test.nw, hparams.data.test.ns, hparams.data.test.nq, hparams.data.test.episode, hparams.data.test.bs
    test_loader = init_loader(
        DATA_INFO[hparams.data.test.name], 'test', nw, ns, nq, episode, bs, dataloader_params
    )
    # -----------------------------------------------------------------------------------------

    # Create Model
    model = PaCLModel(
        hparams=hparams, exp_info=exp_info,
        data_loaders={
            'train': train_loader, 'valid': valid_loader, 'test': test_loader,
        }
    )

    # Start Training
    model.train_loop()


def run_main(args):
    with open(args.config, 'r') as f:
        hparams_from_file = edict(yaml.load(f, Loader=yaml.FullLoader))
    main(hparams_from_file)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('PaCL')
    parser.add_argument('--config', type=str, metavar='PATH')
    args = parser.parse_args()
    run_main(args)

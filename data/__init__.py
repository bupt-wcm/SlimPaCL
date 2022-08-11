import pickle
import random

from torch.utils.data import DataLoader

from data.collect_fn import *
from data.fs_dataset import *
from data.fs_sampler import EpisodicBatchSampler
from data.transforms import FSTransformer


def init_loader(path, phase, nw, ns, nq, episode, batch_scale, dataloader_params):
    with open(path, 'rb') as f:
        data_cls_info = pickle.load(f)

    data_set = EpisodicDataset(
        file_list=data_cls_info[phase]['file'],
        transformer=FSTransformer(train=True if phase == 'train' else False)
    )

    data_sampler = EpisodicBatchSampler(
        data_cls_info[phase]['cls_meta'],
        nw=nw, ns=ns, nq=nq, episode=episode, batch_scale=batch_scale,
    )
    data_loader = DataLoader(
        data_set,
        batch_sampler=data_sampler,
        collate_fn=FastCollate(nw, ns, nq, batch_scale),
        **dataloader_params
    )
    data_loader = DataFetcher(data_loader, batch_scale, nw, ns, nq, True)
    del data_cls_info
    return data_loader


def generate_exp_loaders(hparams):
    dataloader_params = {
        'pin_memory': False,
        'num_workers': 16,
        'persistent_workers': True,
    }
    g = torch.Generator()
    g.manual_seed(hparams.environ.seed)

    data_info = hparams.data.info

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    dataloader_params['worker_init_fn'] = seed_worker
    dataloader_params['generator'] = g

    # ----------------- Create Data Loaders --------------------------------------------------
    nw, ns, nq, episode, bs = \
        hparams.data.train.nw, hparams.data.train.ns, hparams.data.train.nq, hparams.data.train.episode, hparams.data.train.bs
    train_loader = init_loader(
        data_info[hparams.data.train.name], 'train', nw, ns, nq, episode, bs, dataloader_params
    )

    nw, ns, nq, episode, bs = \
        hparams.data.valid.nw, hparams.data.valid.ns, hparams.data.valid.nq, hparams.data.valid.episode, hparams.data.valid.bs
    valid_loader = init_loader(
        data_info[hparams.data.valid.name], 'valid', nw, ns, nq, episode, bs, dataloader_params
    )

    nw, ns, nq, episode, bs = \
        hparams.data.test.nw, hparams.data.test.ns, hparams.data.test.nq, hparams.data.test.episode, hparams.data.test.bs
    test_loader = init_loader(
        data_info[hparams.data.test.name], 'test', nw, ns, nq, episode, bs, dataloader_params
    )

    return {
        'train': train_loader, 'valid': valid_loader, 'test': test_loader
    }

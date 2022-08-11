import os
import re

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from data import init_loader
from model.pacl_net import PaCLNet

DATA_INFO = {
    'cub': './data/source_data/pickle_data/cub_cls_info-100-50-50.pkl',
    'dog': './data/source_data/pickle_data/dog_cls_info-70-20-30.pkl',
    'car': './data/source_data/pickle_data/car_cls_info-130-17-49.pkl'
}

dataloader_params = {
    'pin_memory': False,
    'num_workers': 8,
    'persistent_workers': True,
}


# override this function since it may face the import problem
def load_file_to_dict(hparams, file_path, args, relative=False):
    with open(file_path, 'r') as f:
        hparams_from_file = edict(yaml.load(f, Loader=yaml.FullLoader))
    if '_base' in hparams_from_file:

        if args is not None and '_base' in args:
            hparams_from_file['_base'] = args['_base']  # different from the source code
            print('Replace _base in config file with _based in args.')
        if relative:
            hparams_from_file['_base'] = '.' + hparams_from_file['_base']
        print('Load %s into the hparams' % hparams_from_file['_base'])
        hparams = load_file_to_dict(hparams, hparams_from_file['_base'], None)  # only for the first level
        del hparams_from_file['_base']
    hparams = combine_hparams(hparams, hparams_from_file)
    return hparams


def test(shot_ckp, episode):
    args = {'exp.config': os.path.join(shot_ckp, 'train_cfg.yml')}
    hparams = reset_hparams()
    hparams_from_args = edict(args)
    if 'exp.config' in args:
        hparams = load_file_to_dict(hparams, args['exp.config'], args)
        del hparams_from_args['exp.config']
    hparams = load_args_to_dict(hparams, hparams_from_args)

    data_name, shot = hparams.data.test.name, hparams.data.test.ns

    data_loader = init_loader(
        DATA_INFO[data_name], 'test', 5, shot, 15, episode, 1, dataloader_params
    )
    data_iter = iter(data_loader)

    best_mean = 0.0
    print(shot_ckp)
    ckp_files = os.path.join(shot_ckp, 'BestModel')

    for file_name in os.listdir(ckp_files):
        mean = re.search("\d+\.\d+\|", file_name)
        mean = float(mean.group()[:-1])
        if mean > best_mean:
            best_mean = mean
            ckpt = file_name

    network = PaCLNet(hparams.model.network).cuda()
    print('Start Loading.')
    network.load_state_dict(
        torch.load(os.path.join(ckp_files, ckpt))['model_state_dict']
    )
    network.eval()
    with torch.no_grad():
        valid_outputs = []
        for idx, batch_data in tqdm(enumerate(data_iter)):
            _, valid_output = network(batch_data)
            valid_outputs.append(valid_output)

        loss_info, acc_info = '', ''
        for key_name in valid_outputs[0].keys():
            if 'loss' in key_name.lower():
                cl_losses = np.array([item[key_name] for item in valid_outputs]).mean()
                loss_info += '{name:<6s}:{value:>6.2f}|'.format(name=key_name, value=cl_losses)
            if 'acc' in key_name.lower():
                accuracies = np.array([item[key_name] for item in valid_outputs])
                mean = np.mean(accuracies)
                std = 1.96 * np.std(accuracies) / np.sqrt(len(valid_outputs))
                acc_info += '{:s} {:05.2f}|{:04.2f}|'.format(key_name, mean, std)
        print(acc_info)


if __name__ == '__main__':
    import fire

    fire.Fire(test)

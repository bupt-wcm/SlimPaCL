import numpy as np
import torch
import os
import torch.nn as nn
import logging

from libs.checkpoint import ModelCheckpoint
from libs.count_params import count_params
from libs.lr_scheduler import WarmMultiStonesScheduler
from model.pacl_net import PaCLNet


def weights_init_relu(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        n = m.weight.size(1)
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


class PaCLModel:
    def __init__(self, exp_info, hparams, data_loaders):
        self.exp_info = exp_info
        self.hparams = hparams

        self.data_loaders = data_loaders

        self._create_net(hparams)
        self.optimizer, self.scheduler = self._init_optimizer(hparams)
        self.checkpoint = ModelCheckpoint(
            self,
            self.exp_info,
            model_save_freq=hparams.model.io.model_save_freq
        )

        # intra log values
        self._best_acc = 0.0
        self._current_epoch = 0
        self._train_batch_num, self._valid_batch_num = 0, 0

        self.epochs = hparams.model.learning.epochs
        self.train_episode = hparams.data.train.episode
        self.valid_episode = hparams.data.valid.episode

        # model save frequent
        self.save_freq, self.print_freq = hparams.model.io.metric_save_freq, hparams.model.io.metric_info_freq
        self.model_eval_freq = hparams.model.io.model_eval_freq

        # statics of training
        self.train_metrics = {}
        self.train_count = 0

        self.logger = self._init_logger(exp_info)

    def _init_logger(self, exp_info):
        logger = logging.getLogger(exp_info['exp_name'] + exp_info['train_name'])
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("%(asctime)s %(message)s", "%Y%b%d-%H:%M:%S")

        # init handles
        fh = logging.FileHandler(os.path.join(exp_info['saved_path'], 'Train.log'), mode='w')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        # save the configures
        return logger

    def _create_net(self, hparams, *args, **kwargs):
        net_hparams = hparams.model.network
        net = PaCLNet(net_hparams)
        net.apply(weights_init_relu)
        param_info = count_params(net, depth=2)
        print(param_info)
        net = net.cuda()
        self.net = net
        return net

    def _obtain_parameters(self):
        return self.net.parameters()

    def _init_optimizer(self, hparams, *args, **kwargs):
        optim_params = hparams.model.learning.optimizer
        scheduler_params = hparams.model.learning.scheduler

        optimizer = torch.optim.Adam(
            params=self._obtain_parameters(),
            **optim_params.params
        )
        scheduler = WarmMultiStonesScheduler(
            optimizer=optimizer,
            **scheduler_params.params
        )
        return optimizer, scheduler

    def train_step(self, batch_data, batch_idx):
        self.optimizer.zero_grad()
        loss, ret_info = self.net(batch_data)
        loss.backward()
        self.optimizer.step()
        return ret_info

    def valid_step(self, batch_data, batch_idx):
        with torch.no_grad():
            loss, ret_info = self.net(batch_data)
        return ret_info

    def train_epoch(self, epoch):
        self._current_epoch = epoch
        train_outputs_collect = []
        self.net.train()
        for idx, batch_data in enumerate(self.data_loaders['train']):
            train_output = self.train_step(batch_data, idx)
            self.train_step_end(batch_data, train_output, idx)
            self._train_batch_num += 1
            train_outputs_collect.append(train_output)
        return train_outputs_collect

    def valid_epoch(self, loader):
        self.net.eval()
        train_outputs_collect = []
        for idx, batch_data in enumerate(loader):
            valid_output = self.valid_step(batch_data, idx)
            self._valid_batch_num += 1
            train_outputs_collect.append(valid_output)
        return train_outputs_collect

    def train_loop(self):
        self.logger.info('-' * 20 + 'Start Training' + '-' * 20)
        model_state_dict = None
        for epoch in range(self.epochs):
            self.train_epoch(epoch)

            is_save, model_save_metrics = None, None
            if (epoch + 1) % self.model_eval_freq == 0:
                valid_outputs = self.valid_epoch(self.data_loaders['valid'])
                is_save, model_save_metrics = self.valid_epoch_end(valid_outputs)
            # save model
            self.checkpoint.save_model(is_save, epoch, model_save_metrics)
            if is_save:
                model_state_dict = self._state_dict()
        self._load_state_dict(model_state_dict)
        valid_outputs = self.valid_epoch(self.data_loaders['test'])
        self.valid_epoch_end(valid_outputs)

    def _state_dict(self):
        return self.net.state_dict()

    def _load_state_dict(self, state_dict, *args, **kwargs):
        return self.net.load_state_dict(state_dict)

    def train_step_end(self, batch_data, train_output, idx) -> None:
        if (idx + 1) % self.save_freq == 0:
            for name in train_output:
                if name not in self.train_metrics:
                    self.train_metrics[name] = []
                self.train_metrics[name].append(train_output[name])

        if (idx + 1) % self.print_freq == 0:
            prefix = '{epoch:03d}|{epochs:03d}  {c_episode:04d}|{episode:04d} |Train| '.format(
                epoch=self._current_epoch, epochs=self.epochs, c_episode=idx,
                episode=self.train_episode
            )
            current_lr = self.scheduler.get_learning_rate(self._train_batch_num)
            if isinstance(current_lr, list):
                current_lr = current_lr[0]
            info_strs = []
            for name in self.train_metrics.keys():
                # print(name, self.scalar_store[name], range_num)
                num_items = len(self.train_metrics[name])
                info_value = sum([value for value in self.train_metrics[name]]) / num_items
                info_strs.append('{name:<6s}:{value:>6.2f}'.format(name=name, value=info_value))
                self.train_metrics[name] = []
            info_str = '|'.join(info_strs)

            self.logger.info(
                prefix + info_str + ' |LR:{:.4f}'.format(current_lr)
            )
            self.train_count = 0

        # adjust lr
        if self.scheduler is not None:
            self.scheduler.step(self._train_batch_num)

    def valid_epoch_end(self, valid_outputs):
        info_str = '{epoch:03d}|{epochs:03d} \t {episode:04d} |Valid| '.format(
            epoch=self._current_epoch, epochs=self.epochs,
            episode=self.valid_episode
        )

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
        info_str = info_str + loss_info + acc_info

        is_best = mean > self._best_acc
        if is_best:
            self._best_acc = mean
        info_str += '|Best Acc. {:06.3f}%'.format(self._best_acc)

        self.logger.info(info_str)
        return is_best, {
            'im_mean': mean,
            'im_std': std
        }

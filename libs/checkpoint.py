import os

import torch


class ModelCheckpoint:
    def __init__(self, model, exp_info, model_save_freq=30, regular_save=True):
        super(ModelCheckpoint, self).__init__()
        self.exp_info = exp_info
        self.model = model
        self.regular_save = regular_save
        self.regular_save_freq = model_save_freq

        os.mkdir(os.path.join(self.exp_info['saved_path'], 'BestModel'))
        if self.regular_save:
            os.mkdir(os.path.join(self.exp_info['saved_path'], 'RegularSave'))

    def save_model(self, is_best, current_epoch, metric):
        if self.regular_save:
            if (current_epoch + 1) % self.regular_save_freq == 0:
                regular_saved_name = "{ExpName}-{Epoch:03d}.pth.tar".format(
                    ExpName=self.exp_info['exp_name'], Epoch=self.model._current_epoch)
                regular_saved_dict = {
                    'model_state_dict': self.model._state_dict(),
                    'optimizer': self.model.optimizer.state_dict(),
                }
                torch.save(regular_saved_dict,
                           f=os.path.join(
                               os.path.join(self.exp_info['saved_path'], 'RegularSave'),
                               regular_saved_name
                           ))

        if is_best is not None and is_best:
            saved_name = "{ExpName}-{Epoch:03d}-{Acc:4.2f}|{Std:4.2f}.pth.tar".format(
                ExpName=self.exp_info['exp_name'], Epoch=self.model._current_epoch,
                Acc=metric['im_mean'] * 100, Std=metric['im_std'] * 100)
            saved_dict = {
                'model_state_dict': self.model._state_dict(),
                'optimizer': self.model.optimizer.state_dict(),
                'accuracy': metric['im_mean'],
                'std': metric['im_std'],
            }
            torch.save(saved_dict,
                       f=os.path.join(
                           os.path.join(self.exp_info['saved_path'], 'BestModel'),
                           saved_name
                       ))

from torch.utils.data import Dataset

from .transforms import FSTransformer


class EpisodicDataset(Dataset):
    def __init__(self, file_list, transformer: FSTransformer):
        super(EpisodicDataset, self).__init__()
        self.file_list = file_list
        self.transformer = transformer

    def __len__(self):
        return len(self.file_list)  # the len function is not used for the episodic sampler

    def __getitem__(self, index):
        im, label = self.file_list[index]
        im = self.transformer(im)
        return im, label

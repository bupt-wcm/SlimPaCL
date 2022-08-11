import numpy as np


class EpisodicBatchSampler:
    def __init__(self, cls_meta: dict, nw: int = 5, ns: int = 5, nq: int = 15, episode: int = 100,
                 batch_scale: int = 1):
        self.cls_meta = cls_meta

        self.nw = nw
        self.ns = ns
        self.nq = nq

        self.episode = episode

        self.batch_scale = batch_scale
        self.episode = self.episode + self.batch_scale - (self.episode % self.batch_scale)
        self.n_classes = len(list(cls_meta.keys()))

    def __len__(self):
        return self.episode

    def __iter__(self):
        cls_index = np.arange(self.n_classes)
        for i in range(self.episode // self.batch_scale):
            selected_indexes = []
            for j in range(self.batch_scale):
                np.random.shuffle(cls_index)
                selected_classes = cls_index[:self.nw]

                for cls in selected_classes:  # sorted is must.
                    np.random.shuffle(self.cls_meta[cls])  # very faster than permutation
                    selected_indexes.extend(self.cls_meta[cls][:self.ns + self.nq])
            yield selected_indexes

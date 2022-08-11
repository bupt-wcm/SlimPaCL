import numpy as np
import torch


class FastCollate:
    def __init__(self, nw, ns, nq, bs):
        self.nw, self.ns, self.nq = nw, ns, nq
        self.bs = bs

    def __call__(self, batch):
        imgs = [img[0] for img in batch]
        targets = torch.tensor([target[1] for target in batch], dtype=torch.int64)

        w = imgs[0].size[0]
        h = imgs[0].size[1]
        tensor = torch.zeros((len(imgs), 3, h, w), dtype=torch.float).contiguous()
        for i, img in enumerate(imgs):
            numpy_array = np.asarray(img, dtype=np.uint8)

            if numpy_array.ndim < 3:
                numpy_array = np.expand_dims(numpy_array, axis=-1)

            numpy_array = np.rollaxis(numpy_array, 2)
            tensor[i] += torch.from_numpy(numpy_array.copy())

        episodic_targets = torch.from_numpy(
            np.repeat(np.repeat(range(self.nw), self.ns + self.nq), self.bs)
        ).reshape(-1, self.bs).permute(1, 0).reshape(-1)
        # print(targets, episodic_targets)
        return tensor, targets, episodic_targets


class DataFetcher:
    def __init__(self, torch_loader, bs, nw, ns, nq, norm=True):
        self.torch_loader = torch_loader
        self.norm = norm
        self.bs, self.nw, self.ns, self.nq = bs, nw, ns, nq

        self.tensor_mean = torch.tensor([255., 255., 255.]).view(1, 3, 1, 1).cuda()
        self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).view(1, 3, 1, 1).cuda()
        self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).view(1, 3, 1, 1).cuda()

    def __len__(self):
        return len(self.torch_loader)

    def __iter__(self):
        self.stream = torch.cuda.Stream()
        self.loader = iter(self.torch_loader)
        self.preload()
        return self

    def preload(self):
        try:
            self.next_input, self.next_label, self.episodic_label = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_label = None
            self.episodic_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)
            self.episodic_label = self.episodic_label.cuda(non_blocking=True)

            self.next_input = self.next_input.float()
            if self.norm:
                self.next_input = self.next_input.sub(self.mean).div(self.std)
            else:
                self.next_input = self.next_input.div(self.tensor_mean)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        label = self.next_label
        episodic_label = self.episodic_label
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
            label.record_stream(torch.cuda.current_stream())
            episodic_label.record_stream(torch.cuda.current_stream())
        else:
            raise StopIteration
        self.preload()
        return input, label, episodic_label, self.bs, self.nw, self.ns, self.nq

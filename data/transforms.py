import torchvision


class FSTransformer:
    def __init__(self, train=True, pil_to_tensor=False):
        if train:
            trans = [
                torchvision.transforms.Resize(96),
                torchvision.transforms.RandomCrop(84),
                # torchvision.transforms.RandomRotation(15),
                torchvision.transforms.ColorJitter(0.1, 0.1, 0.1),
                torchvision.transforms.RandomHorizontalFlip(),
                # torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        else:
            trans = [
                torchvision.transforms.Resize(96),
                torchvision.transforms.CenterCrop(84),
                # torchvision.transforms.ToTensor(),
                # torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ]
        if pil_to_tensor:
            trans.extend(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                ]
            )
        self.trans = torchvision.transforms.Compose(trans)

    def __call__(self, im):
        return self.trans(im)

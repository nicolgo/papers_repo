from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CustomCIFAR10(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self):
        self.data_dir = "dataset/CIFAR10/data"
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        self.dataset = CIFAR10(self.data_dir, train=self.train, download=True)
        self.data = self.dataset.data
        self.target = self.dataset.targets

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = dict()
        example["image"] = (self.transform(self.data[idx])).permute(1, 2, 0)
        example["class_label"] = self.target[idx]
        return example


class CIFAR10Train(CustomCIFAR10):
    def __init__(self):
        self.train = True
        super().__init__()


class CIFAR10Validation(CustomCIFAR10):
    def __init__(self):
        self.train = False
        super().__init__()

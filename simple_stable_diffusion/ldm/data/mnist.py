from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import MNIST
import taming.data.utils as tdu
import os


class MNISTBase(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.base = self.get_base()

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        example = self.base[i]
        example["image"] = 1
        return example

    def _prepare(self):
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

        self.data_dir = os.path.join(self.root, "data")
        if not tdu.is_prepared(self.root):
            MNIST(self.data_dir, train=True, download=True)
            MNIST(self.data_dir, train=False, download=True)
            tdu.mark_prepared(self.root)


class MNISTTrain(MNISTBase):
    def __init__(self):
        pass

    def get_base(self):
        return MNIST(self.data_dir, train=False, transform=self.transform)


class MNISTValidation(MNISTBase):
    def __init__(self):
        pass

    def get_base(self):
        return MNIST(self.data_dir, train=False, transform=self.transform)

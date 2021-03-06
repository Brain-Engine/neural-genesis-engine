from torchvision import datasets
from torchvision.transforms import transforms
from core.datasets.transforms.custom_transform import *


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


def train_dataset(data_dir, transform=MNISTTrainTransform, **kwargs):
    return datasets.FashionMNIST(root=data_dir, train=True, transform=transform, download=True, **kwargs)


# val dataset example for tiny-image-net
def val_dataset(data_dir, transform=MNISTValidationTransform, **kwargs):
    return datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True, **kwargs)


# test dataset example for tiny-image-net
def test_dataset(data_dir, transform=MNISTTestTransform, **kwargs):
    return datasets.FashionMNIST(root=data_dir, train=False, transform=transform, download=True, **kwargs)

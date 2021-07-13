from core.datasets.data_zoo import get_data_by_name
from torchvision import datasets

if __name__ == '__main__':
    get_data_by_name("mnist", data_dir="./data/dataset")
    get_data_by_name("cifar10", data_dir="./data/dataset")
    get_data_by_name("fashion_mnist", data_dir="./data/dataset")

    datasets.MNIST(root="./data/dataset", download=True)
    datasets.CIFAR10(root="./data/dataset", download=True)
    datasets.FashionMNIST(root="./data/dataset", download=True)

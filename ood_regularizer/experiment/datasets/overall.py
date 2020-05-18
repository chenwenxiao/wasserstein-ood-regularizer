from ood_regularizer.experiment.datasets.celeba import load_celeba
from ood_regularizer.experiment.datasets.imagenet import load_imagenet_test
from ood_regularizer.experiment.datasets.isun import load_isun_test
from ood_regularizer.experiment.datasets.kmnist import load_kmnist
from ood_regularizer.experiment.datasets.lsun import load_lsun_test
from ood_regularizer.experiment.datasets.not_mnist import load_not_mnist
from tfsnippet.datasets import load_cifar10, load_cifar100, load_fashion_mnist, load_mnist

from ood_regularizer.experiment.datasets.omniglot import load_omniglot
from ood_regularizer.experiment.datasets.svhn import load_svhn


def load_dataset(dataset_name):
    x_train = None
    x_test = None
    if dataset_name == 'celeba':
        x_train, x_validate, x_test = load_celeba()
    elif dataset_name == 'imagenet':
        x_test, y_test = load_imagenet_test()
    elif dataset_name == 'isun':
        x_test, y_test = load_isun_test()
    elif dataset_name == 'kmnist':
        (_x_train, _y_train), (_x_test, _y_test) = load_kmnist()
    elif dataset_name == 'lsun':
        x_test, y_test = load_lsun_test()
    elif dataset_name == 'not_mnist':
        (_x_train, _y_train), (_x_test, _y_test) = load_not_mnist()
    elif dataset_name == 'omniglot':
        (_x_train, _y_train), (_x_test, _y_test) = load_omniglot()
    elif dataset_name == 'svhn':
        (_x_train, _y_train), (_x_test, _y_test) = load_svhn()
    elif dataset_name == 'cifar10':
        (_x_train, _y_train), (_x_test, _y_test) = load_cifar10()
    elif dataset_name == 'cifar100':
        (_x_train, _y_train), (_x_test, _y_test) = load_cifar100()
    elif dataset_name == 'fashion_mnist':
        (_x_train, _y_train), (_x_test, _y_test) = load_fashion_mnist()
    elif dataset_name == 'mnist':
        (_x_train, _y_train), (_x_test, _y_test) = load_mnist()
    else:
        raise RuntimeError('dataset {} is not supported'.format(dataset_name))
    return x_train, x_test

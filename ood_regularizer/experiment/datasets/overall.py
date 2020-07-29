from ood_regularizer.experiment.datasets.celeba import load_celeba
from ood_regularizer.experiment.datasets.constant import load_constant
from ood_regularizer.experiment.datasets.imagenet import load_imagenet
from ood_regularizer.experiment.datasets.isun import load_isun_test
from ood_regularizer.experiment.datasets.kmnist import load_kmnist
from ood_regularizer.experiment.datasets.lsun import load_lsun_test
from ood_regularizer.experiment.datasets.noise import load_noise
from ood_regularizer.experiment.datasets.not_mnist import load_not_mnist
from tfsnippet.datasets import load_cifar10, load_cifar100, load_fashion_mnist, load_mnist

from ood_regularizer.experiment.datasets.omniglot import load_omniglot
from ood_regularizer.experiment.datasets.sun import load_sun
from ood_regularizer.experiment.datasets.svhn import load_svhn
from ood_regularizer.experiment.datasets.tinyimagenet import load_tinyimagenet

import numpy as np
import os

__all_datasets__ = ['mnist', 'fashion_mnist', 'kmnist', 'not_mnist', 'omniglot', 'celeba', 'tinyimagenet', 'svhn',
                    'cifar10', 'cifar100', 'isun', 'lsun', 'constant', 'noise']


def load_overall(dataset_name, dtype=np.uint8):
    x_train = None
    x_test = None
    y_train = None
    y_test = None
    if dataset_name == 'celeba':
        x_train, x_validate, x_test = load_celeba(img_size=32)
    elif dataset_name == 'tinyimagenet':
        (x_train, y_train), (x_test, y_test) = load_tinyimagenet()
    elif dataset_name == 'svhn':
        (x_train, y_train), (x_test, y_test) = load_svhn()
    elif dataset_name == 'cifar10':
        (x_train, y_train), (x_test, y_test) = load_cifar10()
    elif dataset_name == 'cifar100':
        (x_train, y_train), (x_test, y_test) = load_cifar100()
    elif dataset_name == 'imagenet':
        (x_train, y_train), (x_test, y_test) = load_imagenet()
    elif dataset_name == 'isun':
        x_test, y_test = load_isun_test()
    elif dataset_name == 'sun':
        (x_train, y_train), (x_test, y_test) = load_sun()
    elif dataset_name == 'lsun':
        x_test, y_test = load_lsun_test()
    elif dataset_name == 'kmnist':
        (x_train, y_train), (x_test, y_test) = load_kmnist()
    elif dataset_name == 'not_mnist':
        (x_train, y_train), (x_test, y_test) = load_not_mnist()
    elif dataset_name == 'omniglot':
        (x_train, y_train), (x_test, y_test) = load_omniglot()
    elif dataset_name == 'fashion_mnist':
        (x_train, y_train), (x_test, y_test) = load_fashion_mnist(x_shape=(28, 28, 1))
    elif dataset_name == 'mnist':
        (x_train, y_train), (x_test, y_test) = load_mnist(x_shape=(28, 28, 1))
    elif dataset_name == 'constant':
        (x_train, y_train), (x_test, y_test) = load_constant()
    elif dataset_name == 'noise':
        (x_train, y_train), (x_test, y_test) = load_noise()
    else:
        find_others = False
        for i in range(len(__all_datasets__)):
            if dataset_name == __all_datasets__[i] + '_others':
                find_others = True
        x_test = np.load('/home/cwx17/new_data/' + dataset_name + '/' + 'test.npy')
        if not find_others:
            raise RuntimeError('dataset {} is not supported'.format(dataset_name))

    if dataset_name in ['mnist', 'fashion_mnist', 'kmnist', 'not_mnist', 'omniglot']:
        x_train = np.pad(x_train, ((0, 0), (2, 2), (2, 2), (0, 0)))
        x_test = np.pad(x_test, ((0, 0), (2, 2), (2, 2), (0, 0)))
        x_train = np.tile(x_train, (1, 1, 1, 3))
        x_test = np.tile(x_test, (1, 1, 1, 3))

    if x_train is None:
        x_train = x_test
    x_train = x_train.astype(dtype)
    x_test = x_test.astype(dtype)
    if y_train is None:
        y_train = np.random.randint(0, 10, len(x_train))
    if y_test is None:
        y_test = np.random.randint(0, 10, len(x_test))
    print(x_train.shape, y_train.shape)
    print(x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def load_complexity(dataset_name, compressor):
    train_complexity_path = '/home/cwx17/new_data/' + dataset_name + '/' + 'train_complexity.npy'
    test_complexity_path = '/home/cwx17/new_data/' + dataset_name + '/' + 'test_complexity.npy'
    x_train_complexity = None
    x_test_complexity = None
    if os.path.exists(train_complexity_path):
        x_train_complexity = np.load(train_complexity_path)
    if os.path.exists(test_complexity_path):
        x_test_complexity = np.load(test_complexity_path)
    if x_train_complexity is None:
        x_train_complexity = x_test_complexity
    print(x_train_complexity.shape, x_test_complexity.shape)
    # if dataset_name in ['celeba', 'tinyimagenet', 'svhn', 'cifar10', 'cifar100', 'imagenet', 'isun', 'sun', 'lsun']:
    x_train_complexity = x_train_complexity / (32 * 32 * 3 * np.log(2))
    x_test_complexity = x_test_complexity / (32 * 32 * 3 * np.log(2))
    # else:
    #     x_train_complexity = x_train_complexity / (28 * 28 * 1 * np.log(2))
    #     x_test_complexity = x_test_complexity / (28 * 28 * 1 * np.log(2))

    return x_train_complexity[..., compressor], x_test_complexity[..., compressor]

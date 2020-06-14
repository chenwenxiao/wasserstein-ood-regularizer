from functools import partial
from typing import *

import mltk
import numpy as np

from ...misc import get_bit_depth
from ..types import *
from .base import *

__all__ = ['Cifar10', 'Cifar100']


class BaseCifar(ArrayDataSet):

    name: str

    def __init__(self, val_split: Optional[float] = None):
        if self.name == 'cifar10':
            loader = partial(mltk.data.load_cifar10, x_dtype=np.uint8)
            n_categories = 10

        elif self.name == 'cifar100':
            loader = partial(mltk.data.load_cifar100, x_dtype=np.uint8)
            n_categories = 100

        (train_x, train_y), (test_x, test_y) = loader()
        val_x, val_y = None, None
        if val_split is not None and val_split > 0.:
            (train_x, train_y), (val_x, val_y) = mltk.utils.split_numpy_arrays(
                [train_x, train_y], portion=val_split, shuffle=True)

        slots = {
            'x': ArrayInfo(
                dtype='uint8', shape=[32, 32, 3], is_discrete=True, min_val=0,
                max_val=255, n_discrete_vals=256, bit_depth=8),
            'y': ArrayInfo(
                dtype='int32', shape=[], is_discrete=True, min_val=0,
                max_val=n_categories - 1, n_discrete_vals=n_categories,
                bit_depth=get_bit_depth(n_categories)
            ),
        }
        splits = {
            'train': SplitInfo(data_count=len(train_x)),
            'val': None,
            'test': SplitInfo(data_count=len(test_x)),
        }
        arrays = {
            'train': {'x': train_x, 'y': train_y},
            'val': None,
            'test': {'x': test_x, 'y': test_y},
        }

        if val_x is not None:
            splits['val'] = SplitInfo(data_count=len(val_x))
            arrays['val'] = {'x': val_x, 'y': val_y}
        else:
            splits.pop('val')
            arrays.pop('val')

        super().__init__(splits=splits, slots=slots, arrays=arrays)


class Cifar10(BaseCifar):
    name: str = 'cifar10'


class Cifar100(BaseCifar):
    name: str = 'cifar100'

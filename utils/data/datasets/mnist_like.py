from functools import partial
from typing import *

import mltk
import numpy as np

from ...misc import get_bit_depth
from ..types import *
from .base import *

__all__ = ['MNIST', 'FashionMNIST']


class BaseMNISTLike(ArrayDataSet):

    name: str

    def __init__(self, val_split: Optional[float] = None):
        if self.name == 'mnist':
            loader = partial(mltk.data.load_mnist, x_shape=(28, 28, 1), x_dtype=np.uint8)
            n_categories = 10

        elif self.name == 'fashion_mnist':
            loader = partial(mltk.data.load_fashion_mnist, x_shape=(28, 28, 1), x_dtype=np.uint8)
            n_categories = 10

        (train_x, train_y), (test_x, test_y) = loader()
        val_x, val_y = None, None
        if val_split is not None and val_split > 0.:
            (train_x, train_y), (val_x, val_y) = mltk.utils.split_numpy_arrays(
                [train_x, train_y], portion=val_split, shuffle=True)

        slots = {
            'x': ArrayInfo(
                dtype='uint8', shape=[28, 28, 1], is_discrete=True, min_val=0,
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


class MNIST(BaseMNISTLike):
    name = 'mnist'


class FashionMNIST(BaseMNISTLike):
    name = 'fashion_mnist'

'''
load lsun dataset as numpy array

usage:

    import lsun

    (test_x, test_y) = load_lsun_test()

'''
import tarfile

from PIL import Image
from scipy.ndimage import filters
import os
import tensorflow as tf
import numpy as np
import io

TEST_X_PATH = '/home/cwx17/data/lsun'
TRAIN_X_ARR_PATH = '/home/cwx17/new_data/constant/train.npy'
TEST_X_ARR_PATH = '/home/cwx17/new_data/constant/test.npy'


def load_constant(x_shape=(32, 32, 3), x_dtype=np.float32, y_dtype=np.int32,
                  normalize_x=False):
    """
    Load the lsun dataset as NumPy arrays.
    samilar to load_not_mnist

    Args:
        Unimplemented!(haven't found a good way to resize) x_shape: Reshape each digit into this shape.  Default ``(218, 178)``.
        x_dtype: Cast each digit into this data type.  Default `np.float32`.
        y_dtype: Cast each label into this data type.  Default `np.int32`.
        normalize_x (bool): Whether or not to normalize x into ``[0, 1]``,
            by dividing each pixel value with 255.?  (default :obj:`False`)

    Returns:
        (np.ndarray, np.ndarray), (np.ndarray, np.ndarray): The
            (train_x, train_y), (test_x, test_y)
            
    """

    train_x = np.load(TRAIN_X_ARR_PATH)
    test_x = np.load(TEST_X_ARR_PATH)
    train_y = None
    test_y = None

    return (train_x, train_y), (test_x, test_y)


if __name__ == '__main__':
    load_constant()
    # arr, label = prepare_numpy('/home/cwx17/data/tinyimagenet/tiny-imagenet-200/train/')
    # np.save('/home/cwx17/data/tinyimagenet/train', arr, allow_pickle=False)
    # np.save('/home/cwx17/data/tinyimagenet/train_label', label, allow_pickle=False)
    # print(arr.shape)
    # print(label.shape)

    # (x_test, y_test) = load_lsun_test()
    # print(x_test.shape)
    # np.save(TEST_X_PATH, x_test)
    # export_images('/home/cwx17/data/lsungit/bedroom_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/classroom_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/kitchen_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/bridge_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/conference_room_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/living_room_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/tower_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/church_outdoor_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/dining_room_train_lmdb')
    # export_images('/home/cwx17/data/lsungit/restaurant_train_lmdb')

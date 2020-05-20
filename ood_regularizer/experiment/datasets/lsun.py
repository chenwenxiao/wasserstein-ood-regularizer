'''
load lsun dataset as numpy array

usage:

    import lsun

    (test_x, test_y) = load_lsun_test()

'''
import tarfile

from PIL import Image
from scipy.ndimage import filters
from scipy.misc import imresize, imsave
import os
import tensorflow as tf
import numpy as np

TEST_X_PATH = '/home/cwx17/data/lsun'
TEST_X_ARR_PATH = '/home/cwx17/data/lsun/test.npy'


def _fetch_array_x(path):
    file_names = os.listdir(path)
    file_names.sort()
    imgs = []
    scale = 148 / float(64)
    sigma = np.sqrt(scale) / 2.0
    for name in file_names:
        im = Image.open(os.path.join(path, name))
        wd = im.size[0]
        he = im.size[1]
        side = min(wd, he)
        dhe = he - side
        dwd = wd - side

        im = im.crop((dwd / 2, dhe / 2, wd - dwd / 2, he - dhe / 2))
        img = np.asarray(im)
        # img.setflags(write=True)
        # for dim in range(img.shape[2]):
        # img[...,dim] = filters.gaussian_filter(img[...,dim], sigma=(sigma,sigma))
        img = imresize(img, (32, 32, 3))
        if len(img.shape) > 2:
            imgs.append(img)

    return np.array(imgs)


def _fetch_array_y(path):
    evalue = []
    with open(path, 'rb') as f:
        for line in f.readlines():
            q = line.decode('utf-8')
            q = q.strip()
            q = int(q.split(' ')[1])
            evalue.append(q)
    return np.array(evalue)


def load_lsun_test(x_shape=(32, 32), x_dtype=np.float32, y_dtype=np.int32,
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
    if (not os.path.exists(TEST_X_PATH)):
        print("test dir not found")
        return

    test_x = np.load(TEST_X_ARR_PATH)

    test_y = range(0, len(test_x))

    return (test_x, test_y)


def prepare_numpy(path):
    imgs = []
    scale = 148 / float(64)
    sigma = np.sqrt(scale) / 2.0
    for root, dirs, files in os.walk(path):
        for name in files:
            try:
                if name.endswith('.jpg'):
                    im = Image.open(os.path.join(root, name))
                    print(len(imgs), os.path.join(root, name))
                    wd = im.size[0]
                    he = im.size[1]
                    side = min(wd, he)
                    dhe = he - side
                    dwd = wd - side

                    im = im.crop((dwd / 2, dhe / 2, wd - dwd / 2, he - dhe / 2))
                    img = np.asarray(im)
                    # img.setflags(write=True)
                    # for dim in range(img.shape[2]):
                    # img[...,dim] = filters.gaussian_filter(img[...,dim], sigma=(sigma,sigma))
                    img = imresize(img, (32, 32, 3))
                    if len(img.shape) > 2:
                        imgs.append(img)
            except Exception as e:
                print(e)
    imgs = np.array(imgs)
    np.random.shuffle(imgs)
    return imgs


if __name__ == '__main__':
    arr = prepare_numpy('/home/cwx17/data/sun/SUN397/')
    np.save('/home/cwx17/data/sun/train', arr[:int(len(arr) * 0.7)])
    np.save('/home/cwx17/data/sun/test', arr[int(len(arr) * 0.7):])
    # (_x_test, _y_test) = load_lsun_test()
    # print(_x_test.shape)
    # np.save(TEST_X_PATH, _x_test)

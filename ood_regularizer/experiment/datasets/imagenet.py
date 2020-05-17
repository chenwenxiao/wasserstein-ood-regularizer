'''
load imagenet test dataset as numpy array
len(test_x)==50000

usage:

    import imagenet

    test_x, test_y = imagenet.load_imagenet_test()


'''
from PIL import Image
from scipy.ndimage import filters
from scipy.misc import imresize, imsave
import os
import tensorflow as tf
import numpy as np

# TRAIN_DIR_PATH = '/home/cwx17/data/imagenet/train'
# TRAIN_X_PATH = '/home/cwx17/data/imagenet/train/img'
# TRAIN_X_ARR_PATH = '/home/cwx17/data/imagenet/train/imgarr.npy'

TEST_DIR_PATH = '/home/cwx17/data/imagenet/test/valid_32x32'
TEST_X_PATH = '/home/cwx17/data/imagenet/test/valid_32x32'
TEST_X_ARR_PATH = '/home/cwx17/data/imagenet/test.npy'


def _fetch_array_x(path):
    file_names = os.listdir(path)
    file_names.sort()
    imgs = []
    # scale = 148 / float(64)
    # sigma = np.sqrt(scale) / 2.0
    for name in file_names:
        im = Image.open(os.path.join(path, name))
        img = np.asarray(im)
        # print(img.shape)
        if img.shape[0] != 32:
            print('err')
        # im = im.crop((15,40,163,188))
        # # img.setflags(write=True)
        # # for dim in range(img.shape[2]):
        # #     img[...,dim] = filters.gaussian_filter(img[...,dim], sigma=(sigma,sigma))
        # img = imresize(img,(32,32,3))
        imgs.append(img)
        print(len(imgs))

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


def load_imagenet_test(x_shape=(32, 32), x_dtype=np.float32, y_dtype=np.int32,
                       normalize_x=False):
    """
    Load the imagenet dataset as NumPy arrays.
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

    test_x = np.load(TEST_X_ARR_PATH)
    test_y = np.array(range(0, len(test_x)))

    return (test_x, test_y)


if __name__ == '__main__':
    print('pre load')
    (_x_test, _y_test) = load_imagenet_test()
    print(_x_test.shape)

    np.save(TEST_X_PATH, _x_test)

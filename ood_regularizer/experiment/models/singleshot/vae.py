# -*- coding: utf-8 -*-
import functools
import sys
from argparse import ArgumentParser
from contextlib import contextmanager
import tensorflow as tf
from pprint import pformat

from matplotlib import pyplot
from tensorflow.contrib.framework import arg_scope, add_arg_scope

import tfsnippet as spt
from tfsnippet import DiscretizedLogistic
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      bernoulli_flow,
                                      print_with_title)
import numpy as np

from tfsnippet.preprocessing import UniformNoiseSampler

from ood_regularizer.experiment.datasets.overall import load_overall
from ood_regularizer.experiment.datasets.svhn import load_svhn
from ood_regularizer.experiment.utils import make_diagram, plot_fig, get_ele
import os
import scipy
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 256
    act_norm = False
    weight_norm = False
    batch_norm = False
    l2_reg = 0.0002
    kernel_size = 3
    shortcut_kernel_size = 1
    nf_layers = 20

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 30
    warm_up_start = 300
    initial_beta = -3.0
    uniform_scale = False
    use_transductive = True
    mixed_train = False
    mixed_train_epoch = 256
    mixed_train_skip = 64
    mixed_times = 64
    mixed_replace = 64
    mixed_replace_ratio = 1.0
    augment_range = 0
    dynamic_epochs = False
    retrain_for_batch = True
    in_dataset_test_ratio = 1.0
    pretrain = True
    distill_ratio = 1.0
    stand_weight = 1.0

    in_dataset = 'cifar10'
    out_dataset = 'svhn'

    max_step = None
    batch_size = 64
    smallest_step = 5e-5
    initial_lr = 0.0002
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = []
    lr_anneal_step_freq = None

    n_critical = 5
    # evaluation parameters
    train_n_qz = 1
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 100
    plot_epoch_freq = 20

    epsilon = -20.0
    min_logstd_of_q = -3.0

    sample_n_z = 100

    x_shape = (32, 32, 3)
    x_shape_multiple = 3072
    extra_stride = 2
    count_experiment = False


config = ExpConfig()


@add_arg_scope
def batch_norm(inputs, training=False, scope=None):
    return tf.layers.batch_normalization(inputs, training=training, name=scope)


@add_arg_scope
def dropout(inputs, training=False, scope=None):
    print(inputs, training)
    return spt.layers.dropout(inputs, rate=0.2, training=training, name=scope)


@add_arg_scope
@spt.global_reuse
def compress_conv2d(input):
    return spt.layers.conv2d(input, 1, strides=2, scope='level_0')  # output: (28, 28, 16)


@add_arg_scope
@spt.global_reuse
def entropy_net(x):
    normalizer_fn = None
    # x = tf.round(256.0 * x / 2 + 127.5)
    # x = (x - 127.5) / 256.0 * 2
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, strides=2, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, strides=2, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, strides=2, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, strides=2, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    # sample z ~ q(z|x)
    h_x = spt.layers.dense(h_x, 1, scope='level_-1', use_bias=True)
    # h_x = tf.clip_by_value(h_x, -1000, 1000)
    return tf.squeeze(h_x, axis=-1)


class MyIterator(object):
    def __init__(self, iterator):
        self._iterator = iter(iterator)
        self._next = None
        self._has_next = True
        self.next()

    @property
    def has_next(self):
        return self._has_next

    def next(self):
        if not self._has_next:
            raise StopIteration()

        ret = self._next
        try:
            self._next = next(self._iterator)
        except StopIteration:
            self._next = None
            self._has_next = False
        else:
            self._has_next = True
        return ret

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()


def limited(iterator, n):
    i = 0
    try:
        while i < n:
            yield next(iterator)
            i += 1
    except StopIteration:
        pass


def get_var(name):
    pfx = name.rsplit('/', 1)
    if len(pfx) == 2:
        vars = tf.global_variables(pfx[0] + '/')
    else:
        vars = tf.global_variables()
    for var in vars:
        if var.name.split(':', 1)[0] == name:
            return var
    raise NameError('Variable {} not exist.'.format(name))


def main():
    # parse the arguments
    arg_parser = ArgumentParser()
    spt.register_config_arguments(config, arg_parser, title='Model options')
    spt.register_config_arguments(spt.settings, arg_parser, prefix='tfsnippet',
                                  title='TFSnippet options')
    arg_parser.parse_args(sys.argv[1:])

    # print the config
    print_with_title('Configurations', pformat(config.to_dict()), after='\n')

    # open the result object and prepare for result directories
    results = MLResults(config.result_dir)
    results.save_config(config)  # save experiment settings for review
    while True:
        try:
            results.make_dirs('plotting/sample', exist_ok=True)
            results.make_dirs('plotting/z_plot', exist_ok=True)
            results.make_dirs('plotting/train.reconstruct', exist_ok=True)
            results.make_dirs('plotting/test.reconstruct', exist_ok=True)
            results.make_dirs('train_summary', exist_ok=True)
            results.make_dirs('checkpoint/checkpoint', exist_ok=True)
            break
        except Exception:
            pass

    if config.count_experiment:
        with open('/home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/count_experiments', 'a') as f:
            f.write(results.system_path("") + '\n')
            f.close()

    # prepare for training and testing data
    (x_train, y_train, x_test, y_test) = load_overall(config.in_dataset)
    (svhn_train, svhn_train_y, svhn_test, svhn_test_y) = load_overall(config.out_dataset)

    def normalize(x):
        return [(x - 127.5) / 256.0 * 2]

    def normalize2d(x, y):
        return [(x - 127.5) / 256.0 * 2, y]

    config.x_shape = x_train.shape[1:]
    config.x_shape_multiple = 1
    for x in config.x_shape:
        config.x_shape_multiple *= x
    if config.x_shape == (28, 28, 1):
        config.extra_stride = 1

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    input_ll = tf.placeholder(
        dtype=tf.float32, shape=(None,), name='input_ll')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_q_net = entropy_net(input_x)
        VAE_loss = tf.reduce_sum((train_q_net - input_ll) ** 2)
        VAE_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_entropy_op = entropy_net(input_x)

    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('entropy_net')
        with tf.variable_scope('theta_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)

    datasets = ['--in_dataset=celeba --out_dataset=tinyimagenet',
                '--in_dataset=celeba --out_dataset=svhn',
                '--in_dataset=celeba --out_dataset=cifar10',
                '--in_dataset=celeba --out_dataset=cifar100',
                '--in_dataset=celeba --out_dataset=isun',
                '--in_dataset=celeba --out_dataset=lsun',
                '--in_dataset=celeba --out_dataset=constant',
                '--in_dataset=celeba --out_dataset=noise',
                '--in_dataset=tinyimagenet --out_dataset=celeba',
                '--in_dataset=tinyimagenet --out_dataset=svhn',
                '--in_dataset=tinyimagenet --out_dataset=isun',
                '--in_dataset=tinyimagenet --out_dataset=lsun',
                '--in_dataset=tinyimagenet --out_dataset=constant',
                '--in_dataset=tinyimagenet --out_dataset=noise',
                '--in_dataset=svhn --out_dataset=celeba',
                '--in_dataset=svhn --out_dataset=tinyimagenet',
                '--in_dataset=svhn --out_dataset=cifar10',
                '--in_dataset=svhn --out_dataset=cifar100',
                '--in_dataset=svhn --out_dataset=isun',
                '--in_dataset=svhn --out_dataset=lsun',
                '--in_dataset=svhn --out_dataset=constant',
                '--in_dataset=svhn --out_dataset=noise',
                '--in_dataset=cifar10 --out_dataset=celeba',
                '--in_dataset=cifar10 --out_dataset=svhn',
                '--in_dataset=cifar10 --out_dataset=isun',
                '--in_dataset=cifar10 --out_dataset=lsun',
                '--in_dataset=cifar10 --out_dataset=constant',
                '--in_dataset=cifar10 --out_dataset=noise',
                '--in_dataset=cifar100 --out_dataset=celeba',
                '--in_dataset=cifar100 --out_dataset=svhn',
                '--in_dataset=cifar100 --out_dataset=isun',
                '--in_dataset=cifar100 --out_dataset=lsun',
                '--in_dataset=cifar100 --out_dataset=constant',
                '--in_dataset=cifar100 --out_dataset=noise',
                '--in_dataset=constant --out_dataset=celeba',
                '--in_dataset=constant --out_dataset=tinyimagenet',
                '--in_dataset=constant --out_dataset=svhn',
                '--in_dataset=constant --out_dataset=cifar10',
                '--in_dataset=constant --out_dataset=cifar100',
                '--in_dataset=constant --out_dataset=isun',
                '--in_dataset=constant --out_dataset=lsun',
                '--in_dataset=constant --out_dataset=noise',
                '--in_dataset=noise --out_dataset=celeba',
                '--in_dataset=noise --out_dataset=tinyimagenet',
                '--in_dataset=noise --out_dataset=svhn',
                '--in_dataset=noise --out_dataset=cifar10',
                '--in_dataset=noise --out_dataset=cifar100',
                '--in_dataset=noise --out_dataset=isun',
                '--in_dataset=noise --out_dataset=lsun',
                '--in_dataset=noise --out_dataset=constant',
                '--in_dataset=mnist28 --out_dataset=fashion_mnist28',
                '--in_dataset=mnist28 --out_dataset=kmnist28',
                '--in_dataset=mnist28 --out_dataset=not_mnist28',
                '--in_dataset=mnist28 --out_dataset=omniglot28',
                '--in_dataset=mnist28 --out_dataset=constant28',
                '--in_dataset=mnist28 --out_dataset=noise28',
                '--in_dataset=fashion_mnist28 --out_dataset=mnist28',
                '--in_dataset=fashion_mnist28 --out_dataset=kmnist28',
                '--in_dataset=fashion_mnist28 --out_dataset=not_mnist28',
                '--in_dataset=fashion_mnist28 --out_dataset=omniglot28',
                '--in_dataset=fashion_mnist28 --out_dataset=constant28',
                '--in_dataset=fashion_mnist28 --out_dataset=noise28',
                '--in_dataset=kmnist28 --out_dataset=mnist28',
                '--in_dataset=kmnist28 --out_dataset=fashion_mnist28',
                '--in_dataset=kmnist28 --out_dataset=not_mnist28',
                '--in_dataset=kmnist28 --out_dataset=omniglot28',
                '--in_dataset=kmnist28 --out_dataset=constant28',
                '--in_dataset=kmnist28 --out_dataset=noise28',
                '--in_dataset=not_mnist28 --out_dataset=mnist28',
                '--in_dataset=not_mnist28 --out_dataset=fashion_mnist28',
                '--in_dataset=not_mnist28 --out_dataset=kmnist28',
                '--in_dataset=not_mnist28 --out_dataset=omniglot28',
                '--in_dataset=not_mnist28 --out_dataset=constant28',
                '--in_dataset=not_mnist28 --out_dataset=noise28',
                '--in_dataset=omniglot28 --out_dataset=mnist28',
                '--in_dataset=omniglot28 --out_dataset=fashion_mnist28',
                '--in_dataset=omniglot28 --out_dataset=kmnist28',
                '--in_dataset=omniglot28 --out_dataset=not_mnist28',
                '--in_dataset=omniglot28 --out_dataset=constant28',
                '--in_dataset=omniglot28 --out_dataset=noise28',
                '--in_dataset=constant28 --out_dataset=mnist28',
                '--in_dataset=constant28 --out_dataset=fashion_mnist28',
                '--in_dataset=constant28 --out_dataset=kmnist28',
                '--in_dataset=constant28 --out_dataset=not_mnist28',
                '--in_dataset=constant28 --out_dataset=omniglot28',
                '--in_dataset=constant28 --out_dataset=noise28',
                '--in_dataset=noise28 --out_dataset=mnist28',
                '--in_dataset=noise28 --out_dataset=fashion_mnist28',
                '--in_dataset=noise28 --out_dataset=kmnist28',
                '--in_dataset=noise28 --out_dataset=not_mnist28',
                '--in_dataset=noise28 --out_dataset=omniglot28',
                '--in_dataset=noise28 --out_dataset=constant28',
                '--in_dataset=cifar10 --out_dataset=tinyimagenet',
                '--in_dataset=cifar10 --out_dataset=cifar100',
                '--in_dataset=cifar100 --out_dataset=tinyimagenet',
                '--in_dataset=cifar100 --out_dataset=cifar10',
                '--in_dataset=tinyimagenet --out_dataset=cifar10',
                '--in_dataset=tinyimagenet --out_dataset=cifar100']
    experiments_dirs = ["/mnt/mfs/mlstorage-experiments/cwx17/ff/f5/02279d802d3a37b583f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/4e/f5/02c52d867e432f1683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/1e/f5/02c52d867e43e67583f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/3c/e5/02732c28dc8d397683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/00/06/02279d802d3a05d583f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/be/f5/02c52d867e430c3783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/8b/e5/02732c28dc8de81583f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/76/e5/02812baa4f702b1683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/7e/f5/02c52d867e43a7d683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/10/06/02279d802d3a58c683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/86/e5/02812baa4f709a8683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/4c/e5/02732c28dc8d22a683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/6e/f5/02c52d867e4383c683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/70/06/02279d802d3a772883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/5c/e5/02732c28dc8dcd7783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/9e/f5/02c52d867e43340783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/2c/e5/02732c28dc8d9b2683f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/ce/f5/02c52d867e43496783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/de/f5/02c52d867e43359783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/ee/f5/02c52d867e4348c783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/30/06/02279d802d3a8e6783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/ae/f5/02c52d867e43ae1783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/1f/f5/02c52d867e434b3883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/fe/f5/02c52d867e43be0883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/c6/e5/02812baa4f70086883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/b0/06/02279d802d3acea883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/60/06/02279d802d3ab00883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/9c/e5/02732c28dc8d0a9883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/ac/e5/02732c28dc8dfa9883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/0f/f5/02c52d867e43e12883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/3f/f5/02c52d867e4324e883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/fc/e5/02732c28dc8da24983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/c0/06/02279d802d3ad4d883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/bc/e5/02732c28dc8d61c883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/f6/e5/02812baa4f70d15983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/d6/e5/02812baa4f70018883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/d0/06/02279d802d3ad07983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/4f/f5/02c52d867e43371983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/ec/e5/02732c28dc8d053983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/dc/e5/02732c28dc8d9f0983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/cc/e5/02732c28dc8dcbf883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/6f/f5/02c52d867e43d77983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/f0/06/02279d802d3aefc983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/af/f5/02c52d867e439c2a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/cf/f5/02c52d867e43a96a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/17/e5/02812baa4f70c1f983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/42/06/02279d802d3aef7c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/61/06/02279d802d3a59da83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/87/e5/02812baa4f708c3c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/c1/06/02279d802d3aedfb83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/77/e5/02812baa4f70db3c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/bf/f5/02c52d867e433e2a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/7f/f5/02c52d867e4360b983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/07/e5/02812baa4f7041b983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/e0/06/02279d802d3a717983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/37/e5/02812baa4f70d99a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/71/06/02279d802d3a3c0b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/ef/f5/02c52d867e4352ea83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/81/06/02279d802d3a532b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/8f/f5/02c52d867e4372b983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/01/06/02279d802d3a39e983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/f0/06/02c52d867e432d7c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/27/e5/02812baa4f70e85a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/df/f5/02c52d867e4307aa83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/ff/f5/02c52d867e43c82b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/0d/e5/02732c28dc8d5f6a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/5f/f5/02c52d867e43513983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/b1/06/02279d802d3a058b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/1d/e5/02732c28dc8da7ea83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/50/06/02c52d867e4349ab83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/21/06/02279d802d3a982a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/57/e5/02812baa4f70d4cb83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/41/06/02c52d867e431e9c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/60/06/02c52d867e43ecfb83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/11/06/02279d802d3a8ae983f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/18/e5/02812baa4f70198c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/10/06/02c52d867e434d4b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/91/06/02279d802d3a946b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/2d/e5/02732c28dc8da80b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/9f/f5/02c52d867e43922a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/51/06/02279d802d3ad8ca83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/00/06/02c52d867e434f3b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/a1/06/02279d802d3a0a6b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/47/e5/02812baa4f706ebb83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/80/06/02c52d867e43f76c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/70/06/02c52d867e430a2c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/20/06/02c52d867e43ce7b83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/67/e5/02812baa4f707aeb83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/d1/06/02279d802d3af83c83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/4d/e5/02732c28dc8dfbfb83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/41/06/02279d802d3acbaa83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/31/06/02279d802d3a5d6a83f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/40/06/02279d802d3acdc783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/50/06/02279d802d3aace783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/a0/06/02279d802d3afd6883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/e6/e5/02812baa4f7059f883f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/20/06/02279d802d3aad2783f5/",
                        "/mnt/mfs/mlstorage-experiments/cwx17/8e/f5/02c52d867e43add683f5/"]

    def load_data(a_name, b_name, train=False):
        s = "--in_dataset={} --out_dataset={}".format(a_name, b_name)
        index = datasets.index(s)
        dir = experiments_dirs[index]
        if train:
            a_ll_path = '{}log_prob_histogram{} Train.npy'.format(dir, a_name)
            b_ll_path = '{}log_prob_histogram{} Train.npy'.format(dir, b_name)
        else:
            a_ll_path = '{}log_prob_histogram{} Test.npy'.format(dir, a_name)
            b_ll_path = '{}log_prob_histogram{} Test.npy'.format(dir, b_name)
        return np.load(a_ll_path), np.load(b_ll_path)

    if '28' in config.in_dataset:
        assist_datasets = ['noise28', 'constant28']
    else:
        assist_datasets = ['noise', 'constant']

    cifar_train_nll, _ = load_data(config.in_dataset, config.out_dataset, train=True)
    cifar_test_nll, svhn_test_nll = load_data(config.in_dataset, config.out_dataset, train=False)
    print(np.mean(cifar_train_nll), np.std(cifar_train_nll))
    print(np.mean(cifar_test_nll), np.std(cifar_test_nll))
    print(np.mean(svhn_test_nll), np.std(svhn_test_nll))
    entropy_x = [x_train]
    entropy_nll = [cifar_train_nll]
    for assist_dataset in assist_datasets:
        (assist_train, _, __, ___) = load_overall(assist_dataset)
        entropy_x.append(assist_train)
        assist_nll, _ = load_data(assist_dataset, config.in_dataset, train=True)
        entropy_nll.append(assist_nll)
    entropy_x = np.concatenate(entropy_x)
    entropy_nll = np.concatenate(entropy_nll)
    print(entropy_x.shape, entropy_nll.shape)

    train_flow = spt.DataFlow.arrays([entropy_x, entropy_nll], config.batch_size, shuffle=True,
                                     skip_incomplete=True).map(normalize2d)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        experiment_dict = {
        }
        print(experiment_dict)
        if config.in_dataset in experiment_dict:
            restore_dir = experiment_dict[config.in_dataset] + '/checkpoint'
            restore_checkpoint = os.path.join(
                restore_dir, 'checkpoint',
                'checkpoint.dat-{}'.format(config.max_epoch))
        else:
            restore_dir = results.system_path('checkpoint')
            restore_checkpoint = None

        # train the network
        with spt.TrainLoop(tf.trainable_variables(),
                           var_groups=['entropy_net'],
                           max_epoch=config.max_epoch + 1,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False,
                           checkpoint_dir=results.system_path('checkpoint'),
                           restore_checkpoint=restore_checkpoint
                           ) as loop:

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()
            # print(loop.epoch)
            # adversarial training
            for epoch in epoch_iterator:
                if epoch > config.max_epoch:

                    loop.collect_metrics(ll_histogram=plot_fig(
                        data_list=[cifar_test_nll, svhn_test_nll],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        x_label='bits/dim',
                        fig_name='ll_histogram'))
                    cifar_test_entropy = -get_ele(test_entropy_op,
                                                  spt.DataFlow.arrays([x_test], config.test_batch_size).map(normalize),
                                                  input_x)
                    svhn_test_entropy = -get_ele(test_entropy_op,
                                                 spt.DataFlow.arrays([svhn_test], config.test_batch_size).map(
                                                     normalize),
                                                 input_x)
                    cifar_train_entropy = -get_ele(test_entropy_op,
                                                   spt.DataFlow.arrays([x_train], config.test_batch_size).map(
                                                       normalize),
                                                   input_x)
                    loop.collect_metrics(predict_entropy_histogram=plot_fig(
                        data_list=[-cifar_test_entropy, -svhn_test_entropy],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                        x_label='bits/dim',
                        fig_name='predict_entropy_histogram'))
                    cifar_kl = cifar_test_entropy + cifar_test_nll
                    svhn_kl = svhn_test_entropy + svhn_test_nll
                    cifar_train_kl = cifar_train_entropy + cifar_train_nll

                    loop.collect_metrics(kl_histogram=plot_fig([-cifar_kl, -svhn_kl],
                                                               ['red', 'green'],
                                                               [config.in_dataset + ' Test',
                                                                config.out_dataset + ' Test'], 'log(bit/dims)',
                                                               'kl_histogram'))

                    def stand(base, another_arrays=None):
                        mean, std = np.mean(base), np.std(base)
                        return_arrays = []
                        for array in another_arrays:
                            return_arrays.append(-np.abs((array - mean) / std) * config.stand_weight)
                        return return_arrays

                    [cifar_kl_stand, svhn_kl_stand] = stand(cifar_train_kl, [cifar_kl, svhn_kl])
                    loop.collect_metrics(kl_stand_histogram=plot_fig([cifar_kl_stand, svhn_kl_stand],
                                                                     ['red', 'green'],
                                                                     [config.in_dataset + ' Test',
                                                                      config.out_dataset + ' Test'], 'log(bit/dims)',
                                                                     'stand_kl_histogram'))

                    [cifar_test_stand, svhn_test_stand] = stand(cifar_train_nll, [cifar_test_nll, svhn_test_nll])
                    loop.collect_metrics(stand_histogram=plot_fig([cifar_test_stand, svhn_test_stand],
                                                                  ['red', 'green'],
                                                                  [config.in_dataset + ' Test',
                                                                   config.out_dataset + ' Test'],
                                                                  'log(bit/dims)',
                                                                  'stand_histogram'))
                    cifar_kl_mean = np.mean(cifar_train_kl)
                    cifar_kl_std = np.std(cifar_train_kl)
                    loop.collect_metrics(kl_with_stand_histogram=plot_fig([
                        cifar_test_stand - (cifar_kl - cifar_kl_mean) / cifar_kl_std,
                        svhn_test_stand - (svhn_kl - cifar_kl_mean) / cifar_kl_std],
                        ['red', 'green'],
                        [config.in_dataset + ' Test',
                         config.out_dataset + ' Test'],
                        'log(bit/dims)',
                        'kl_with_stand_histogram'))

                    loop.collect_metrics(stand_kl_with_stand_ll_histogram=plot_fig(
                        [cifar_kl_stand + cifar_test_stand, svhn_kl_stand + svhn_test_stand],
                        ['red', 'green'],
                        [config.in_dataset + ' Test',
                         config.out_dataset + ' Test'],
                        'log(bit/dims)',
                        'stand_kl_with_stand_ll_histogram'))
                    cifar_min = np.where(cifar_kl_stand < cifar_test_stand, cifar_kl_stand,
                                         cifar_test_stand)
                    svhn_min = np.where(svhn_kl_stand < svhn_test_stand, svhn_kl_stand, svhn_test_stand)

                    loop.collect_metrics(max_stand_kl_with_stand_ll_histogram=plot_fig(
                        [cifar_min, svhn_min],
                        ['red', 'green'],
                        [config.in_dataset + ' Test',
                         config.out_dataset + ' Test'],
                        'log(bit/dims)',
                        'max_stand_kl_with_stand_ll_histogram'))

                    break

                for step, [x, ll] in loop.iter_steps(train_flow):
                    _, batch_VAE_loss = session.run([VAE_train_op, VAE_loss], feed_dict={
                        input_x: x, input_ll: ll
                    })
                    loop.collect_metrics(VAE_loss=batch_VAE_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.max_epoch:
                    loop._checkpoint_saver.save(epoch)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()

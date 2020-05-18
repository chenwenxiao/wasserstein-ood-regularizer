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
from ood_regularizer.experiment.utils import make_diagram, plot_fig


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
    max_epoch = 300
    warm_up_start = 300
    initial_beta = -3.0
    uniform_scale = True
    use_transductive = True
    mixed_train_epoch = 10
    mixed_train_skip = 1024
    initial_omega_with_theta = True
    dynamic_epochs = True
    self_ood = False

    in_dataset = 'cifar10'
    out_dataset = 'svhn'

    max_step = None
    batch_size = 128
    smallest_step = 5e-5
    initial_lr = 0.0001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = []
    lr_anneal_step_freq = None

    n_critical = 5
    # evaluation parameters
    train_n_qz = 1
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 300
    plot_epoch_freq = 20

    epsilon = -20.0
    min_logstd_of_q = -3.0

    sample_n_z = 100

    x_shape = (32, 32, 3)


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
def q_net(x, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg), ):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        z_mean = spt.layers.dense(h_x, config.z_dim, scope='z_mean')
        z_logstd = spt.layers.dense(h_x, config.z_dim, scope='z_logstd')

    # sample z ~ q(z|x)
    z = net.add('z', spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.min_logstd_of_q)),
                n_samples=n_z, group_ndims=1)

    return net


@add_arg_scope
@spt.global_reuse
def p_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    z = net.add('z', normal, n_samples=n_z, group_ndims=1)

    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 128 * config.x_shape[0] // 4 * config.x_shape[1] // 4, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 4, config.x_shape[1] // 4, 128)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
        x_mean = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_mean',
            kernel_initializer=tf.zeros_initializer(), activation_fn=tf.nn.tanh
        )
        x_logstd = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_logstd',
            kernel_initializer=tf.zeros_initializer(),
        )

    beta = tf.get_variable(name='beta', shape=(), initializer=tf.constant_initializer(config.initial_beta),
                           dtype=tf.float32, trainable=True)
    x = net.add('x', DiscretizedLogistic(
        mean=x_mean,
        log_scale=spt.ops.maybe_clip_value(beta if config.uniform_scale else x_logstd, min_val=config.epsilon),
        bin_size=2.0 / 256.0,
        min_val=-1.0 + 1.0 / 256.0,
        max_val=1.0 - 1.0 / 256.0,
        epsilon=1e-10
    ), group_ndims=3)
    return net


@add_arg_scope
@spt.global_reuse
def q_omega_net(x, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg), ):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        z_mean = spt.layers.dense(h_x, config.z_dim, scope='z_mean')
        z_logstd = spt.layers.dense(h_x, config.z_dim, scope='z_logstd')

    # sample z ~ q(z|x)
    z = net.add('z', spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.min_logstd_of_q)),
                n_samples=n_z, group_ndims=1)

    return net


@add_arg_scope
@spt.global_reuse
def p_omega_net(observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    z = net.add('z', normal, n_samples=n_z, group_ndims=1)

    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 128 * config.x_shape[0] // 4 * config.x_shape[1] // 4, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 4, config.x_shape[1] // 4, 128)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
        x_mean = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_mean',
            kernel_initializer=tf.zeros_initializer(), activation_fn=tf.nn.tanh
        )
        x_logstd = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_logstd',
            kernel_initializer=tf.zeros_initializer(),
        )

    beta = tf.get_variable(name='beta', shape=(), initializer=tf.constant_initializer(config.initial_beta),
                           dtype=tf.float32, trainable=True)
    x = net.add('x', DiscretizedLogistic(
        mean=x_mean,
        log_scale=spt.ops.maybe_clip_value(beta if config.uniform_scale else x_logstd, min_val=config.epsilon),
        bin_size=2.0 / 256.0,
        min_val=-1.0 + 1.0 / 256.0,
        max_val=1.0 - 1.0 / 256.0,
        epsilon=1e-10
    ), group_ndims=3)
    return net


def get_all_loss(q_net, p_net):
    with tf.name_scope('adv_prior_loss'):
        train_recon = p_net['x'].log_prob()
        train_kl = tf.reduce_mean(
            -p_net['z'].log_prob() + q_net['z'].log_prob()
        )
        VAE_loss = -train_recon + train_kl
    return VAE_loss


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
    results.make_dirs('plotting/sample', exist_ok=True)
    results.make_dirs('plotting/z_plot', exist_ok=True)
    results.make_dirs('plotting/train.reconstruct', exist_ok=True)
    results.make_dirs('plotting/test.reconstruct', exist_ok=True)
    results.make_dirs('train_summary', exist_ok=True)

    # prepare for training and testing data
    (_x_train, _x_test) = load_overall(config.in_dataset)
    x_train = (_x_train - 127.5) / 256.0 * 2
    x_test = (_x_test - 127.5) / 256.0 * 2

    (svhn_train, svhn_test) = load_overall(config.out_dataset)
    svhn_train = (svhn_train - 127.5) / 256.0 * 2
    svhn_test = (svhn_test - 127.5) / 256.0 * 2

    config.x_shape = x_train.shape[1:]

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_q_net = q_net(input_x, n_z=config.train_n_qz)
        train_p_net = p_net(observed={'x': input_x, 'z': train_q_net['z']},
                            n_z=config.train_n_qz)
        VAE_loss = get_all_loss(train_q_net, train_p_net)
        train_q_omega_net = q_omega_net(input_x, n_z=config.train_n_qz)
        train_p_omega_net = p_omega_net(observed={'x': input_x, 'z': train_q_omega_net['z']},
                                        n_z=config.train_n_qz)
        VAE_omega_loss = get_all_loss(train_q_omega_net, train_p_omega_net)

        VAE_loss += tf.losses.get_regularization_loss()
        VAE_omega_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, n_z=config.test_n_qz)
        test_chain = test_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_qz, latent_axis=0)
        test_recon = tf.reduce_mean(
            test_chain.model['x'].log_prob()
        )
        ele_test_ll = test_chain.vi.evaluation.is_loglikelihood()
        test_nll = -tf.reduce_mean(
            ele_test_ll
        )
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

        test_q_omega_net = q_omega_net(input_x, n_z=config.test_n_qz)
        test_omega_chain = test_q_omega_net.chain(p_omega_net, observed={'x': input_x}, n_z=config.test_n_qz,
                                                  latent_axis=0)
        test_omega_recon = tf.reduce_mean(
            test_omega_chain.model['x'].log_prob()
        )
        ele_test_omega_ll = test_omega_chain.vi.evaluation.is_loglikelihood()
        test_omega_nll = -tf.reduce_mean(
            ele_test_omega_ll
        )
        test_omega_lb = tf.reduce_mean(test_omega_chain.vi.lower_bound.elbo())

        ele_test_kl = ele_test_omega_ll - ele_test_ll

    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables('p_net')
        VAE_omega_params = tf.trainable_variables('q_omega_net') + tf.trainable_variables('p_omega_net')
        with tf.variable_scope('theta_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)
        with tf.variable_scope('omega_optimizer'):
            VAE_omega_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_omega_grads = VAE_omega_optimizer.compute_gradients(VAE_omega_loss, VAE_omega_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)
            VAE_omega_train_op = VAE_omega_optimizer.apply_gradients(VAE_omega_grads)

        print(VAE_params)
        print(VAE_omega_params)
        copy_op = []
        for i in range(len(VAE_params)):
            copy_op.append(tf.assign(VAE_omega_params[i], VAE_params[i]))
            print(VAE_omega_params[i], VAE_omega_params[i])
        copy_op = tf.group(*copy_op)

    # derive the plotting function
    with tf.name_scope('plotting'):
        plot_net = p_net(n_z=config.sample_n_z)
        vae_plots = tf.reshape(plot_net['x'].distribution.mean, (-1,) + config.x_shape)
        vae_plots = 256.0 * vae_plots / 2 + 127.5
        reconstruct_q_net = q_net(input_x)
        reconstruct_z = reconstruct_q_net['z']
        reconstruct_plots = 256.0 * tf.reshape(
            p_net(observed={'z': reconstruct_z})['x'].distribution.mean,
            (-1,) + config.x_shape
        ) / 2 + 127.5
        reconstruct_plots = tf.clip_by_value(reconstruct_plots, 0, 255)
        vae_plots = tf.clip_by_value(vae_plots, 0, 255)

        plot_omega_net = p_omega_net(n_z=config.sample_n_z)
        vae_omega_plots = tf.reshape(plot_omega_net['x'].distribution.mean, (-1,) + config.x_shape)
        vae_omega_plots = 256.0 * vae_omega_plots / 2 + 127.5
        reconstruct_q_omega_net = q_omega_net(input_x)
        reconstruct_omega_z = reconstruct_q_omega_net['z']
        reconstruct_omega_plots = 256.0 * tf.reshape(
            p_omega_net(observed={'z': reconstruct_omega_z})['x'].distribution.mean,
            (-1,) + config.x_shape
        ) / 2 + 127.5
        reconstruct_omega_plots = tf.clip_by_value(reconstruct_omega_plots, 0, 255)
        vae_omega_plots = tf.clip_by_value(vae_omega_plots, 0, 255)

    def plot_samples(loop, extra_index=None):
        if extra_index is None:
            extra_index = loop.epoch

        try:
            with loop.timeit('plot_time'):
                # plot reconstructs
                def plot_reconnstruct(flow, name, plots):
                    for [x] in flow:
                        x_samples = x
                        images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                        images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                        images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                        batch_reconstruct_plots = session.run(
                            plots, feed_dict={input_x: x_samples})
                        images[2::3, ...] = np.round(batch_reconstruct_plots)
                        # print(np.mean(batch_reconstruct_z ** 2, axis=-1))
                        save_images_collection(
                            images=images,
                            filename='plotting/{}-{}.png'.format(name, extra_index),
                            grid_size=(20, 15),
                            results=results,
                        )
                        break

                plot_reconnstruct(reconstruct_test_flow, 'test.reconstruct/theta', reconstruct_plots)
                plot_reconnstruct(reconstruct_train_flow, 'train.reconstruct/theta', reconstruct_plots)
                plot_reconnstruct(reconstruct_omega_test_flow, 'test.reconstruct/omega', reconstruct_omega_plots)
                plot_reconnstruct(reconstruct_omega_train_flow, 'train.reconstruct/omega', reconstruct_omega_plots)

                # plot samples
                images = session.run(vae_plots)
                save_images_collection(
                    images=np.round(images),
                    filename='plotting/sample/{}-{}.png'.format('theta', extra_index),
                    grid_size=(10, 10),
                    results=results,
                )
                images = session.run(vae_omega_plots)
                save_images_collection(
                    images=np.round(images),
                    filename='plotting/sample/{}-{}.png'.format('omega', extra_index),
                    grid_size=(10, 10),
                    results=results,
                )
        except Exception as e:
            print(e)

    # prepare for training and testing data
    (_x_train, _y_train), (_x_test, _y_test) = spt.datasets.load_cifar10(x_shape=config.x_shape)
    x_train = (_x_train - 127.5) / 256.0 * 2
    x_test = (_x_test - 127.5) / 256.0 * 2
    cifar_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size)
    cifar_test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size)

    (svhn_train, _y_train), (svhn_test, _y_test) = load_svhn(x_shape=config.x_shape)
    svhn_train = (svhn_train - 127.5) / 256.0 * 2
    svhn_test = (svhn_test - 127.5) / 256.0 * 2
    svhn_train_flow = spt.DataFlow.arrays([svhn_train], config.test_batch_size)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test], config.test_batch_size)

    train_flow = spt.DataFlow.arrays([x_train], config.batch_size, shuffle=True, skip_incomplete=True)

    if config.use_transductive:
        mixed_array = np.concatenate([x_test, svhn_test])
    else:
        mixed_array = np.random.randint(0, 256, size=(20000,) + config.x_shape)
        mixed_array = (mixed_array - 127.5) / 256.0 * 2

    reconstruct_test_flow = spt.DataFlow.arrays([x_test], 100, shuffle=True, skip_incomplete=True)
    reconstruct_train_flow = spt.DataFlow.arrays([x_train], 100, shuffle=True, skip_incomplete=True)
    reconstruct_omega_test_flow = spt.DataFlow.arrays([svhn_test], 100, shuffle=True, skip_incomplete=True)
    reconstruct_omega_train_flow = spt.DataFlow.arrays([svhn_train], 100, shuffle=True, skip_incomplete=True)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        restore_checkpoint = None

        # train the network
        with spt.TrainLoop(tf.trainable_variables(),
                           var_groups=['q_net', 'p_net', 'posterior_flow', 'G_theta', 'D_psi', 'G_omega', 'D_kappa'],
                           max_epoch=config.max_epoch,
                           max_step=config.max_step,
                           summary_dir=(results.system_path('train_summary')
                                        if config.write_summary else None),
                           summary_graph=tf.get_default_graph(),
                           early_stopping=False,
                           checkpoint_dir=results.system_path('checkpoint'),
                           checkpoint_epoch_freq=100,
                           restore_checkpoint=restore_checkpoint
                           ) as loop:

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()
            # adversarial training
            for epoch in epoch_iterator:
                for step, [x] in loop.iter_steps(train_flow):
                    _, batch_VAE_loss = session.run([VAE_train_op, VAE_loss], feed_dict={
                        input_x: x
                    })
                    loop.collect_metrics(VAE_loss=batch_VAE_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.warm_up_start:
                    learning_rate.set(config.initial_lr)

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                if epoch == config.max_epoch:
                    if config.initial_omega_with_theta:
                        session.run(copy_op)
                    mixed_kl = []
                    for i in range(0, len(mixed_array), config.mixed_train_skip):
                        mixed_test_flow = spt.DataFlow.arrays([mixed_array[:i + config.mixed_train_skip]],
                                                              config.batch_size, shuffle=True)
                        if config.dynamic_epochs:
                            repeat_epoch = int(
                                config.mixed_train_epoch * len(mixed_array) / (mixed_test_flow.data_length))
                        else:
                            repeat_epoch = config.mixed_train_epoch
                        for pse_epoch in range(repeat_epoch):
                            for step, [x] in loop.iter_steps(mixed_test_flow):
                                _, batch_VAE_omega_loss = session.run([VAE_omega_train_op, VAE_omega_loss], feed_dict={
                                    input_x: x
                                })
                                loop.collect_metrics(VAE_omega_loss=batch_VAE_omega_loss)
                        for j in range(len(mixed_array[i: i + config.mixed_train_skip])):
                            mixed_kl.append(session.run(ele_test_kl, feed_dict={
                                input_x: mixed_array[i + j: i + j + 1]
                            })[0])
                        print(repeat_epoch, len(mixed_kl))
                        loop.print_logs()

                    mixed_kl = np.asarray(mixed_kl)
                    cifar_kl = mixed_kl[:len(x_test)]
                    svhn_kl = mixed_kl[len(x_test):]
                    plot_fig([-cifar_kl, -svhn_kl],
                             ['red', 'green'],
                             ['CIFAR-10 kl', 'SVHN kl'], 'log(bit/dims)',
                             'kl_histogram', auc_pair=(0, 1))

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()

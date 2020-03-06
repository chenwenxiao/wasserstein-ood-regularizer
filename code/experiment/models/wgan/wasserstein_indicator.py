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
from tfsnippet.examples.utils import (MLResults,
                                      save_images_collection,
                                      bernoulli_as_pixel,
                                      bernoulli_flow,
                                      bernoulli_flow,
                                      print_with_title)
import numpy as np

from tfsnippet.preprocessing import UniformNoiseSampler

from code.experiment.datasets.svhn import load_svhn
from code.experiment.utils import make_diagram


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 256
    act_norm = False
    weight_norm = False
    l2_reg = 0.0002
    kernel_size = 3
    shortcut_kernel_size = 1
    batch_norm = True
    nf_layers = 20

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 1000
    warm_up_start = 1000
    warm_up_epoch = 500
    beta = 1e-8
    initial_xi = 0.0  # TODO
    pull_back_energy_weight = 256

    use_gan = False
    max_step = None
    batch_size = 256
    initial_lr = 0.0001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    lr_anneal_step_freq = None

    gradient_penalty_algorithm = 'interpolate'  # both or interpolate
    gradient_penalty_weight = 2
    gradient_penalty_index = 6
    kl_balance_weight = 1.0

    n_critical = 5  # TODO
    # evaluation parameters
    train_n_pz = 256
    train_n_qz = 1
    test_n_pz = 1000
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 200
    plot_epoch_freq = 10
    grad_epoch_freq = 10

    test_fid_n_pz = 5000
    test_x_samples = 1
    log_Z_times = 10

    epsilon = -20

    @property
    def x_shape(self):
        return (32, 32, 3)

    x_shape_multiple = 32 * 32 * 3


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
def G_omega(z):
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 512 * config.x_shape[0] // 8 * config.x_shape[1] // 8, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 8, config.x_shape[1] // 8, 512)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 512, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 256, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
    x_mean = spt.layers.conv2d(
        h_z, config.x_shape[-1], (1, 1), padding='same', scope='feature_map_mean_to_pixel',
        kernel_initializer=tf.zeros_initializer(), activation_fn=tf.nn.tanh
    )
    return x_mean


@add_arg_scope
@spt.global_reuse
def D_psi(x, y=None):
    # if y is not None:
    #     return D_psi(y) + 0.1 * tf.sqrt(tf.reduce_sum((x - y) ** 2, axis=tf.range(-len(config.x_shape), 0)))
    # TODO
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
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_4')  # output: (14, 14, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_6')  # output: (7, 7, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 512, strides=2, scope='level_8')  # output: (7, 7, 64)

        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        h_x = spt.layers.dense(h_x, 64, scope='level_-2')
    # sample z ~ q(z|x)
    h_x = spt.layers.dense(h_x, 1, scope='level_-1')
    return tf.squeeze(h_x, axis=-1)


@add_arg_scope
@spt.global_reuse
def p_net(observed=None, n_z=None, beta=1.0):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    z = net.add('z', normal, n_samples=n_z, group_ndims=1)
    x_mean = G_omega(z)
    x = net.add('x', spt.Normal(
        mean=x_mean, logstd=0.0
    ), group_ndims=3)
    return net


def get_all_loss(input_x, input_y):
    with tf.name_scope('adv_prior_loss'):
        x = input_x
        x_ = input_y
        energy_real = D_psi(x)
        energy_fake = D_psi(x_)

        gradient_penalty = 0.0

        if config.gradient_penalty_algorithm == 'interpolate':
            # Sample from interpolates
            alpha = tf.random_uniform(
                tf.concat([[config.batch_size], [1] * len(config.x_shape)], axis=0),
                minval=0, maxval=1.0
            )
            x = tf.reshape(x, (-1,) + config.x_shape)
            x_ = tf.reshape(x_, (-1,) + config.x_shape)
            differences = x - x_
            interpolates = x_ + alpha * differences
            # print(interpolates)
            D_interpolates = D_psi(interpolates)
            # print(D_interpolates)
            gradient_penalty = tf.square(tf.gradients(D_interpolates, [interpolates])[0])
            gradient_penalty = tf.reduce_sum(gradient_penalty, tf.range(-len(config.x_shape), 0))
            gradient_penalty = tf.pow(gradient_penalty, config.gradient_penalty_index / 2.0)
            gradient_penalty = tf.reduce_mean(gradient_penalty) * config.gradient_penalty_weight

        if config.gradient_penalty_algorithm == 'both':
            # Sample from fake and real
            gradient_penalty_real = tf.square(tf.gradients(energy_real, [x.tensor if hasattr(x, 'tensor') else x])[0])
            gradient_penalty_real = tf.reduce_sum(gradient_penalty_real, tf.range(-len(config.x_shape), 0))
            gradient_penalty_real = tf.pow(gradient_penalty_real, config.gradient_penalty_index / 2.0)

            gradient_penalty_fake = tf.square(
                tf.gradients(energy_fake, [x_.tensor if hasattr(x_, 'tensor') else x_])[0])
            gradient_penalty_fake = tf.reduce_sum(gradient_penalty_fake, tf.range(-len(config.x_shape), 0))
            gradient_penalty_fake = tf.pow(gradient_penalty_fake, config.gradient_penalty_index / 2.0)

            gradient_penalty = (tf.reduce_mean(gradient_penalty_fake) + tf.reduce_mean(gradient_penalty_real)) \
                               * config.gradient_penalty_weight / 2.0

        adv_D_loss = -tf.reduce_mean(energy_fake) + tf.reduce_mean(
            energy_real) + gradient_penalty
        adv_G_loss = tf.reduce_mean(energy_fake)
    return adv_D_loss, adv_G_loss, tf.reduce_mean(energy_real)


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

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    input_y = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_y')
    warm = tf.placeholder(
        dtype=tf.float32, shape=(), name='warm')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    beta = tf.Variable(initial_value=0.0, dtype=tf.float32, name='beta', trainable=True)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_pn_omega = p_net(n_z=config.train_n_pz, beta=beta)
        D_loss, G_loss, D_real = get_all_loss(input_x, input_y)
        train_D_loss, train_G_loss, train_D_real = get_all_loss(train_pn_omega['x'].distribution.mean, input_x)
        D_loss += tf.losses.get_regularization_loss()
        G_loss += tf.losses.get_regularization_loss()
        train_D_loss += tf.losses.get_regularization_loss()
        train_G_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        ele_test_energy = D_psi(input_x)

    # derive the optimizer
    with tf.name_scope('optimizing'):
        D_params = tf.trainable_variables('D_psi')
        G_params = tf.trainable_variables('G_omega')
        print("========D_params=========")
        print(D_params)
        print("========G_params=========")
        print(G_params)
        with tf.variable_scope('D_optimizer'):
            D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            D_grads = D_optimizer.compute_gradients(D_loss, D_params)
            train_D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            train_D_grads = train_D_optimizer.compute_gradients(train_D_loss, D_params)
        with tf.variable_scope('G_optimizer'):
            train_G_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            train_G_grads = train_G_optimizer.compute_gradients(train_G_loss, G_params)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            train_G_train_op = train_G_optimizer.apply_gradients(train_G_grads)
            D_train_op = D_optimizer.apply_gradients(D_grads)
            train_D_train_op = train_D_optimizer.apply_gradients(train_D_grads)

        # derive the plotting function
        with tf.name_scope('plotting'):
            sample_n_z = config.sample_n_z
            plot_net = p_net(n_z=sample_n_z, beta=beta)
            x_plots = 256.0 * tf.reshape(
                plot_net['x'].distribution.mean, (-1,) + config.x_shape) / 2 + 127.5
            x_plots = tf.clip_by_value(x_plots, 0, 255)

        def plot_samples(loop, extra_index=None):
            if extra_index is None:
                extra_index = loop.epoch
            with loop.timeit('plot_time'):
                # plot reconstructs
                # plot samples
                images = session.run(x_plots)

                try:
                    save_images_collection(
                        images=np.round(images),
                        filename='plotting/sample/{}.png'.format(extra_index),
                        grid_size=(10, 10),
                        results=results,
                    )
                except Exception as e:
                    print(e)

                return images

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
    mixed_test_flow = spt.DataFlow.arrays([np.concatenate([x_test, svhn_test])], config.batch_size, shuffle=True,
                                          skip_incomplete=True)

    with spt.utils.create_session().as_default() as session, train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        restore_checkpoint = None

        # train the network
        with spt.TrainLoop(tf.trainable_variables(),
                           var_groups=['q_net', 'p_net', 'posterior_flow', 'G_theta', 'D_psi'],
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

            n_critical = config.n_critical
            # adversarial training
            for epoch in epoch_iterator:
                if epoch > config.warm_up_start:
                    if config.use_gan:
                        for [x] in mixed_test_flow:
                            # spec-training discriminator
                            [_, batch_D_loss, batch_D_real] = session.run(
                                [train_D_train_op, train_D_loss, train_D_real], feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(D_loss=batch_D_loss)
                            loop.collect_metrics(D_real=batch_D_real)

                    else:
                        for [x] in train_flow:
                            for [y] in mixed_test_flow:
                                # spec-training discriminator
                                [_, batch_D_loss, batch_D_real] = session.run(
                                    [D_train_op, D_loss, D_real], feed_dict={
                                        input_x: x, input_y: y
                                    })
                                loop.collect_metrics(D_loss=batch_D_loss)
                                loop.collect_metrics(D_real=batch_D_real)
                                break

                else:
                    step_iterator = MyIterator(train_flow)
                    while step_iterator.has_next:
                        for step, [x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            # pre-training discriminator
                            [_, batch_D_loss, batch_D_real] = session.run(
                                [train_D_train_op, train_D_loss, train_D_real], feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(D_loss=batch_D_loss)
                            loop.collect_metrics(D_real=batch_D_real)

                        # generator training
                        [_, batch_G_loss] = session.run(
                            [train_G_train_op, train_G_loss], feed_dict={
                            })
                        loop.collect_metrics(G_loss=batch_G_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.warm_up_start:
                    learning_rate.set(config.initial_lr)

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                if epoch % config.test_epoch_freq == 0:
                    make_diagram(
                        ele_test_energy,
                        [cifar_train_flow, cifar_test_flow, svhn_train_flow, svhn_test_flow], input_x,
                        fig_name='log_prob_histogram_{}'.format(epoch)
                    )

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
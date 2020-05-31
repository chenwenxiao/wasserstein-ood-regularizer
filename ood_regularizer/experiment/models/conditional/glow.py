# -*- coding: utf-8 -*-
import functools
import sys
import os
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

from ood_regularizer.experiment.datasets.celeba import load_celeba
from ood_regularizer.experiment.datasets.overall import load_overall
from ood_regularizer.experiment.datasets.svhn import load_svhn
from ood_regularizer.experiment.models.real_nvp import make_real_nvp, RealNVPConfig
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import make_diagram, get_ele, plot_fig


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
    max_epoch = 200
    warm_up_start = 200
    initial_beta = -3.0
    uniform_scale = False
    use_transductive = True
    mixed_train = False
    self_ood = False
    mixed_ratio = 1.0
    mutation_rate = 0.1
    noise_type = "mutation"  # or unit
    in_dataset_test_ratio = 1.0

    in_dataset = 'cifar10'
    out_dataset = 'svhn'

    max_step = None
    batch_size = 128
    smallest_step = 5e-5
    initial_lr = 0.0005
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = []
    lr_anneal_step_freq = None
    clip_norm = 5

    n_critical = 5
    # evaluation parameters
    train_n_qz = 1
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 200
    plot_epoch_freq = 20
    distill_ratio = 1.0
    distill_epoch = 10

    epsilon = -20.0
    min_logstd_of_q = -3.0

    sample_n_z = 100

    x_shape = (32, 32, 3)
    x_shape_multiple = 3072
    class_num = 10


config = ExpConfig()


class MyRNVPConfig(RealNVPConfig):
    flow_depth = 15
    strict_invertible = True
    conv_coupling_squeeze_before_first_block = True


myRNVPConfig = MyRNVPConfig()


@add_arg_scope
def batch_norm(inputs, training=False, scope=None):
    return tf.layers.batch_normalization(inputs, training=training, name=scope)


@add_arg_scope
def dropout(inputs, training=False, scope=None):
    print(inputs, training)
    return spt.layers.dropout(inputs, rate=0.2, training=training, name=scope)


@add_arg_scope
@spt.global_reuse
def p_net(glow_theta, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros(config.x_shape),
                        logstd=tf.zeros(config.x_shape))
    z = net.add('z', normal, n_samples=n_z, group_ndims=len(config.x_shape))
    _ = glow_theta.transform(z)
    x = net.add('x', spt.distributions.FlowDistribution(
        normal, glow_theta
    ), n_samples=n_z)

    return net


def get_all_loss(p_net):
    with tf.name_scope('adv_prior_loss'):
        glow_loss = -p_net['x'].log_prob() + config.x_shape_multiple * np.log(128)
    return glow_loss


@add_arg_scope
@spt.global_reuse
def resnet34(input_x):
    h_x = spt.layers.conv2d(input_x, 64, kernel_size=7, strides=2, padding='valid')
    h_x = batch_norm(h_x)
    h_x = spt.layers.max_pool2d(h_x, pool_size=3)

    normalizer_fn = None
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = spt.layers.resnet_conv2d_block(h_x, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64)

        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128)

        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256)

        h_x = spt.layers.resnet_conv2d_block(h_x, 512, strides=2)
        h_x = spt.layers.resnet_conv2d_block(h_x, 512)
        h_x = spt.layers.resnet_conv2d_block(h_x, 512)

        h_x = spt.layers.avg_pool2d(h_x, pool_size=2, strides=2)
        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        h_x = spt.layers.dense(h_x, config.class_num)  # (batch_size, class_num)
    return h_x


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
    (x_train, y_train, x_test, y_test) = load_overall(config.in_dataset)
    x_train = (x_train - 127.5) / 256.0 * 2
    x_test = (x_test - 127.5) / 256.0 * 2

    (svhn_train, _svhn_train_y, svhn_test, svhn_test_y) = load_overall(config.out_dataset)
    svhn_train = (svhn_train - 127.5) / 256.0 * 2
    svhn_test = (svhn_test - 127.5) / 256.0 * 2

    config.class_num = np.max(y_train) + 1
    config.x_shape = x_train.shape[1:]
    config.x_shape_multiple = 1
    for x in config.x_shape:
        config.x_shape_multiple *= x
    if config.x_shape == (28, 28, 1):
        config.extra_stride = 1
    config.max_epoch = config.warm_up_start + config.test_epoch_freq * config.class_num

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    input_y = tf.placeholder(
        dtype=tf.int32, shape=(None,), name='input_y')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)

    with tf.variable_scope('glow_theta'):
        glow_theta = make_real_nvp(
            rnvp_config=myRNVPConfig, is_conv=True, is_prior_flow=False, normalizer_fn=None,
            scope=tf.get_variable_scope())

    with tf.variable_scope('glow_omega'):
        glow_omega = make_real_nvp(
            rnvp_config=myRNVPConfig, is_conv=True, is_prior_flow=False, normalizer_fn=None,
            scope=tf.get_variable_scope())

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_p_net = p_net(glow_theta, observed={'x': input_x},
                            n_z=config.train_n_qz)
        glow_loss = get_all_loss(train_p_net)
        glow_loss += tf.losses.get_regularization_loss()

        predict = resnet34(input_x)
        classify_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predict, labels=input_y)
        classify_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_p_net = p_net(glow_theta, observed={'x': input_x},
                           n_z=config.test_n_qz)
        ele_test_ll = test_p_net['x'].log_prob() - config.x_shape_multiple * np.log(128)
        test_nll = -tf.reduce_mean(
            ele_test_ll
        )
        predict = tf.argmax(resnet34(input_x), axis=-1)

    # derive the optimizer
    with tf.name_scope('optimizing'):
        glow_params = tf.trainable_variables('glow_theta')
        classify_params = tf.trainable_variables('resnet34')
        with tf.variable_scope('theta_optimizer'):
            glow_optimizer = tf.train.AdamOptimizer(learning_rate)
            glow_grads = glow_optimizer.compute_gradients(glow_loss, glow_params)
            grads, vars_ = zip(*glow_grads)
            grads, gradient_norm = tf.clip_by_global_norm(grads, clip_norm=config.clip_norm)
            gradient_norm = tf.check_numerics(gradient_norm, "Gradient norm is NaN or Inf.")
            glow_grads = zip(grads, vars_)
        with tf.variable_scope('classify_optimizer'):
            classify_optimizer = tf.train.AdamOptimizer(1e-4)
            classify_grads = classify_optimizer.compute_gradients(classify_loss, classify_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            glow_train_op = glow_optimizer.apply_gradients(glow_grads)
            classify_train_op = classify_optimizer.apply_gradients(classify_grads)

    # derive the plotting function
    with tf.name_scope('plotting'):
        plot_net = p_net(glow_theta, n_z=config.sample_n_z)
        vae_plots = tf.reshape(plot_net['x'], (-1,) + config.x_shape)
        vae_plots = 256.0 * vae_plots / 2 + 127.5
        vae_plots = tf.clip_by_value(vae_plots, 0, 255)

    def plot_samples(loop, extra_index=None):
        if extra_index is None:
            extra_index = loop.epoch

        try:
            with loop.timeit('plot_time'):
                # plot samples
                images = session.run(vae_plots)
                print(images.shape)
                save_images_collection(
                    images=np.round(images),
                    filename='plotting/sample/{}-{}.png'.format('theta', extra_index),
                    grid_size=(10, 10),
                    results=results,
                )
        except Exception as e:
            print(e)

    cifar_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size)
    cifar_test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size)
    svhn_train_flow = spt.DataFlow.arrays([svhn_train], config.test_batch_size)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test], config.test_batch_size)

    train_flow = spt.DataFlow.arrays([x_train, y_train], config.batch_size, shuffle=True,
                                     skip_incomplete=True)
    mixed_array = get_mixed_array(config, x_train, x_test, svhn_train, svhn_test)
    mixed_array = mixed_array[:int(config.mixed_ratio * len(mixed_array))]
    mixed_test_flow = spt.DataFlow.arrays([mixed_array], config.batch_size,
                                          shuffle=True,
                                          skip_incomplete=True)

    reconstruct_test_flow = spt.DataFlow.arrays([x_test], 100, shuffle=True, skip_incomplete=True)
    reconstruct_train_flow = spt.DataFlow.arrays([x_train], 100, shuffle=True, skip_incomplete=True)
    reconstruct_omega_test_flow = spt.DataFlow.arrays([svhn_test], 100, shuffle=True, skip_incomplete=True)
    reconstruct_omega_train_flow = spt.DataFlow.arrays([svhn_train], 100, shuffle=True, skip_incomplete=True)


    cifar_test_predict = None

    current_class = -1
    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        restore_checkpoint = None
        restore_dir = None

        # train the network
        with spt.TrainLoop(tf.trainable_variables(),
                           var_groups=['q_net', 'p_net', 'posterior_flow', 'G_theta', 'D_psi', 'G_omega', 'D_kappa'],
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
            # adversarial training
            for epoch in epoch_iterator:

                if epoch > config.warm_up_start and cifar_test_predict is None:
                    cifar_test_predict = get_ele(predict, cifar_test_flow, input_x)
                    print('Correct number in cifar test is {}'.format(
                        np.sum(cifar_test_predict == y_test)))
                    svhn_test_predict = get_ele(predict, svhn_test_flow, input_x)
                    mixed_array_predict = get_ele(
                        predict, spt.DataFlow.arrays([mixed_array], config.batch_size), input_x)

                if epoch == config.max_epoch + 1:
                    final_cifar_test_ll = np.zeros(len(x_test))
                    final_svhn_test_ll = np.zeros(len(svhn_test))
                    if restore_dir is None:
                        restore_dir = results.system_path('checkpoint')
                    for current_class in range(0, config.class_num):
                        cifar_mask = cifar_test_predict == current_class
                        svhn_mask = svhn_test_predict == current_class
                        loop._checkpoint_saver.restore(os.path.join(
                            restore_dir, 'checkpoint', 'checkpoint.dat-{}'.format(epoch)))
                        cifar_test_ll = get_ele(ele_test_ll, spt.DataFlow.arrays([
                            x_test[cifar_mask]
                        ], config.test_batch_size), input_x)
                        svhn_test_ll = get_ele(ele_test_ll, spt.DataFlow.arrays([
                            svhn_test[svhn_mask]
                        ], config.test_batch_size), input_x)

                        final_cifar_test_ll[cifar_mask] = cifar_test_ll[cifar_mask]
                        final_svhn_test_ll[svhn_mask] = svhn_test_ll[svhn_mask]

                    plot_fig(
                        [final_cifar_test_ll, final_svhn_test_ll],
                        color_list=['red', 'green'],
                        label_list=[config.in_dataset + ' Test', config.out_dataset + ' Test'], x_label='log(bit/dims)',
                        fig_name='log_prob_histogram', auc_pair=(0, 1)
                    )
                    break

                def update_training_data():
                    train_flow = spt.DataFlow.arrays([x_train[y_train == current_class]],
                                                     config.batch_size, shuffle=True,
                                                     skip_incomplete=True)
                    mixed_test_flow = spt.DataFlow.arrays([mixed_array[mixed_array_predict == current_class]],
                                                          config.batch_size,
                                                          shuffle=True,
                                                          skip_incomplete=True)
                    return train_flow, mixed_test_flow

                if (epoch - config.warm_up_start) % config.test_epoch_freq == 1 and epoch > config.warm_up_start:
                    current_class = current_class + 1
                    session.run(tf.global_variables_initializer())  # Initialize all variables
                    train_flow, mixed_test_flow = update_training_data()

                if epoch > config.warm_up_start:
                    for step, [x] in loop.iter_steps(train_flow):
                        try:
                            _, batch_glow_loss = session.run([glow_train_op, glow_loss], feed_dict={
                                input_x: x
                            })
                            loop.collect_metrics(glow_loss=batch_glow_loss)
                        except Exception as e:
                            pass
                else:
                    for step, [x, y] in loop.iter_steps(train_flow):
                        _, batch_classify_loss = session.run([classify_train_op, classify_loss], feed_dict={
                            input_x: x, input_y: y
                        })
                        loop.collect_metrics(classify_loss=batch_classify_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.warm_up_start:
                    learning_rate.set(config.initial_lr)

                if (epoch - config.warm_up_start) % config.test_epoch_freq == 0:
                    loop._checkpoint_saver.save(epoch)

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()

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
import os


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
    max_epoch = 100
    warm_up_start = 300
    initial_beta = -3.0
    uniform_scale = False
    use_transductive = True
    mixed_train = False
    mixed_ratio1 = 0.1
    mixed_ratio2 = 0.9
    self_ood = False
    in_dataset_test_ratio = 1.0

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
def q_net(x, observed=None, n_z=None):
    net = spt.BayesianNet(observed=observed)
    normalizer_fn = batch_norm

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
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=config.extra_stride,
                                             scope='level_4')  # output: (14, 14, 32)
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

    normalizer_fn = batch_norm

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 128 * config.x_shape[0] // 4 * config.x_shape[
            1] // 4 // config.extra_stride // config.extra_stride, scope='level_0',
                               normalizer_fn=None)
        h_z = spt.ops.reshape_tail(
            h_z,
            ndims=1,
            shape=(config.x_shape[0] // 4 // config.extra_stride, config.x_shape[1] // 4 // config.extra_stride, 128)
        )
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_2')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_3')  # output: (7, 7, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=config.extra_stride,
                                               scope='level_5')  # output: (14, 14, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output:
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (28, 28, 16)
        x_mean = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_mean',
            kernel_initializer=tf.zeros_initializer(),  # activation_fn=tf.nn.tanh
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

    config.x_shape = x_train.shape[1:]
    config.x_shape_multiple = 1
    for x in config.x_shape:
        config.x_shape_multiple *= x
    if config.x_shape == (28, 28, 1):
        config.extra_stride = 1

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

        VAE_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'), \
         arg_scope([batch_norm], training=True):
        test_q_net = q_net(input_x, n_z=config.test_n_qz)
        test_chain = test_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_qz, latent_axis=0)
        test_recon = tf.reduce_mean(
            test_chain.model['x'].log_prob()
        )
        ele_test_ll = test_chain.vi.evaluation.is_loglikelihood() / config.x_shape_multiple / np.log(2)
        test_nll = -tf.reduce_mean(
            ele_test_ll
        )
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

    # derive the nll and logits output for testing
    with tf.name_scope('evaluating'), \
         arg_scope([batch_norm], training=False):
        eval_q_net = q_net(input_x, n_z=config.test_n_qz)
        eval_chain = eval_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_qz, latent_axis=0)
        eval_recon = tf.reduce_mean(
            eval_chain.model['x'].log_prob()
        )
        ele_eval_ll = eval_chain.vi.evaluation.is_loglikelihood() / config.x_shape_multiple / np.log(2)
        eval_nll = -tf.reduce_mean(
            ele_eval_ll
        )
        eval_lb = tf.reduce_mean(eval_chain.vi.lower_bound.elbo())

    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables('p_net')
        with tf.variable_scope('theta_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)

    # derive the plotting function
    with tf.name_scope('plotting'), \
         arg_scope([batch_norm], training=True):
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

                # plot samples
                images = session.run(vae_plots)
                save_images_collection(
                    images=np.round(images),
                    filename='plotting/sample/{}-{}.png'.format('theta', extra_index),
                    grid_size=(10, 10),
                    results=results,
                )
        except Exception as e:
            print(e)

    cifar_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size).map(normalize)
    cifar_test_flow = spt.DataFlow.arrays([x_test], config.test_batch_size).map(normalize)
    svhn_train_flow = spt.DataFlow.arrays([svhn_train], config.test_batch_size).map(normalize)
    svhn_test_flow = spt.DataFlow.arrays([svhn_test], config.test_batch_size).map(normalize)

    train_flow = spt.DataFlow.arrays([x_train], config.batch_size, shuffle=True, skip_incomplete=True).map(normalize)

    tmp_train_flow = spt.DataFlow.arrays([x_train], config.test_batch_size, shuffle=True, skip_incomplete=True).map(
        normalize)
    mixed_array = np.concatenate([x_test, svhn_test])
    mixed_test_flow = spt.DataFlow.arrays([mixed_array], config.batch_size, shuffle=True, skip_incomplete=True).map(
        normalize)

    reconstruct_test_flow = spt.DataFlow.arrays([x_test], 100, shuffle=True, skip_incomplete=True).map(normalize)
    reconstruct_train_flow = spt.DataFlow.arrays([x_train], 100, shuffle=True, skip_incomplete=True).map(normalize)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        experiment_dict = {
            'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/5c/f5/02732c28dc8d9673d3f5',
            'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/6c/f5/02732c28dc8d9673d3f5',
            'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/ab/16/02c52d867e439673d3f5',
            'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/4c/f5/02732c28dc8d9673d3f5',
            'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/51/26/02279d802d3ac5cad3f5',
            'constant': '/mnt/mfs/mlstorage-experiments/cwx17/94/26/02279d802d3a3dc8e3f5',
            'noise': '/mnt/mfs/mlstorage-experiments/cwx17/4d/16/02279d802d3a9673d3f5',
            'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/73/f5/02812baa4f709673d3f5',
            'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/dd/f5/02732c28dc8d7c1ad3f5',
            'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/ce/f5/02732c28dc8dedecd3f5',
            'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/41/26/02279d802d3a29bad3f5',
            'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/be/f5/02732c28dc8d58dad3f5',
            'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/5d/16/02279d802d3a9673d3f5',
            'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/63/26/02c52d867e431dc8e3f5',
        }
        print(experiment_dict)
        if config.in_dataset in experiment_dict:
            restore_dir = experiment_dict[config.in_dataset] + '/checkpoint'
            restore_checkpoint = os.path.join(
                restore_dir, 'checkpoint', 'checkpoint.dat-{}'.format(config.max_epoch))
        else:
            restore_dir = results.system_path('checkpoint')
            restore_checkpoint = None

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
                           checkpoint_epoch_freq=100,
                           restore_checkpoint=restore_checkpoint
                           ) as loop:

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()
            # adversarial training
            for epoch in epoch_iterator:

                if epoch == config.max_epoch + 1:
                    def permutation_test(flow, ratio):
                        R = min(max(1, int(ratio * config.test_batch_size - 1)), config.test_batch_size - 1)
                        print('R={}'.format(R))
                        packs = []
                        for [batch_x] in flow:
                            for i in range(len(batch_x)):
                                for [batch_y] in mixed_test_flow:
                                    for [batch_z] in tmp_train_flow:
                                        batch = np.concatenate(
                                            [batch_x[i:i + 1], batch_y[:R], batch_z[:config.test_batch_size - R - 1]],
                                            axis=0)
                                        pack = session.run(
                                            ele_test_ll, feed_dict={
                                                input_x: batch,
                                            })  # [batch_size]
                                        pack = np.asarray(pack)[:1]
                                        break
                                    break
                                packs.append(pack)
                        packs = np.concatenate(packs, axis=0)  # [len_of_flow]
                        print(packs.shape)
                        return packs

                    def delta_test(flow):
                        return permutation_test(flow, config.mixed_ratio1) - permutation_test(flow, config.mixed_ratio2)

                    cifar_r1 = permutation_test(cifar_test_flow, config.mixed_ratio1)
                    cifar_r2 = permutation_test(cifar_test_flow, config.mixed_ratio2)
                    svhn_r1 = permutation_test(svhn_test_flow, config.mixed_ratio1)
                    svhn_r2 = permutation_test(svhn_test_flow, config.mixed_ratio2)

                    loop.collect_metrics(r1_histogram=plot_fig([cifar_r1, cifar_r2, svhn_r1, svhn_r2],
                                                               ['red', 'salmon', 'green', 'lightgreen'],
                                                               [config.in_dataset + ' r1', config.in_dataset + ' r2',
                                                                config.out_dataset + ' r1', config.out_dataset + ' r2'],
                                                               'log(bit/dims)',
                                                               'r1_histogram', auc_pair=(0, 2)))

                    loop.collect_metrics(r2_histogram=plot_fig([cifar_r1, cifar_r2, svhn_r1, svhn_r2],
                                                               ['red', 'salmon', 'green', 'lightgreen'],
                                                               [config.in_dataset + ' r1', config.in_dataset + ' r2',
                                                                config.out_dataset + ' r1', config.out_dataset + ' r2'],
                                                               'log(bit/dims)',
                                                               'r2_histogram', auc_pair=(1, 3)))

                    loop.collect_metrics(r1_r2_histogram=plot_fig(
                        [cifar_r1 - cifar_r2, svhn_r1 - svhn_r2],
                        ['red', 'green'],
                        [config.in_dataset + ' test',
                         config.out_dataset + ' test'], 'log(bit/dims)',
                        'r1_r2_log_pro_histogram',
                        auc_pair=(0, 1)))

                    make_diagram(loop,
                                 ele_test_ll,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='log_prob_with_batch_norm_histogram'
                                 )

                    make_diagram(loop,
                                 ele_eval_ll,
                                 [cifar_test_flow, svhn_test_flow], input_x,
                                 names=[config.in_dataset + ' Test', config.out_dataset + ' Test'],
                                 fig_name='log_prob_without_batch_norm_histogram'
                                 )

                for step, [x] in loop.iter_steps(train_flow):
                    _, batch_VAE_loss = session.run([VAE_train_op, VAE_loss], feed_dict={
                        input_x: x
                    })
                    loop.collect_metrics(VAE_loss=batch_VAE_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.max_epoch:
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
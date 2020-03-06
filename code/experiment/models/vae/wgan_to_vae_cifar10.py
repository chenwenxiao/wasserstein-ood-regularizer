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

from code.experiment.datasets.svhn import load_svhn
from code.experiment.models.gradient_penalty import get_gradient_penalty, spectral_norm
from code.experiment.utils import make_diagram


class ExpConfig(spt.Config):
    # model parameters
    z_dim = 256
    act_norm = False
    weight_norm = False
    l2_reg = 0.0002
    kernel_size = 3
    shortcut_kernel_size = 1
    batch_norm = False
    nf_layers = 20

    # training parameters
    result_dir = None
    write_summary = True
    max_epoch = 1600
    warm_up_start = 800
    warm_up_epoch = 800
    beta = 1e-8
    initial_xi = 0.0
    uniform_scale = True

    max_step = None
    batch_size = 128
    initial_lr = 0.0001
    lr_anneal_factor = 0.5
    lr_anneal_epoch_freq = [200, 400, 600, 800, 1000, 1200, 1400, 1600]
    lr_anneal_step_freq = None

    gradient_penalty_algorithm = 'interpolate'  # both or interpolate
    gradient_penalty_weight = 2
    gradient_penalty_index = 6
    wasserstein_regularizer_weight = 1.0

    n_critical = 5
    # evaluation parameters
    train_n_pz = 128
    train_n_qz = 1
    test_n_pz = 1000
    test_n_qz = 10
    test_batch_size = 64
    test_epoch_freq = 100
    plot_epoch_freq = 10

    sample_n_z = 100

    epsilon = -20.0
    min_logstd_of_q = -3.0

    @property
    def x_shape(self):
        return (32, 32, 3)


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
def q_net(x, posterior_flow, observed=None, n_z=None):
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
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (32, 32, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_1')  # output: (32, 32, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_2')  # output: (32, 32, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_5')  # output: (16, 16, 256)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_7')  # output: (8, 8, 256)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_8')  # output: (4, 4, 256)
        h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
        z_mean = spt.layers.dense(h_x, config.z_dim)
        z_logstd = spt.layers.dense(h_x, config.z_dim)

    # sample z ~ q(z|x)
    z = net.add('z', spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.min_logstd_of_q)),
                n_samples=n_z, group_ndims=1)

    return net


@add_arg_scope
@spt.global_reuse
def G_theta(z, return_std=False):
    normalizer_fn = None

    # compute the hidden features
    with arg_scope([spt.layers.resnet_deconv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_z = spt.layers.dense(z, 256 * config.x_shape[0] // 8 * config.x_shape[1] // 8, normalizer_fn=None)
        h_z = spt.ops.reshape_tail(h_z, ndims=1, shape=(config.x_shape[0] // 8, config.x_shape[1] // 8, 256))
        h_z = spt.layers.resnet_deconv2d_block(h_z, 256, strides=2, scope='level_2')  # output: (8, 8, 256)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 256, strides=2, scope='level_3')  # output: (16, 16, 256)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 128, strides=2, scope='level_5')  # output: (32, 32, 128)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 64, scope='level_6')  # output: (32, 32, 64)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 32, scope='level_7')  # output: (32, 32, 32)
        h_z = spt.layers.resnet_deconv2d_block(h_z, 16, scope='level_8')  # output: (32, 32, 16)
        x_mean = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_mean',
            kernel_initializer=tf.zeros_initializer(), activation_fn=tf.nn.tanh
        )
        x_logstd = spt.layers.conv2d(
            h_z, config.x_shape[-1], (1, 1), padding='same', scope='x_logstd',
            kernel_initializer=tf.zeros_initializer(),
        )
    if return_std:
        return x_mean, x_logstd
    else:
        return x_mean


@add_arg_scope
@spt.global_reuse
def D_psi(x, y=None):
    # if y is not None:
    #     return D_psi(y) + 0.1 * tf.sqrt(tf.reduce_sum((x - y) ** 2, axis=tf.range(-len(config.x_shape), 0)))
    normalizer_fn = None
    # x = tf.round(256.0 * x / 2 + 127.5)
    # x = (x - 127.5) / 256.0 * 2
    with arg_scope([spt.layers.resnet_conv2d_block],
                   kernel_size=config.kernel_size,
                   shortcut_kernel_size=config.shortcut_kernel_size,
                   activation_fn=tf.nn.leaky_relu,
                   normalizer_fn=normalizer_fn,
                   weight_norm=spectral_norm if config.gradient_penalty_algorithm == 'spectral' else None,
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg)):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (32, 32, 16)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (32, 32, 32)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (32, 32, 64)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_4')  # output: (16, 16, 128)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_6')  # output: (8, 8, 256)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_8')  # output: (4, 4, 256)

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
    x_mean, x_logstd = G_theta(z, return_std=True)
    x = net.add('x', DiscretizedLogistic(
        mean=x_mean,
        log_scale=spt.ops.maybe_clip_value(beta if config.uniform_scale else x_logstd, min_val=config.epsilon),
        bin_size=2.0 / 256.0,
        min_val=-1.0 + 1.0 / 256.0,
        max_val=1.0 - 1.0 / 256.0,
        epsilon=1e-10
    ), group_ndims=3)
    return net


def get_all_loss(q_net, p_net, pn_theta):
    with tf.name_scope('adv_prior_loss'):
        log_px_z = p_net['x'].log_prob()
        global train_recon
        global train_kl
        global train_grad_penalty
        train_recon = tf.reduce_mean(log_px_z)
        train_kl = tf.reduce_mean(
            -p_net['z'].log_prob() + q_net['z'].log_prob()
        )
        VAE_nD_loss = -train_recon + train_kl

        sample_x = pn_theta['x'].distribution.mean
        gp = get_gradient_penalty(p_net['x'], sample_x, D_psi, config.batch_size, config.x_shape)
        energy_fake = D_psi(sample_x)
        energy_real = D_psi(p_net['x'])

        adv_G_loss = tf.reduce_mean(energy_fake)
        adv_D_real = tf.reduce_mean(energy_real)
        adv_D_loss = -adv_G_loss + adv_D_real + gp
        VAE_loss = VAE_nD_loss + config.wasserstein_regularizer_weight * (adv_G_loss - adv_D_real)
    return VAE_loss, VAE_nD_loss, adv_D_loss, adv_G_loss, adv_D_real


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


train_recon = None
train_kl = None
train_grad_penalty = None


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

    posterior_flow = spt.layers.planar_normalizing_flows(
        config.nf_layers, name='posterior_flow')

    # input placeholders
    input_x = tf.placeholder(
        dtype=tf.float32, shape=(None,) + config.x_shape, name='input_x')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    beta = tf.Variable(initial_value=-3.0, dtype=tf.float32, name='beta', trainable=True)

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_pn_theta = p_net(n_z=config.train_n_pz, beta=beta)
        train_q_net = q_net(input_x, posterior_flow, n_z=config.train_n_qz)
        train_p_net = p_net(observed={'x': input_x, 'z': train_q_net['z']},
                            n_z=config.train_n_qz, beta=beta)

        VAE_loss, VAE_nD_loss, D_loss, G_loss, D_real = get_all_loss(train_q_net, train_p_net, train_pn_theta)
        VAE_loss += tf.losses.get_regularization_loss()
        VAE_nD_loss += tf.losses.get_regularization_loss()
        D_loss += tf.losses.get_regularization_loss()
        G_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, posterior_flow, n_z=config.test_n_qz)
        test_pn_net = p_net(n_z=config.test_n_pz, beta=beta)
        test_chain = test_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_qz, latent_axis=0,
                                      beta=beta)
        test_recon = tf.reduce_mean(test_chain.model['x'].log_prob())
        test_ele_nll = test_chain.vi.evaluation.is_loglikelihood()
        test_nll = -tf.reduce_mean(test_ele_nll)
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())
    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables(
            'posterior_flow') + tf.trainable_variables('beta')
        D_params = tf.trainable_variables('D_psi')
        G_params = tf.trainable_variables('G_theta')
        with tf.variable_scope('VAE_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)
        with tf.variable_scope('VAE_nD_optimizer'):
            VAE_nD_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_nD_grads = VAE_nD_optimizer.compute_gradients(VAE_nD_loss, VAE_params)
        with tf.variable_scope('D_optimizer'):
            D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            D_grads = D_optimizer.compute_gradients(D_loss, D_params)
        with tf.variable_scope('G_optimizer'):
            G_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            G_grads = G_optimizer.compute_gradients(G_loss, G_params)

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)
            VAE_nD_train_op = VAE_optimizer.apply_gradients(VAE_nD_grads)
            D_train_op = D_optimizer.apply_gradients(D_grads)
            G_train_op = G_optimizer.apply_gradients(G_grads)

        # derive the plotting function
        with tf.name_scope('plotting'):
            sample_n_z = config.sample_n_z
            plot_net = p_net(n_z=sample_n_z, beta=beta)
            x_plots = 256.0 * tf.reshape(
                plot_net['x'].distribution.mean, (-1,) + config.x_shape) / 2 + 127.5
            reconstruct_q_net = q_net(input_x, posterior_flow)
            reconstruct_z = reconstruct_q_net['z']
            reconstruct_plots = 256.0 * tf.reshape(
                p_net(observed={'z': reconstruct_z}, beta=beta)['x'].distribution.mean,
                (-1,) + config.x_shape
            ) / 2 + 127.5
            x_plots = tf.clip_by_value(x_plots, 0, 255)
            reconstruct_plots = tf.clip_by_value(reconstruct_plots, 0, 255)

        def plot_samples(loop, extra_index=None):
            if extra_index is None:
                extra_index = loop.epoch
            with loop.timeit('plot_time'):
                # plot reconstructs
                for [x] in reconstruct_test_flow:
                    x_samples = x
                    images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                    images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                    batch_reconstruct_plots, batch_reconstruct_z = session.run(
                        [reconstruct_plots, reconstruct_z], feed_dict={input_x: x_samples})
                    images[2::3, ...] = np.round(batch_reconstruct_plots)
                    # print(np.mean(batch_reconstruct_z ** 2, axis=-1))
                    save_images_collection(
                        images=images,
                        filename='plotting/test.reconstruct/{}.png'.format(extra_index),
                        grid_size=(20, 15),
                        results=results,
                    )
                    break

                for [x] in reconstruct_train_flow:
                    x_samples = x
                    images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                    images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                    batch_reconstruct_plots, batch_reconstruct_z = session.run(
                        [reconstruct_plots, reconstruct_z], feed_dict={input_x: x_samples})
                    images[2::3, ...] = np.round(batch_reconstruct_plots)
                    # print(np.mean(batch_reconstruct_z ** 2, axis=-1))
                    save_images_collection(
                        images=images,
                        filename='plotting/train.reconstruct/{}.png'.format(extra_index),
                        grid_size=(20, 15),
                        results=results,
                    )
                    break

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
    reconstruct_train_flow = spt.DataFlow.arrays([x_train], 100, shuffle=True, skip_incomplete=False)
    reconstruct_test_flow = spt.DataFlow.arrays([x_test], 100, shuffle=True, skip_incomplete=False)

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

            evaluator = spt.Evaluator(
                loop,
                metrics={'test_nll': test_nll, 'test_lb': test_lb,
                         'test_recon': test_recon},
                inputs=[input_x],
                data_flow=cifar_test_flow,
                time_metric_name='test_time'
            )

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()

            n_critical = config.n_critical
            # adversarial training
            for epoch in epoch_iterator:
                if epoch > config.warm_up_start:
                    step_iterator = MyIterator(train_flow)
                    while step_iterator.has_next:
                        for step, [x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            # vae training
                            [_, batch_VAE_loss, beta_value, batch_train_recon, batch_train_kl,
                             batch_train_grad_penalty] = session.run(
                                [VAE_nD_train_op, VAE_nD_loss, beta, train_recon, train_kl,
                                 train_grad_penalty],
                                feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(batch_VAE_loss=batch_VAE_loss)
                            loop.collect_metrics(beta=beta_value)
                            loop.collect_metrics(train_kl=batch_train_kl)
                            loop.collect_metrics(train_recon=batch_train_recon)
                            loop.collect_metrics(train_grad_penalty=batch_train_grad_penalty)
                else:
                    step_iterator = MyIterator(train_flow)
                    while step_iterator.has_next:
                        for step, [x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            # discriminator training
                            [_, batch_D_loss, batch_D_real, beta_value] = session.run(
                                [D_train_op, D_loss, D_real, beta], feed_dict={
                                    input_x: x
                                })
                            loop.collect_metrics(D_loss=batch_D_loss)
                            loop.collect_metrics(D_real=batch_D_real)
                            loop.collect_metrics(beta=beta_value)

                        # generator training x
                        [_, batch_G_loss] = session.run(
                            [G_train_op, G_loss], feed_dict={
                            })
                        loop.collect_metrics(G_loss=batch_G_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.warm_up_start:
                    learning_rate.set(config.initial_lr)

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                if epoch % config.test_epoch_freq == 0:
                    with loop.timeit('eval_time'):
                        evaluator.run()
                    make_diagram(
                        test_ele_nll,
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

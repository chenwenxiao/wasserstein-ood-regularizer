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
from ood_regularizer.experiments.utils import get_inception_score, get_fid
import numpy as np
from scipy.misc import logsumexp

from tfsnippet.preprocessing import UniformNoiseSampler


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


class EnergyDistribution(spt.Distribution):
    """
    A distribution derived from an energy function `D(x)` and a generator
    function `x = G(z)`, where `p(z) = exp(-xi * D(G(z)) - 0.5 * z^2) / Z`.
    """

    def __init__(self, pz, G, D, log_Z=0., xi=1.0, mcmc_iterator=0, mcmc_alpha=0.001, mcmc_algorithm='mala',
                 mcmc_space='z', initial_z=None):
        """
        Construct a new :class:`EnergyDistribution`.

        Args:
            pz (spt.Distribution): The base distribution `p(z)`.
            G: The function `x = G(z)`.
            D: The function `D(x)`.
            Z: The partition factor `Z`.
            xi: The weight of energy.
        """
        if not pz.is_continuous:
            raise TypeError('`base_distribution` must be a continuous '
                            'distribution.')

        super(EnergyDistribution, self).__init__(
            dtype=pz.dtype,
            is_continuous=True,
            is_reparameterized=pz.is_reparameterized,
            batch_shape=pz.batch_shape,
            batch_static_shape=pz.get_batch_shape(),
            value_ndims=pz.value_ndims
        )
        log_Z = spt.ops.convert_to_tensor_and_cast(log_Z, dtype=pz.dtype)

        self._pz = pz
        self._G = G
        self._D = D
        self._xi = xi
        with tf.name_scope('log_Z', values=[log_Z]):
            self._log_Z = tf.maximum(log_Z, -20)
        self._mcmc_iterator = mcmc_iterator
        self._mcmc_alpha = mcmc_alpha
        self._initial_z = initial_z

    @property
    def pz(self):
        return self._pz

    @property
    def G(self):
        return self._G

    @property
    def D(self):
        return self._D

    @property
    def xi(self):
        return self._xi

    @property
    def log_Z(self):
        return self._log_Z

    def log_prob(self, given, group_ndims=0, name=None, y=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(name,
                           default_name=spt.utils.get_default_scope_name(
                               'log_prob', self),
                           values=[given]):
            if y is not None:
                energy = config.pull_back_energy_weight * self.D(self.G(given), y) * self.xi + 0.5 * tf.reduce_sum(
                    tf.square(given), axis=-1)
            else:
                energy = config.pull_back_energy_weight * self.D(self.G(given)) * self.xi + 0.5 * tf.reduce_sum(
                    tf.square(given), axis=-1)
            log_px = self.pz.log_prob(given=given, group_ndims=group_ndims)
            log_px.log_energy_prob = -energy - self.log_Z
            log_px.energy = energy

        return log_px

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        self._validate_sample_is_reparameterized_arg(is_reparameterized)
        if is_reparameterized is None:
            is_reparameterized = self.is_reparameterized

        with tf.name_scope(name, default_name=spt.utils.get_default_scope_name('sample', self)):
            origin_z = self.pz.sample(
                n_samples=n_samples, is_reparameterized=is_reparameterized,
                compute_density=False
            )
            z = origin_z
            if self._initial_z is not None:
                z = self._initial_z
            history_z = []
            history_e_z = []
            history_pure_e_z = []
            history_ratio = []
            for i in range(self._mcmc_iterator):
                e_z, grad_e_z, z_prime, pure_energy_z = self.get_sgld_proposal(z)
                e_z_prime, grad_e_z_prime, _, __ = self.get_sgld_proposal(z_prime)
                history_e_z.append(e_z)
                history_z.append(z)
                history_pure_e_z.append(pure_energy_z)
                log_q_zprime_z = tf.reduce_sum(
                    tf.square(z_prime - z + self._mcmc_alpha * grad_e_z), axis=-1
                )
                log_q_zprime_z *= -1. / (4 * self._mcmc_alpha)

                log_q_z_zprime = tf.reduce_sum(
                    tf.square(z - z_prime + self._mcmc_alpha * grad_e_z_prime), axis=-1
                )
                log_q_z_zprime *= -1. / (4 * self._mcmc_alpha)

                log_ratio_1 = -e_z_prime + e_z  # log [p(z_prime) / p(z)]
                log_ratio_2 = log_q_z_zprime - log_q_zprime_z  # log [q(z | z_prime) / q(z_prime | z)]
                # print(log_ratio_1.mean().item(), log_ratio_2.mean().item())
                ratio = tf.clip_by_value(
                    tf.exp(log_ratio_1 + log_ratio_2), 0.0, 1.0
                )
                history_ratio.append(ratio)
                # print(ratio.mean().item())
                rnd_u = tf.random.uniform(
                    shape=ratio.shape
                )
                mask = tf.cast(tf.less(rnd_u, ratio), tf.float32)
                mask = tf.expand_dims(mask, axis=-1)
                z = (z_prime * mask + z * (1 - mask))

            t = spt.StochasticTensor(
                distribution=self,
                tensor=z,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized
            )
            t.history_z = tf.stack(history_z)
            t.history_e_z = tf.stack(history_e_z)
            t.history_pure_e_z = tf.stack(history_pure_e_z)
            t.history_ratio = tf.stack(history_ratio)
        return t

    def get_sgld_proposal(self, z):
        energy_z = config.pull_back_energy_weight * self.D(self.G(z)) * self.xi + 0.5 * tf.reduce_sum(tf.square(z),
                                                                                                      axis=-1)
        pure_energy_z = self.D(self.G(z))
        # energy_z = pure_energy_z  # TODO
        grad_energy_z = tf.gradients(energy_z, [z.tensor if hasattr(z, 'tensor') else z])[0]
        grad_energy_z = tf.reshape(grad_energy_z, shape=z.shape)
        eps = tf.random.normal(
            shape=z.shape
        ) * np.sqrt(self._mcmc_alpha * 2)
        z_prime = z - self._mcmc_alpha * grad_energy_z + eps
        return energy_z, grad_energy_z, z_prime, pure_energy_z


class ExponentialDistribution(spt.Distribution):
    """
    A distribution in R^N, derived from an a mean `\mu`,
    where `p(x) = exp(-\|x - \mu\| / \beta) / Z(\beta)`,
    Z(\beta) = 2 * \pi^(N/2) * \Gamma(N) / \Gamma(N/2) * \beta^N.
    """

    def __init__(self, mean, beta, D):
        """
        Construct a new :class:`EnergyDistribution`.

        Args:
            mean: The mean of ExponentialDistribution.
            beta: The beta of ExponentialDistribution.
        """
        beta = spt.ops.convert_to_tensor_and_cast(beta, dtype=tf.float32)
        _ = spt.Normal(mean, logstd=beta)
        super(ExponentialDistribution, self).__init__(
            dtype=tf.float32,
            is_continuous=True,
            is_reparameterized=True,
            batch_shape=_.batch_shape,
            batch_static_shape=_.get_batch_shape(),
            value_ndims=_.value_ndims
        )
        self._beta = beta
        self._mean = mean
        self._D = D
        self._N = 1
        for i in config.x_shape:
            self._N *= i
        self._log_Z = np.log(2) + self.N / 2.0 * np.log(np.pi) + self.N * tf.log(self.beta)
        bias = 0.0
        for i in range(1, self.N):
            bias += np.log(i)
        for i in range(self.N - 2, 0, -2):
            bias -= np.log(i / 2.0)
        if self.N % 2 == 1:
            bias -= 0.5 * np.log(np.pi)
        self._log_Z += bias

    @property
    def beta(self):
        return self._beta

    @property
    def mean(self):
        return self._mean

    @property
    def D(self):
        return self._D

    @property
    def N(self):
        return self._N

    @property
    def log_Z(self):
        return self._log_Z

    def log_prob(self, given, group_ndims=0, name=None):
        given = tf.convert_to_tensor(given)
        with tf.name_scope(name,
                           default_name=spt.utils.get_default_scope_name('log_prob', self),
                           values=[given]):
            log_px = -tf.sqrt(
                tf.reduce_sum(tf.square(given - self.mean), axis=tf.range(-group_ndims, 0))) / self.beta - self.log_Z
            log_px.energy = self.D(given)
            log_px.mean_energy = self.D(self.mean)

        return log_px

    def sample(self, n_samples=None, group_ndims=0, is_reparameterized=None,
               compute_density=None, name=None):
        self._validate_sample_is_reparameterized_arg(is_reparameterized)
        if is_reparameterized is None:
            is_reparameterized = self.is_reparameterized

        with tf.name_scope(name, default_name=spt.utils.get_default_scope_name('sample', self)):
            t = spt.StochasticTensor(
                distribution=self,
                tensor=self.mean,
                n_samples=n_samples,
                group_ndims=group_ndims,
                is_reparameterized=is_reparameterized
            )
        return t


@add_arg_scope
def batch_norm(inputs, training=False, scope=None):
    return tf.layers.batch_normalization(inputs, training=training, name=scope)


@add_arg_scope
def dropout(inputs, training=False, scope=None):
    print(inputs, training)
    return spt.layers.dropout(inputs, rate=0.2, training=training, name=scope)


def get_z_moments(z, value_ndims, name=None):
    """
    Get the per-dimensional mean and variance of `z`.

    Args:
        z (Tensor): The `z` tensor.
        value_ndims (int): Number of value dimensions, must be `1` or `3`.
            `1` indicates `z` is a dense latent variable, otherwise `3`
            indicates `z` is a convolutional latent variable.

    Returns:
        (tf.Tensor, tf.Tensor): The `(mean, variance)` of `z`.
    """
    value_ndims = spt.utils.validate_enum_arg(
        'value_ndims', value_ndims, [1, 3])

    if value_ndims == 1:
        z = spt.utils.InputSpec(shape=['...', '*']).validate('z', z)
    else:
        z = spt.utils.InputSpec(shape=['...', '?', '?', '*']).validate('z', z)

    with tf.name_scope(name, default_name='get_z_moments', values=[z]):
        rank = len(spt.utils.get_static_shape(z))
        if value_ndims == 1:
            axes = list(range(0, rank - 1))
        else:
            axes = list(range(0, rank - 3))
        mean, variance = tf.nn.moments(z, axes=axes)
        return mean, variance


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
                   kernel_regularizer=spt.layers.l2_regularizer(config.l2_reg),
                   dropout_fn=dropout):
        h_x = tf.to_float(x)
        h_x = spt.layers.resnet_conv2d_block(h_x, 16, scope='level_0')  # output: (28, 28, 16)
        h_x = tf.concat([h_x, x], axis=-1)
        h_x = spt.layers.resnet_conv2d_block(h_x, 32, scope='level_2')  # output: (14, 14, 32)
        h_x = tf.concat([h_x, x], axis=-1)
        h_x = spt.layers.resnet_conv2d_block(h_x, 64, scope='level_3')  # output: (14, 14, 32)
        h_x = tf.concat([h_x, x], axis=-1)
        h_x = spt.layers.resnet_conv2d_block(h_x, 128, strides=2, scope='level_4')  # output: (14, 14, 32)
        x = spt.ops.reshape_tail(x, ndims=3,
                                 shape=[config.x_shape[0] // 2, config.x_shape[1] // 2, config.x_shape[2] * 4])
        h_x = tf.concat([h_x, x], axis=-1)
        h_x = spt.layers.resnet_conv2d_block(h_x, 256, strides=2, scope='level_6')  # output: (7, 7, 64)
        x = spt.ops.reshape_tail(x, ndims=3,
                                 shape=[config.x_shape[0] // 4, config.x_shape[1] // 4, config.x_shape[2] * 16])
        h_x = tf.concat([h_x, x], axis=-1)
        h_x = spt.layers.resnet_conv2d_block(h_x, 512, strides=2, scope='level_8')  # output: (7, 7, 64)
        x = spt.ops.reshape_tail(x, ndims=3,
                                 shape=[config.x_shape[0] // 8, config.x_shape[1] // 8, config.x_shape[2] * 64])
        h_x = tf.concat([h_x, x], axis=-1)

    # sample z ~ q(z|x)
    h_x = spt.ops.reshape_tail(h_x, ndims=3, shape=[-1])
    # x = spt.ops.reshape_tail(x, ndims=3, shape=[-1])
    # h_x = tf.concat([h_x, x], axis=-1)
    z_mean = spt.layers.dense(h_x, config.z_dim, scope='z_mean', kernel_initializer=tf.zeros_initializer())
    z_logstd = spt.layers.dense(h_x, config.z_dim, scope='z_logstd', kernel_initializer=tf.zeros_initializer())
    z_distribution = spt.FlowDistribution(
        spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.epsilon)),
        posterior_flow
    )
    z = net.add('z', z_distribution, n_samples=n_z)
    # z = net.add('z', spt.Normal(mean=z_mean, logstd=spt.ops.maybe_clip_value(z_logstd, min_val=config.epsilon)),
    #             n_samples=n_z, group_ndims=1)

    return net


__log_Z = None


@spt.global_reuse
def get_log_Z():
    global __log_Z
    if __log_Z is None:
        __log_Z = spt.ScheduledVariable('log_Z', dtype=tf.float32, initial_value=1., model_var=True)
    return __log_Z




@add_arg_scope
@spt.global_reuse
def G_theta(z):
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
def p_net(observed=None, n_z=None, beta=1.0, mcmc_iterator=0, log_Z=0.0, initial_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, config.z_dim]),
                        logstd=tf.zeros([1, config.z_dim]))
    normal = normal.batch_ndims_to_value(1)
    xi = tf.get_variable(name='xi', shape=(), initializer=tf.constant_initializer(config.initial_xi),
                         dtype=tf.float32, trainable=True)
    # xi = tf.square(xi)
    xi = tf.nn.sigmoid(xi)  # TODO
    pz = EnergyDistribution(normal, G=G_theta, D=D_psi, log_Z=log_Z, xi=xi, mcmc_iterator=mcmc_iterator,
                            initial_z=initial_z)
    z = net.add('z', pz, n_samples=n_z)
    x_mean = G_theta(z)
    x = net.add('x', ExponentialDistribution(
        mean=x_mean,
        beta=beta,
        D=D_psi
    ), group_ndims=3)
    return net


@add_arg_scope
@spt.global_reuse
def p_omega_net(observed=None, n_z=None, beta=1.0, mcmc_iterator=0, log_Z=0.0, initial_z=None):
    net = spt.BayesianNet(observed=observed)
    # sample z ~ p(z)
    normal = spt.Normal(mean=tf.zeros([1, 256]),
                        logstd=tf.zeros([1, 256]))
    normal = normal.batch_ndims_to_value(1)
    xi = tf.get_variable(name='xi', shape=(), initializer=tf.constant_initializer(config.initial_xi),
                         dtype=tf.float32, trainable=True)
    # xi = tf.square(xi)
    xi = tf.nn.sigmoid(xi)  # TODO
    pz = EnergyDistribution(normal, G=G_omega, D=D_psi, log_Z=log_Z, xi=xi, mcmc_iterator=mcmc_iterator,
                            initial_z=initial_z)
    z = net.add('z', pz, n_samples=n_z)
    x_mean = G_omega(z)
    x = net.add('x', ExponentialDistribution(
        mean=x_mean,
        beta=beta,
        D=D_psi
    ), group_ndims=3)
    return net


def get_all_loss(q_net, p_net, pn_net, warm=1.0):
    with tf.name_scope('adv_prior_loss'):
        x = p_net['x']
        x_ = pn_net['x'].distribution.mean
        log_px_z = x.log_prob()
        energy_real = D_psi(x)
        energy_fake = D_psi(x_)

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

        # VAE_loss = tf.reduce_mean(
        #     -log_px_z - p_net['z'].log_prob() + q_net['z'].log_prob()
        # )
        global train_recon
        train_recon = tf.reduce_mean(log_px_z)
        global train_recon_energy
        global train_kl
        global train_grad_penalty
        train_recon_energy = tf.reduce_mean(D_psi(p_net['x'].distribution.mean, p_net['x']))
        train_kl = tf.reduce_mean(
            -p_net['z'].distribution.log_prob(p_net['z'], group_ndims=1, y=p_net['x']).log_energy_prob +
            q_net['z'].log_prob()
        )
        train_grad_penalty = gradient_penalty
        train_kl = tf.maximum(train_kl, 0.0)  # TODO
        VAE_loss = -train_recon + warm * train_kl  # + gradient_penalty * 128.0
        adv_D_loss = -tf.reduce_mean(energy_fake) + tf.reduce_mean(
            energy_real) + gradient_penalty
        adv_G_loss = tf.reduce_mean(energy_fake)
    return VAE_loss, adv_D_loss, adv_G_loss, tf.reduce_mean(energy_real)


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
train_recon_energy = None
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
    warm = tf.placeholder(
        dtype=tf.float32, shape=(), name='warm')
    learning_rate = spt.AnnealingVariable(
        'learning_rate', config.initial_lr, config.lr_anneal_factor)
    beta = tf.Variable(initial_value=0.0, dtype=tf.float32, name='beta', trainable=True)

    # derive the loss for initializing
    with tf.name_scope('initialization'), \
         arg_scope([spt.layers.act_norm], initializing=True), \
         spt.utils.scoped_set_config(spt.settings, auto_histogram=False):
        init_pn_net = p_net(n_z=config.train_n_pz, beta=beta)
        init_q_net = q_net(input_x, posterior_flow, n_z=config.train_n_qz)
        init_p_net = p_net(observed={'x': input_x, 'z': init_q_net['z']}, n_z=config.train_n_qz, beta=beta)
        init_loss = sum(get_all_loss(init_q_net, init_p_net, init_pn_net))

    # derive the loss and lower-bound for training
    with tf.name_scope('training'), \
         arg_scope([batch_norm], training=True):
        train_pn_theta = p_net(n_z=config.train_n_pz, beta=beta)
        train_pn_omega = p_omega_net(n_z=config.train_n_pz, beta=beta)
        train_log_Z = spt.ops.log_mean_exp(-train_pn_theta['z'].log_prob().energy - train_pn_theta['z'].log_prob())
        train_q_net = q_net(input_x, posterior_flow, n_z=config.train_n_qz)
        train_p_net = p_net(observed={'x': input_x, 'z': train_q_net['z']},
                            n_z=config.train_n_qz, beta=beta, log_Z=train_log_Z)

        VAE_loss, _, VAE_G_loss, VAE_D_real = get_all_loss(train_q_net, train_p_net, train_pn_theta, warm)
        _, D_loss, G_loss, D_real = get_all_loss(train_q_net, train_p_net, train_pn_omega, warm)

        VAE_loss += tf.losses.get_regularization_loss()
        VAE_G_loss += tf.losses.get_regularization_loss()
        D_loss += tf.losses.get_regularization_loss()
        G_loss += tf.losses.get_regularization_loss()

    # derive the nll and logits output for testing
    with tf.name_scope('testing'):
        test_q_net = q_net(input_x, posterior_flow, n_z=config.test_n_qz)
        # test_pd_net = p_net(n_z=config.test_n_pz // 20, mcmc_iterator=20, beta=beta, log_Z=get_log_Z())
        test_pn_net = p_net(n_z=config.test_n_pz, mcmc_iterator=0, beta=beta, log_Z=get_log_Z())
        test_chain = test_q_net.chain(p_net, observed={'x': input_x}, n_z=config.test_n_qz, latent_axis=0,
                                      beta=beta, log_Z=get_log_Z())
        test_recon = tf.reduce_mean(
            test_chain.model['x'].log_prob()
        )
        test_mse = tf.reduce_sum(
            (tf.round(test_chain.model['x'].distribution.mean * 128 + 127.5) - tf.round(
                test_chain.model['x'] * 128 + 127.5)) ** 2, axis=[-1, -2, -3])  # (sample_dim, batch_dim, x_sample_dim)
        test_mse = tf.reduce_min(test_mse, axis=[0])
        test_mse = tf.reduce_mean(tf.reduce_mean(tf.reshape(
            test_mse, (-1, config.test_x_samples,)
        ), axis=-1))
        test_nll = -tf.reduce_mean(
            spt.ops.log_mean_exp(
                tf.reshape(
                    test_chain.vi.evaluation.is_loglikelihood(), (-1, config.test_x_samples,)
                ), axis=-1)
        ) + config.x_shape_multiple * np.log(128.0)
        test_lb = tf.reduce_mean(test_chain.vi.lower_bound.elbo())

        vi = spt.VariationalInference(
            log_joint=test_chain.model['x'].log_prob() + test_chain.model['z'].distribution.log_prob(
                test_chain.model['z'], group_ndims=1, y=test_chain.model['x']
            ).log_energy_prob,
            latent_log_probs=[test_q_net['z'].log_prob()],
            axis=0
        )
        adv_test_nll = -tf.reduce_mean(
            spt.ops.log_mean_exp(
                tf.reshape(
                    vi.evaluation.is_loglikelihood(), (-1, config.test_x_samples,)
                ), axis=-1)
        ) + config.x_shape_multiple * np.log(128.0)
        adv_test_lb = tf.reduce_mean(vi.lower_bound.elbo())

        real_energy = tf.reduce_mean(D_psi(input_origin_x))
        reconstruct_energy = tf.reduce_mean(D_psi(test_chain.model['x'].distribution.mean))
        pd_energy = tf.reduce_mean(
            D_psi(test_pn_net['x'].distribution.mean) * tf.exp(
                test_pn_net['z'].log_prob().log_energy_prob - test_pn_net['z'].log_prob()))
        pn_energy = tf.reduce_mean(D_psi(test_pn_net['x'].distribution.mean))
        log_Z_compute_op = spt.ops.log_mean_exp(
            -test_pn_net['z'].log_prob().energy - test_pn_net['z'].log_prob())
        kl_adv_and_gaussian = tf.reduce_mean(
            test_pn_net['z'].log_prob() - test_pn_net['z'].log_prob().log_energy_prob
        )
    xi_node = get_var('p_net/xi')
    # derive the optimizer
    with tf.name_scope('optimizing'):
        VAE_params = tf.trainable_variables('q_net') + tf.trainable_variables('G_theta') + tf.trainable_variables(
            'beta') + tf.trainable_variables('posterior_flow') + tf.trainable_variables('p_net/xi')
        D_params = tf.trainable_variables('D_psi')
        VAE_G_params = tf.trainable_variables('G_theta')
        G_params = tf.trainable_variables('G_omega')
        print("========VAE_params=========")
        print(VAE_params)
        print("========D_params=========")
        print(D_params)
        print("========G_params=========")
        print(G_params)
        with tf.variable_scope('VAE_optimizer'):
            VAE_optimizer = tf.train.AdamOptimizer(learning_rate)
            VAE_grads = VAE_optimizer.compute_gradients(VAE_loss, VAE_params)
        with tf.variable_scope('VAE_G_optimizer'):
            VAE_G_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            VAE_G_grads = VAE_G_optimizer.compute_gradients(VAE_G_loss, VAE_G_params)
        with tf.variable_scope('D_optimizer'):
            D_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            D_grads = D_optimizer.compute_gradients(D_loss, D_params)
        with tf.variable_scope('G_optimizer'):
            G_optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.5, beta2=0.999)
            G_grads = G_optimizer.compute_gradients(G_loss, G_params)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            VAE_train_op = VAE_optimizer.apply_gradients(VAE_grads)
            VAE_G_train_op = VAE_optimizer.apply_gradients(VAE_G_grads)
            G_train_op = G_optimizer.apply_gradients(G_grads)
            D_train_op = D_optimizer.apply_gradients(D_grads)

    # derive the plotting function
    with tf.name_scope('plotting'):
        sample_n_z = 100
        gan_plots = tf.reshape(p_omega_net(n_z=sample_n_z, beta=beta)['x'].distribution.mean,
                               (-1,) + config.x_shape)
        initial_z = q_net(gan_plots, posterior_flow)['z']
        gan_plots = 256.0 * gan_plots / 2 + 127.5
        initial_z = tf.expand_dims(initial_z, axis=1)
        plot_net = p_net(n_z=sample_n_z, mcmc_iterator=20, beta=beta, initial_z=initial_z)
        plot_history_e_z = plot_net['z'].history_e_z
        plot_history_z = plot_net['z'].history_z
        plot_history_pure_e_z = plot_net['z'].history_pure_e_z
        plot_history_ratio = plot_net['z'].history_ratio
        x_plots = 256.0 * tf.reshape(
            plot_net['x'].distribution.mean, (-1,) + config.x_shape) / 2 + 127.5
        reconstruct_q_net = q_net(input_x, posterior_flow)
        reconstruct_z = reconstruct_q_net['z']
        reconstruct_plots = 256.0 * tf.reshape(
            p_net(observed={'z': reconstruct_z}, beta=beta)['x'].distribution.mean,
            (-1,) + config.x_shape
        ) / 2 + 127.5
        plot_reconstruct_energy = D_psi(reconstruct_plots)
        gan_plots = tf.clip_by_value(gan_plots, 0, 255)
        x_plots = tf.clip_by_value(x_plots, 0, 255)
        reconstruct_plots = tf.clip_by_value(reconstruct_plots, 0, 255)

    def plot_samples(loop):
        with loop.timeit('plot_time'):
            try:
                # plot reconstructs
                for [x] in reconstruct_train_flow:
                    x_samples = uniform_sampler.sample(x)
                    images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                    images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                    images[2::3, ...] = np.round(session.run(
                        reconstruct_plots, feed_dict={input_x: x}))
                    batch_reconstruct_z = session.run(reconstruct_z, feed_dict={input_x: x})
                    # print(np.mean(batch_reconstruct_z ** 2, axis=-1))
                    save_images_collection(
                        images=images,
                        filename='plotting/train.reconstruct/{}.png'.format(loop.epoch),
                        grid_size=(20, 15),
                        results=results,
                    )
                    break

                # plot reconstructs
                for [x] in reconstruct_test_flow:
                    x_samples = uniform_sampler.sample(x)
                    images = np.zeros((300,) + config.x_shape, dtype=np.uint8)
                    images[::3, ...] = np.round(256.0 * x / 2 + 127.5)
                    images[1::3, ...] = np.round(256.0 * x_samples / 2 + 127.5)
                    images[2::3, ...] = np.round(session.run(
                        reconstruct_plots, feed_dict={input_x: x}))
                    save_images_collection(
                        images=images,
                        filename='plotting/test.reconstruct/{}.png'.format(loop.epoch),
                        grid_size=(20, 15),
                        results=results,
                    )
                    break
                # plot samples
                [images, gan_images, batch_history_e_z, batch_history_z, batch_history_pure_e_z,
                 batch_history_ratio] = session.run(
                    [x_plots, gan_plots, plot_history_e_z, plot_history_z, plot_history_pure_e_z, plot_history_ratio])
                # print(batch_history_e_z)
                # print(np.mean(batch_history_z ** 2, axis=-1))
                # print(batch_history_pure_e_z)
                # print(batch_history_ratio)
                save_images_collection(
                    images=np.round(gan_images),
                    filename='plotting/sample/gan-{}.png'.format(loop.epoch),
                    grid_size=(10, 10),
                    results=results,
                )
                save_images_collection(
                    images=np.round(images),
                    filename='plotting/sample/{}.png'.format(loop.epoch),
                    grid_size=(10, 10),
                    results=results,
                )
            except Exception as e:
                print(e)

    # prepare for training and testing data
    (_x_train, _y_train), (_x_test, _y_test) = \
        spt.datasets.load_cifar10(x_shape=config.x_shape)
    # train_flow = bernoulli_flow(
    #     x_train, config.batch_size, shuffle=True, skip_incomplete=True)
    x_train = (_x_train - 127.5) / 256.0 * 2
    x_test = (_x_test - 127.5) / 256.0 * 2
    uniform_sampler = UniformNoiseSampler(-1.0 / 256.0, 1.0 / 256.0, dtype=np.float)
    train_flow = spt.DataFlow.arrays([x_train], config.batch_size, shuffle=True, skip_incomplete=True)
    # train_flow = train_flow.map(uniform_sampler)
    gan_train_flow = spt.DataFlow.arrays(
        [np.concatenate([x_train, x_test], axis=0)], config.batch_size, shuffle=True, skip_incomplete=True)
    gan_train_flow = gan_train_flow.map(uniform_sampler)
    reconstruct_train_flow = spt.DataFlow.arrays(
        [x_train], 100, shuffle=True, skip_incomplete=False)
    reconstruct_test_flow = spt.DataFlow.arrays(
        [x_test], 100, shuffle=True, skip_incomplete=False)
    test_flow = spt.DataFlow.arrays(
        [np.repeat(x_test, config.test_x_samples, axis=0)], config.test_batch_size)
    test_flow = test_flow.map(uniform_sampler)

    with spt.utils.create_session().as_default() as session, \
            train_flow.threaded(5) as train_flow:
        spt.utils.ensure_variables_initialized()

        # initialize the network
        for [x] in train_flow:
            print('Network initialized, first-batch loss is {:.6g}.\n'.
                  format(session.run(init_loss, feed_dict={input_x: x})))
            break

        # if config.z_dim == 512:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/48/19/6f3b6c3ef49ded8ba2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # elif config.z_dim == 1024:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/cd/19/6f9d69b5d1931e67e2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # elif config.z_dim == 2048:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/4d/19/6f9d69b5d19398c8c2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # elif config.z_dim == 3072:
        #     restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/5d/19/6f9d69b5d1936fb2d2d5/checkpoint/checkpoint/checkpoint.dat-390000'
        # else:
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
                         'adv_test_nll': adv_test_nll, 'adv_test_lb': adv_test_lb,
                         'reconstruct_energy': reconstruct_energy,
                         'real_energy': real_energy,
                         'pd_energy': pd_energy, 'pn_energy': pn_energy,
                         'test_recon': test_recon, 'kl_adv_and_gaussian': kl_adv_and_gaussian, 'test_mse': test_mse},
                inputs=[input_x],
                data_flow=test_flow,
                time_metric_name='test_time'
            )

            loop.print_training_summary()
            spt.utils.ensure_variables_initialized()

            epoch_iterator = loop.iter_epochs()

            n_critical = config.n_critical
            # adversarial training
            for epoch in epoch_iterator:
                step_iterator = MyIterator(train_flow)
                while step_iterator.has_next:
                    if epoch <= config.warm_up_start:
                        # discriminator training
                        for step, [x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            [_, batch_D_loss, batch_D_real] = session.run(
                                [D_train_op, D_loss, D_real], feed_dict={
                                    input_x: x,
                                })
                            loop.collect_metrics(D_loss=batch_D_loss)
                            loop.collect_metrics(D_real=batch_D_real)

                        # generator training x
                        [_, batch_G_loss] = session.run(
                            [G_train_op, G_loss], feed_dict={
                            })
                        loop.collect_metrics(G_loss=batch_G_loss)
                    else:
                        # vae training
                        for step, [x] in loop.iter_steps(limited(step_iterator, n_critical)):
                            [_, batch_VAE_loss, beta_value, xi_value, batch_train_recon, batch_train_recon_energy,
                             batch_VAE_D_real, batch_train_kl, batch_train_grad_penalty] = session.run(
                                [VAE_train_op, VAE_loss, beta, xi_node, train_recon, train_recon_energy, VAE_D_real,
                                 train_kl, train_grad_penalty],
                                feed_dict={
                                    input_x: x,
                                    warm: 1.0  # min(1.0, 1.0 * epoch / config.warm_up_epoch)
                                })
                            loop.collect_metrics(batch_VAE_loss=batch_VAE_loss)
                            loop.collect_metrics(xi=xi_value)
                            loop.collect_metrics(beta=beta_value)
                            loop.collect_metrics(train_recon=batch_train_recon)
                            loop.collect_metrics(train_recon_energy=batch_train_recon_energy)
                            loop.collect_metrics(D_real=batch_VAE_D_real)
                            loop.collect_metrics(train_kl=batch_train_kl)
                            loop.collect_metrics(train_grad_penalty=batch_train_grad_penalty)
                            # loop.print_logs()

                        # generator training x
                        # [_, batch_VAE_G_loss] = session.run(
                        #     [VAE_G_train_op, VAE_G_loss], feed_dict={
                        #     })
                        # loop.collect_metrics(VAE_G_loss=batch_VAE_G_loss)

                if epoch in config.lr_anneal_epoch_freq:
                    learning_rate.anneal()

                if epoch == config.warm_up_start:
                    learning_rate.set(config.initial_lr)

                if epoch % config.plot_epoch_freq == 0:
                    plot_samples(loop)

                if epoch % config.test_epoch_freq == 0:
                    log_Z_list = []
                    for i in range(config.log_Z_times):
                        log_Z_list.append(session.run(log_Z_compute_op))
                    from scipy.misc import logsumexp
                    log_Z = logsumexp(np.asarray(log_Z_list)) - np.log(config.log_Z_times)
                    get_log_Z().set(log_Z)
                    print('log_Z_list:{}'.format(log_Z_list))
                    print('log_Z:{}'.format(log_Z))
                    with loop.timeit('eval_time'):
                        evaluator.run()

                    dataset_img = _x_train

                    sample_img = []
                    for i in range((len(x_train)) // 100 + 1):
                        sample_img.append(session.run(gan_plots))
                    sample_img = np.concatenate(sample_img, axis=0).astype('uint8')
                    sample_img = sample_img[:len(dataset_img)]
                    sample_img = np.asarray(sample_img)

                    FID = get_fid(sample_img, dataset_img)
                    # turn to numpy array
                    IS_mean, IS_std = get_inception_score(sample_img)
                    loop.collect_metrics(FID=FID)
                    loop.collect_metrics(IS=IS_mean)

                loop.collect_metrics(lr=learning_rate.get())
                loop.print_logs()

    # print the final metrics and close the results object
    print_with_title('Results', results.format_metrics(), before='\n')
    results.close()


if __name__ == '__main__':
    main()
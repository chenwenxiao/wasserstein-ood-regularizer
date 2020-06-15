# -*- coding: utf-8 -*-
import mltk
from mltk.data import ArraysDataStream
from tensorkit import tensor as T
import sys
from argparse import ArgumentParser

from pprint import pformat

from matplotlib import pyplot
import torch

import tfsnippet as spt
from tfsnippet.examples.utils import (MLResults,
                                      print_with_title)
import numpy as np

from flow_next.common import TrainConfig, DataSetConfig, make_dataset, train_model
from flow_next.models.glow import GlowConfig, Glow
from ood_regularizer.experiment.datasets.overall import load_overall, load_complexity
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import plot_fig, make_diagram_torch

from utils.data import SplitInfo
from utils.evaluation import dequantized_bpd


class ExperimentConfig(mltk.Config):
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
    max_epoch = 400
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
    glow_warm_up_epochs = 50
    pretrain = True

    compressor = 2  # 0 for jpeg, 1 for png, 2 for flif

    max_step = None
    batch_size = 64
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
    distill_epoch = 5000

    epsilon = -20.0
    min_logstd_of_q = -3.0

    sample_n_z = 100

    x_shape = (32, 32, 3)
    x_shape_multiple = 3072
    extra_stride = 2

    train = TrainConfig(
        optimizer='adamax',
        init_batch_size=128,
        batch_size=64,
        test_batch_size=64,
        test_epoch_freq=10,
        max_epoch=50,
        grad_global_clip_norm=None,
        # grad_global_clip_norm=100.0,
        debug=True
    )
    model = GlowConfig(
        hidden_conv_activation='relu',
        hidden_conv_channels=[64, 64],
        depth=3,
        levels=3,
    )
    in_dataset = DataSetConfig(name='cifar10')
    out_dataset = DataSetConfig(name='svhn')


def main():
    with mltk.Experiment(ExperimentConfig, args=sys.argv[2:]) as exp, \
            T.use_device(T.first_gpu_device()):
        exp.make_dirs('plotting')
        config = exp.config
        # prepare for training and testing data
        x_train_complexity, x_test_complexity = load_complexity(config.in_dataset.name, config.compressor)
        svhn_train_complexity, svhn_test_complexity = load_complexity(config.out_dataset.name, config.compressor)

        restore_checkpoint = '/mnt/mfs/mlstorage-experiments/cwx17/85/a7/48d18cbe8ce08c837ee5/model.pkl'

        # load the dataset
        cifar_train_dataset, cifar_test_dataset = make_dataset(config.in_dataset)
        print('CIFAR DataSet loaded.')
        svhn_train_dataset, svhn_test_dataset = make_dataset(config.out_dataset)
        print('SVHN DataSet loaded.')

        cifar_train_flow = cifar_train_dataset.get_stream('train', 'x', config.batch_size)
        cifar_test_flow = cifar_test_dataset.get_stream('test', 'x', config.batch_size)
        svhn_train_flow = svhn_train_dataset.get_stream('train', 'x', config.batch_size)
        svhn_test_flow = svhn_test_dataset.get_stream('test', 'x', config.batch_size)

        if restore_checkpoint is not None:
            model = torch.load(restore_checkpoint)
        else:
            # construct the model
            model = Glow(cifar_train_dataset.slots['x'], exp.config.model)
            print('Model constructed.')

            # train the model
            train_model(exp, model, cifar_train_dataset, cifar_test_dataset)

        torch.save(model, 'model.pkl')

        with mltk.TestLoop() as loop:

            def eval_ll(x):
                x = T.from_numpy(x)
                ll, outputs = model(x)
                bpd = -dequantized_bpd(ll, cifar_train_dataset.slots['x'])
                return T.to_numpy(bpd)

            def eval_log_det(x):
                x = T.from_numpy(x)
                ll, outputs = model(x)
                log_det = outputs[0].log_det
                for output in outputs[1:]:
                    log_det = log_det + output.log_det
                log_det = -dequantized_bpd(log_det, cifar_train_dataset.slots['x'])
                return T.to_numpy(log_det)

            cifar_train_ll, cifar_test_ll, svhn_train_ll, svhn_test_ll = make_diagram_torch(
                loop, eval_ll,
                [cifar_train_flow, cifar_test_flow, svhn_train_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Train', config.in_dataset.name + ' Test',
                       config.out_dataset.name + ' Train', config.out_dataset.name + ' Test'],
                fig_name='log_prob_histogram'
            )

            def t_perm(base, another_arrays=None):
                base = sorted(base)
                N = len(base)
                return_arrays = []
                for array in another_arrays:
                    return_arrays.append(-np.abs(np.searchsorted(base, array) - N // 2))
                return return_arrays

            [cifar_train_nll_t, cifar_test_nll_t, svhn_train_nll_t, svhn_test_nll_t] = t_perm(
                cifar_train_ll, [cifar_train_ll, cifar_test_ll, svhn_train_ll, svhn_test_ll])

            loop.add_metrics(T_perm_histogram=plot_fig(
                data_list=[cifar_train_nll_t, cifar_test_nll_t, svhn_train_nll_t, svhn_test_nll_t],
                color_list=['red', 'salmon', 'green', 'lightgreen'],
                label_list=[config.in_dataset.name + ' Train', config.in_dataset.name + ' Test',
                            config.out_dataset.name + ' Train', config.out_dataset.name + ' Test'],
                x_label='bits/dim', fig_name='T_perm_histogram'))

            loop.add_metrics(ll_with_complexity_histogram=plot_fig(
                data_list=[cifar_train_ll + x_train_complexity, cifar_test_ll + x_test_complexity,
                           svhn_train_ll + svhn_train_complexity, svhn_test_ll + svhn_test_complexity],
                color_list=['red', 'salmon', 'green', 'lightgreen'],
                label_list=[config.in_dataset.name + ' Train', config.in_dataset.name + ' Test',
                            config.out_dataset.name + ' Train', config.out_dataset.name + ' Test'],
                x_label='bits/dim', fig_name='ll_with_complexity_histogram'))

            cifar_train_det, cifar_test_det, svhn_train_det, svhn_test_det = make_diagram_torch(
                loop, eval_log_det,
                [cifar_train_flow, cifar_test_flow, svhn_train_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Train', config.in_dataset.name + ' Test',
                       config.out_dataset.name + ' Train', config.out_dataset.name + ' Test'],
                fig_name='log_det_histogram')

            loop.add_metrics(origin_log_prob_histogram=plot_fig(
                data_list=[cifar_train_ll - cifar_train_det, cifar_test_ll - cifar_test_det,
                           svhn_train_ll - svhn_train_det, svhn_test_ll - svhn_test_det],
                color_list=['red', 'salmon', 'green', 'lightgreen'],
                label_list=[config.in_dataset.name + ' Train', config.in_dataset.name + ' Test',
                            config.out_dataset.name + ' Train', config.out_dataset.name + ' Test'],
                x_label='bits/dim', fig_name='origin_log_prob_histogram'))

            if not config.pretrain:
                model = Glow(cifar_train_dataset.slots['x'], exp.config.model)

            def data_generator():
                mixed_array = get_mixed_array(
                    config,
                    cifar_train_dataset.get_stream('train', ['x'], config.batch_size).get_arrays()[0],
                    cifar_test_dataset.get_stream('test', ['x'], config.batch_size).get_arrays()[0],
                    svhn_train_dataset.get_stream('train', ['x'], config.batch_size).get_arrays()[0],
                    svhn_test_dataset.get_stream('test', ['x'], config.batch_size).get_arrays()[0]
                )
                print(mixed_array.shape)
                steam = ArraysDataStream([mixed_array], batch_size=config.batch_size, shuffle=True,
                                         skip_incomplete=True)
                for [x] in steam:
                    yield [x]

            train_model(exp, model, svhn_train_dataset, svhn_test_dataset, data_generator() if config.use_transductive else None)

            make_diagram_torch(
                loop, eval_ll,
                [cifar_train_flow, cifar_test_flow, svhn_train_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Train', config.in_dataset.name + ' Test',
                       config.out_dataset.name + ' Train', config.out_dataset.name + ' Test'],
                fig_name='log_prob_mixed_histogram'
            )

            make_diagram_torch(
                loop, lambda x: -eval_ll(x),
                [cifar_train_flow, cifar_test_flow, svhn_train_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Train', config.in_dataset.name + ' Test',
                       config.out_dataset.name + ' Train', config.out_dataset.name + ' Test'],
                fig_name='kl_histogram',
                addtion_data=[cifar_train_ll, cifar_test_ll, svhn_train_ll, svhn_test_ll]
            )


if __name__ == '__main__':
    main()

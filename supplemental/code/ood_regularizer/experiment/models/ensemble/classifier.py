# -*- coding: utf-8 -*-
import mltk
from sklearn.covariance import EmpiricalCovariance
from mltk.data import ArraysDataStream, DataStream
import tensorkit as tk
from tensorkit import tensor as T
import sys
import torch
import numpy as np

from flow_next.common import TrainConfig, DataSetConfig, make_dataset, train_model, ImageAugmentationMapper, get_mapper
from flow_next.common.train_utils import train_classifier_ensemble, train_classifier
from flow_next.models.glow import GlowConfig, Glow
from ood_regularizer.experiment.datasets.overall import load_overall, load_complexity
from ood_regularizer.experiment.models.utils import get_mixed_array
from ood_regularizer.experiment.utils import plot_fig, make_diagram_torch, get_ele_torch

from utils.evaluation import dequantized_bpd
import torch.autograd as autograd
import torchvision.models as models
from imgaug import augmenters as iaa


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
    pretrain = False

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
    class_num = 10
    features_nums = 5
    ensemble_times = 5

    classifier_train = TrainConfig(
        optimizer='adamax',
        init_batch_size=128,
        batch_size=64,
        test_batch_size=64,
        test_epoch_freq=10,
        max_epoch=50,
        grad_global_clip_norm=None,
        # grad_global_clip_norm=100.0,
        ensemble_epsilon=0.02,
        debug=True
    )
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
        hidden_conv_channels=[128, 128],
        depth=6,
        levels=3,
    )
    in_dataset = 'cifar10'
    out_dataset = 'svhn'
    count_experiment = False


def main():
    with mltk.Experiment(ExperimentConfig, args=sys.argv[1:]) as exp, \
            T.use_device(T.first_gpu_device()):
        while True:
            try:
                exp.make_dirs('plotting')
                break
            except Exception:
                pass
        config = exp.config
        # prepare for training and testing data
        config.in_dataset = DataSetConfig(name=config.in_dataset)
        config.out_dataset = DataSetConfig(name=config.out_dataset)
        x_train_complexity, x_test_complexity = load_complexity(config.in_dataset.name, config.compressor)
        svhn_train_complexity, svhn_test_complexity = load_complexity(config.out_dataset.name, config.compressor)

        experiment_dict = {
        }
        print(experiment_dict)
        if config.in_dataset.name in experiment_dict:
            restore_dir = experiment_dict[config.in_dataset.name]
        else:
            restore_dir = None
        print('restore model from {}'.format(restore_dir))

        # load the dataset
        cifar_train_dataset, cifar_test_dataset, cifar_dataset = make_dataset(config.in_dataset)
        print('CIFAR DataSet loaded.')
        svhn_train_dataset, svhn_test_dataset, svhn_dataset = make_dataset(config.out_dataset)
        print('SVHN DataSet loaded.')
        config.class_num = cifar_train_dataset.slots['y'].max_val + 1

        cifar_train_flow = cifar_test_dataset.get_stream('train', 'x', config.batch_size)
        cifar_test_flow = cifar_test_dataset.get_stream('test', 'x', config.batch_size)
        svhn_train_flow = svhn_test_dataset.get_stream('train', 'x', config.batch_size)
        svhn_test_flow = svhn_test_dataset.get_stream('test', 'x', config.batch_size)

        x_train = cifar_dataset.get_array('train', 'x')
        y_train = cifar_dataset.get_array('train', 'y')
        x_test = cifar_dataset.get_array('test', 'x')
        y_test = cifar_dataset.get_array('test', 'y')
        svhn_train = svhn_dataset.get_array('train', 'x')
        svhn_test = svhn_dataset.get_array('test', 'x')

        if restore_dir is None:
            for i in range(config.ensemble_times):
                classifier = models.resnet34(num_classes=config.class_num).cuda()
                train_classifier(exp, classifier, cifar_test_dataset, cifar_test_dataset)

                @torch.no_grad()
                def eval_predict(x):
                    x = T.from_numpy(x)
                    predict = classifier(x)
                    predict = T.argmax(predict, axis=-1)
                    return T.to_numpy(predict)

                cifar_test_predict = get_ele_torch(eval_predict, cifar_test_flow)
                print('Correct number in cifar test is {}'.format(
                    np.sum(cifar_test_predict == y_test)))
                cifar_train_predict = get_ele_torch(eval_predict, cifar_train_flow)
                print('Correct number in cifar train is {}'.format(
                    np.sum(cifar_train_predict == y_train)))

                torch.save(classifier, 'classifier_{}.pkl'.format(i))

        with mltk.TestLoop() as loop:
            cifar_test_prob = []
            svhn_test_prob = []
            for i in range(config.ensemble_times):
                if restore_dir is None:
                    classifier = torch.load('classifier_{}.pkl'.format(i))
                else:
                    classifier = torch.load(restore_dir + '/classifier_{}.pkl'.format(i))

                @torch.no_grad()
                def eval_prob(x):
                    x = T.from_numpy(x)
                    predict = classifier(x)
                    prob = torch.softmax(predict, dim=-1)
                    return T.to_numpy(prob)

                cifar_test_prob.append(get_ele_torch(eval_prob, cifar_test_flow))
                svhn_test_prob.append(get_ele_torch(eval_prob, svhn_test_flow))
            cifar_test_prob = np.asarray(cifar_test_prob)  # [ensemble_times, n, class_num]
            svhn_test_prob = np.asarray(svhn_test_prob)  # [ensemble_times, n, class_num]

            cifar_test_orverall_prob = np.mean(cifar_test_prob, axis=0)
            svhn_test_orverall_prob = np.mean(svhn_test_prob, axis=0)

            cifar_test_disagreement = np.mean(
                np.sum(cifar_test_prob * np.log(cifar_test_prob + 1e-10) - np.log(cifar_test_orverall_prob + 1e-10),
                       axis=-1),
                axis=0)

            svhn_test_disagreement = np.mean(
                np.sum(svhn_test_prob * np.log(svhn_test_prob + 1e-10) - np.log(svhn_test_orverall_prob + 1e-10),
                       axis=-1),
                axis=0)

            loop.add_metrics(disagreement_histogram=plot_fig(
                data_list=[cifar_test_disagreement, svhn_test_disagreement],
                color_list=['red', 'green'],
                label_list=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                x_label='bits/dim', fig_name='disagreement_histogram'))

            loop.add_metrics(n_disagreement_histogram=plot_fig(
                data_list=[-cifar_test_disagreement, -svhn_test_disagreement],
                color_list=['red', 'green'],
                label_list=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                x_label='bits/dim', fig_name='n_disagreement_histogram'))


if __name__ == '__main__':
    main()

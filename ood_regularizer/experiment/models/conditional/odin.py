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
from flow_next.common.train_utils import train_classifier
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

    odin_T = 1000
    odin_epsilon = 0.0012 * 2  # multiple 2 for the normalization [-1, 1] instead of [0, 1] in ODIN

    classifier_train = TrainConfig(
        optimizer='adamax',
        init_batch_size=128,
        batch_size=64,
        test_batch_size=64,
        test_epoch_freq=10,
        max_epoch=200,
        grad_global_clip_norm=None,
        # grad_global_clip_norm=100.0,
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
        depth=3,
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

        if config.count_experiment:
            with open('/home/cwx17/research/ml-workspace/projects/wasserstein-ood-regularizer/count_experiments',
                      'a') as f:
                f.write(exp.abspath("") + '\n')
                f.close()

        experiment_dict = {
            'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/0a/66/02c52d867e43a2d825f5',
            'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/5a/66/02c52d867e43b46925f5',
            'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/d1/e5/02279d802d3a5a9d51f5',
            'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/50/e5/02c52d867e437d2851f5',
            'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/c1/e5/02279d802d3ab6c751f5',
            'constant': '/mnt/mfs/mlstorage-experiments/cwx17/f4/e5/02279d802d3a98e102f5',
            'noise': '/mnt/mfs/mlstorage-experiments/cwx17/b4/e5/02279d802d3aa7d002f5',
            'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/1a/66/02c52d867e43a2d825f5',
            'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/3c/d5/02732c28dc8db6c751f5',
            'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/d9/d5/02812baa4f70b68951f5',
            'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/e1/e5/02279d802d3ad38f51f5',
            'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/60/e5/02c52d867e4360ae51f5',
            'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/a2/f5/02279d802d3aa24a03f5',
            'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/92/f5/02279d802d3a3c3a03f5',
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

        config.class_num = np.max(y_train) + 1
        if restore_dir is None:
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

            torch.save(classifier, 'classifier.pkl')
        else:
            classifier = torch.load(restore_dir + '/classifier.pkl')

        with mltk.TestLoop() as loop:

            features_dict = {
                'cifar100': '/mnt/mfs/mlstorage-experiments/cwx17/9f/66/02279d802d3a658925f5',
                'cifar10': '/mnt/mfs/mlstorage-experiments/cwx17/b9/66/02c52d867e43131525f5',
                'fashion_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/5f/66/02279d802d3a328825f5',
                'mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/77/26/02732c28dc8d928825f5',
                'kmnist28': '/mnt/mfs/mlstorage-experiments/cwx17/f9/66/02c52d867e43b28825f5',
                'not_mnist28': '/mnt/mfs/mlstorage-experiments/cwx17/2a/66/02c52d867e43a8d825f5',
                'constant28': '/mnt/mfs/mlstorage-experiments/cwx17/3a/66/02c52d867e43520925f5',
                'noise28': '/mnt/mfs/mlstorage-experiments/cwx17/6f/66/02279d802d3a770925f5',
                'noise': '/mnt/mfs/mlstorage-experiments/cwx17/8f/66/02279d802d3aa57925f5',
                'constant': '/mnt/mfs/mlstorage-experiments/cwx17/54/16/02812baa4f70bf6925f5',
                'celeba': '/mnt/mfs/mlstorage-experiments/cwx17/a7/26/02732c28dc8d890925f5',
                'omniglot28': '/mnt/mfs/mlstorage-experiments/cwx17/ef/66/02279d802d3ae70135f5',
                'svhn': '/mnt/mfs/mlstorage-experiments/cwx17/30/76/02279d802d3a638235f5',
                'tinyimagenet': '/mnt/mfs/mlstorage-experiments/cwx17/9b/66/02c52d867e43fd7335f5',
            }
            if config.in_dataset.name in features_dict:
                outputs_features_mean = []
                outputs_features_precision = []
                for i in range(config.features_nums):
                    obj = np.load(features_dict[config.in_dataset.name] + '/outputs_features_{}.npz'.format(i))
                    outputs_features_mean.append(obj['outputs_features_mean'])
                    outputs_features_precision.append(obj['outputs_features_precision'])
            else:
                outputs_features = [[] for i in range(config.features_nums)]

                @torch.no_grad()
                def get_each_outputs(x):
                    x = T.from_numpy(x)
                    # See note [TorchScript super()]
                    x = classifier.conv1(x)
                    x = classifier.bn1(x)
                    x = classifier.relu(x)
                    x = classifier.maxpool(x)
                    outputs_features[0].append(T.to_numpy(torch.flatten(x, 1)))
                    x = classifier.layer1(x)
                    outputs_features[1].append(T.to_numpy(torch.flatten(x, 1)))
                    x = classifier.layer2(x)
                    outputs_features[2].append(T.to_numpy(torch.flatten(x, 1)))
                    x = classifier.layer3(x)
                    outputs_features[3].append(T.to_numpy(torch.flatten(x, 1)))
                    x = classifier.layer4(x)
                    outputs_features[4].append(T.to_numpy(torch.flatten(x, 1)))

                    x = classifier.avgpool(x)
                    x = torch.flatten(x, 1)
                    x = classifier.fc(x)

                    return T.to_numpy(x)

                with loop.timeit('output_features_time'):
                    outputs_features_mean = []
                    outputs_features_precision = []

                    get_ele_torch(get_each_outputs, cifar_train_flow)
                    for i in range(config.features_nums):
                        outputs_features[i] = np.concatenate(outputs_features[i], axis=0)
                        outputs_features_mean.append([])
                        outputs_features_precision.append([])
                        per_class = config.class_num // 10
                        rest = config.class_num - 10 * per_class
                        index = y_train == 0
                        for k in range(0, rest):
                            index = np.logical_or(index, y_train == k)
                        for j in range(rest, config.class_num, per_class):
                            index = y_train == j
                            for k in range(1, per_class):
                                index = np.logical_or(index, y_train == (j + k))
                            tmp_mean = np.mean(outputs_features[i][index], axis=0)
                            group_lasso = EmpiricalCovariance(assume_centered=False)
                            data = outputs_features[i][index] - tmp_mean
                            group_lasso.fit(data)
                            outputs_features_mean[-1].append(tmp_mean)
                            outputs_features_precision[-1].append(group_lasso.get_precision())
                        outputs_features_mean[-1] = np.asarray(outputs_features_mean[-1])
                        outputs_features_precision[-1] = np.asarray(outputs_features_precision[-1])

                        np.savez('outputs_features_{}'.format(i), outputs_features_mean=outputs_features_mean[i],
                                 outputs_features_precision=outputs_features_precision[i])

            @torch.no_grad()
            def eval_predict(x):
                x = T.from_numpy(x)
                predict = classifier(x)
                odin = T.reduce_max(torch.softmax(predict, dim=-1), axis=[-1])
                print(T.reduce_mean(odin))
                predict = T.argmax(predict, axis=-1)
                return T.to_numpy(predict)

            def get_gaussian_score(x, level):
                outputs = []
                x = classifier.conv1(x)
                x = classifier.bn1(x)
                x = classifier.relu(x)
                x = classifier.maxpool(x)
                outputs.append(torch.flatten(x, 1))
                x = classifier.layer1(x)
                outputs.append(torch.flatten(x, 1))
                x = classifier.layer2(x)
                outputs.append(torch.flatten(x, 1))
                x = classifier.layer3(x)
                outputs.append(torch.flatten(x, 1))
                x = classifier.layer4(x)
                outputs.append(torch.flatten(x, 1))
                gaussian_score = []
                for j in range(10):
                    zero_f = outputs[level] - T.from_numpy(outputs_features_mean[level][j], dtype=torch.float32)
                    gaussian_score.append(-torch.mm(
                        torch.mm(zero_f, T.from_numpy(outputs_features_precision[level][j], dtype=torch.float32)),
                        zero_f.t()).diag())
                gaussian_score = T.stack(gaussian_score, axis=0)
                gaussian_score = T.reduce_max(gaussian_score, axis=[0])
                return gaussian_score

            m_list = [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]
            for magnitude in m_list:
                def eval_Mahalanobis(x):
                    x = T.from_numpy(x)
                    x.requires_grad = True
                    mah = []
                    for level in range(config.features_nums):
                        gau = get_gaussian_score(x, level)
                        gradients = autograd.grad(gau, x, grad_outputs=torch.ones(gau.size()).cuda())[0]
                        sign = torch.sign(gradients)
                        x_hat = x + magnitude * sign
                        mah.append(T.to_numpy(get_gaussian_score(x_hat, level)))
                    return np.sum(np.asarray(mah), axis=0)

                make_diagram_torch(
                    loop, eval_Mahalanobis,
                    [cifar_test_flow, svhn_test_flow],
                    names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                    fig_name='Mahalanobis_{}_histogram'.format(magnitude)
                )

            @torch.no_grad()
            def eval_entropy(x):
                x = T.from_numpy(x)
                predict = classifier(x)
                odin = torch.softmax(predict, dim=-1)
                entropy = T.reduce_sum(odin * torch.log(odin), axis=[-1])
                return T.to_numpy(entropy)

            @torch.no_grad()
            def eval_max_prob(x):
                x = T.from_numpy(x)
                predict = classifier(x)
                prob = torch.softmax(predict, dim=-1)
                max_prob = T.reduce_max(prob, axis=[-1])
                return T.to_numpy(max_prob)

            def eval_odin(x):
                x = T.from_numpy(x)
                x.requires_grad = True
                S = torch.softmax(classifier(x) / config.odin_T, dim=-1)
                S = T.reduce_max(S, axis=[-1])
                log_S = torch.log(S)
                gradients = autograd.grad(-log_S, x, grad_outputs=torch.ones(log_S.size()).cuda())[0]
                sign = torch.sign(gradients)
                x_hat = x - config.odin_epsilon * sign

                odin = torch.softmax(classifier(x_hat) / config.odin_T, dim=-1)
                odin = T.reduce_max(odin, axis=[-1])

                return T.to_numpy(odin)

            make_diagram_torch(
                loop, eval_entropy,
                [cifar_test_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                fig_name='entropy_histogram'
            )

            make_diagram_torch(
                loop, lambda x: -eval_entropy(x),
                [cifar_test_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                fig_name='negative_entropy_histogram'
            )

            make_diagram_torch(
                loop, eval_max_prob,
                [cifar_test_flow, svhn_test_flow],
                names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                fig_name='max_prob_histogram'
            )

            M_list = [0, 0.0004, 0.0008, 0.0012, 0.0016, 0.002, 0.0024, 0.0028, 0.0032, 0.0036, 0.004]
            T_list = [1, 10, 100, 1000]
            for t in T_list:
                for m in M_list:
                    config.odin_T = t
                    config.odin_epsilon = m * 2  # since we normalize image to [-1, 1] instead of [0, 1]
                    make_diagram_torch(
                        loop, eval_odin,
                        [cifar_test_flow, svhn_test_flow],
                        names=[config.in_dataset.name + ' Test', config.out_dataset.name + ' Test'],
                        fig_name='odin_{}_{}_histogram'.format(t, m)
                    )


if __name__ == '__main__':
    main()

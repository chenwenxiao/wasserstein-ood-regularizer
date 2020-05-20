# -*- coding: utf-8 -*-
import tensorflow as tf

from matplotlib import pyplot
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score


def get_ele(op, flow, input_x):
    packs = []
    session = tf.get_default_session()
    for [batch_x] in flow:
        pack = session.run(
            op, feed_dict={
                input_x: batch_x,
            })  # [batch_size]
        pack = np.asarray(pack)
        # print(pack.shape)
        packs.append(pack)
    packs = np.concatenate(packs, axis=0)  # [len_of_flow]
    print(packs.shape)
    return packs


def draw_curve(cifar_test, svhn_test, fig_name):
    label = np.concatenate(([1] * len(cifar_test), [-1] * len(svhn_test)))
    score = np.concatenate((cifar_test, svhn_test))

    fpr, tpr, thresholds = roc_curve(label, score)
    precision, recall, thresholds = precision_recall_curve(label, score)
    pyplot.plot(recall, precision)
    pyplot.plot(fpr, tpr)
    print('%s auc: %4f, aupr: %4f, ap: %4f, FPR@TPR95: %4f' % (
        fig_name, auc(fpr, tpr), auc(recall, precision), average_precision_score(label, score),
        np.min(fpr[tpr > 0.95])))
    return auc(fpr, tpr)


def draw_metric(metric, color, label):
    metric = list(metric)

    n, bins, patches = pyplot.hist(metric, 40, normed=True, facecolor=color, alpha=0.4, label=label)

    index = []
    for i in range(len(bins) - 1):
        index.append((bins[i] + bins[i + 1]) / 2)

    def smooth(c, N=5):
        weights = np.hanning(N)
        return np.convolve(weights / weights.sum(), c)[N - 1:-N + 1]

    n[2:-2] = smooth(n)
    pyplot.plot(index, n, color=color)
    pyplot.legend()
    print('{} done. Mean is {}, Variance is {}'.format(label, np.mean(metric), np.var(metric)))


def plot_fig(data_list, color_list, label_list, x_label, fig_name, auc_pair=(1, -1)):
    pyplot.cla()
    pyplot.plot()
    pyplot.grid(c='silver', ls='--')
    pyplot.xlabel(x_label)
    spines = pyplot.gca().spines
    for sp in spines:
        spines[sp].set_color('silver')

    for i in range(len(data_list)):
        draw_metric(data_list[i], color_list[i], label_list[i])
    pyplot.savefig('plotting/%s.jpg' % fig_name)

    pyplot.cla()
    pyplot.plot()
    tmp = draw_curve(data_list[auc_pair[0]], data_list[auc_pair[1]], fig_name)
    pyplot.savefig('plotting/%s_curve.jpg' % fig_name)
    return tmp


def make_diagram(op, flows, input_x, colors=['red', 'salmon', 'green', 'lightgreen'],
                 names=['CIFAR-10 Train', 'CIFAR-10 Test', 'SVHN Train', 'SVHN Test'],
                 x_label='log(bit/dims)', fig_name='log_pro_histogram'):
    packs = [get_ele(op, flow, input_x) for flow in flows]
    return plot_fig(packs, colors, names, x_label, fig_name)


if __name__ == '__main__':
    pass

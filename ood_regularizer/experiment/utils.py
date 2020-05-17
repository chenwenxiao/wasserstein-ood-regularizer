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
    print('%s auc: %4f, ap: %4f' % (
        fig_name, auc(fpr, tpr), average_precision_score(label, score)))


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
    print('%s done.' % label)


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
    draw_curve(data_list[auc_pair[0]], data_list[auc_pair[1]], fig_name)
    pyplot.savefig('plotting/%s_curve.jpg' % fig_name)


def make_diagram(op, flows, input_x, colors=['red', 'salmon', 'green', 'lightgreen'],
                 names=['CIFAR-10 Train', 'CIFAR-10 Test', 'SVHN Train', 'SVHN Test'],
                 x_label='log(bit/dims)', fig_name='log_pro_histogram'):
    packs = [get_ele(op, flow, input_x) for flow in flows]
    plot_fig(packs, colors, names, x_label, fig_name)
    return packs


if __name__ == '__main__':
    pass

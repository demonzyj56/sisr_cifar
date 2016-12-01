#!/usr/bin/env python3

import numpy as np
import cPickle
import os


def unpickle(datadir, filename):
    if datadir is not None:
        filename = os.path.join(datadir, filename)
    with open(filename, 'rb') as f:
        d = cPickle.load(f)
    return d


def reshape_cifar10(data):
    assert data.shape == (10000, 3072)
    return np.reshape(data, (10000, 3, 32, 32))


def create_cifar10_train(data_dir=None):
    filenames = ["data_batch_" + str(i) for i in range(1, 6)]
    data_train = []
    labels_train = []
    for filename in filenames:
        d = unpickle(data_dir, filename)
        data_train.append(reshape_cifar10(d['data']))
        labels_train.append(d['labels'])
    data_train = np.concatenate(data_train, axis=0)
    labels_train = np.concatenate(labels_train, axis=0)
    assert data_train.shape == (50000, 3, 32, 32)
    assert len(labels_train) == 50000
    return {'data': data_train, 'labels': labels_train}


def create_cifar10_test(data_dir=None):
    d = unpickle(data_dir, "test_batch")
    d['data'] = reshape_cifar10(d['data'])
    assert d['data'].shape == (10000, 3, 32, 32)
    return d


if __name__ == "__main__":
    cifar10_train = create_cifar10_train()
    cifar10_test = create_cifar10_test()

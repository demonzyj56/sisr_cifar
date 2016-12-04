#/usr/bin/env python2

import numpy as np
import os
from create_cifar10 import *
from scipy.misc import imread, imsave


def save_cifar10_images(mode, data_dir, cache_dir, name_format="{}.png"):
    d = create_cifar10_train(data_dir=data_dir) if \
        mode == "train" else \
        create_cifar10_test(data_dir=data_dir)
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)
    for i in range(d['data'].shape[0]):
        img = d['data'][i]
        img = np.transpose(img, (1, 2, 0)) # to WxHxC
        imsave(os.path.join(cache_dir, name_format.format(str(i))), img)


def load_cifar10_images(mode, cache_dir, name_format="{}.png"):
    num_images = 50000 if mode == "train" else 10000
    im_blob = np.zeros((num_images, 3, 32, 32), dtype=np.dtype('uint8'))
    for i in range(num_images):
        img_name = name_format.format(str(i))
        img = imread(os.path.join(cache_dir, img_name), mode='RGB')
        im_blob[i] = np.transpose(img, (2, 0, 1))
    return im_blob


if __name__ == "__main__":
    save_cifar10_images("train",
                  "/home/leoyolo/data/cifar-10-batches-py",
                  "cache")
    im_blob = load_cifar10_images("train", "cache")
    print im_blob.shape

#!/usr/bin/env python

import os
import numpy as np
import cv2


kernel = cv2.getGaussianKernel(3, 0.05)  # size and sigma


def blur_image(img, scale):
    new_img = cv2.filter2D(img, -1, kernel)
    new_img = cv2.resize(new_img, None, fx=1./scale, fy=1./scale)
    # new_img = cv2.resize(new_img, None, fx=scale, fy=scale)
    return new_img


def read_image(idx, cache_dir, format="{}.png"):
    filename = os.path.join(cache_dir, format.format(str(idx)))
    return cv2.imread(filename)


def save_image(img, idx, cache_dir, format="{}_down.png"):
    filename = os.path.join(cache_dir, format.format(str(idx)))
    cv2.imwrite(filename, img)


if __name__ == "__main__":
    cache_dir = "cache_test"
    down_cache_dir = "cache_test_down"
    for i in range(0, 10000):
        print i
        img = read_image(i, cache_dir)
        img = blur_image(img, 2)
        save_image(img, i, down_cache_dir)

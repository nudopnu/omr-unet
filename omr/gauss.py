import cv2
import numpy as np


def gauss(x, mu, sigma, der=0, normalize=True):
    if der == 0:
        func = gauss_d0
    elif der == 1:
        func = gauss_d1
    elif der == 2:
        func = gauss_d2
    elif der == 3:
        func = gauss_d3
    norm = 1
    if normalize:
        for _ in range(der):
            norm *= sigma
    return norm * func(x, mu, sigma)


def gauss_d0(x, mu, sigma):
    if sigma == 0:
        raise Exception("sigma may not be zero!")
    xmms = (x - mu) * (x - mu)
    tss = 2 * sigma * sigma
    ssqrttp = sigma * np.sqrt(2 * np.pi)
    return (1 / ssqrttp) * np.exp(-xmms / tss)


def gauss_d1(x, mu, sigma):
    xmm = x - mu
    ss = sigma * sigma
    return -(xmm / ss) * gauss_d0(x, mu, sigma)


def gauss_d2(x, mu, sigma):
    xmm = x - mu
    ss = sigma * sigma
    ssss = ss * ss
    return (xmm * xmm - ss) / ssss * gauss_d0(x, mu, sigma)


def gauss_d3(x, mu, sigma):
    xmm = x - mu
    ss = sigma * sigma
    ssss = ss * ss
    return -(xmm * (3 * ss - xmm * xmm)) / (ssss * ss) * gauss_d0(x, mu, sigma)


def gauss_kernel_1d(size, mu, sigma, der=0, normalize=False):
    kernel = np.array([gauss(x, mu, sigma, der) for x in range(size)])
    sum_k = np.sum(kernel)
    if sum_k > 0 and sum_k != 1 and normalize:
        kernel = kernel / sum_k
    return kernel


def gauss_2d(p, mu, sigma, der=0):
    x, y = p
    mu_x, mu_y = mu
    dx = mu_x - x
    dy = mu_y - y
    dist = np.sqrt(dx * dx + dy * dy)
    return gauss(dist, 0, sigma, der)


def gauss_kernel_2d(size, mu, sigma, der=0, normalize=False):
    kernel = np.array(
        [[gauss_2d((x, y), mu, sigma, der) for x in range(size)] for y in range(size)]
    )
    sum_k = np.sum(np.abs(kernel))
    if sum_k > 0 and sum_k != 1 and normalize:
        kernel = kernel / sum_k
    return kernel

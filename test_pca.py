import os
from math import log10, log
import cv2
import rle
import numpy as np
from scipy.fft import dct, idct
from zigzag.my_zigzag import zigzag, inverse_zigzag
from scipy.io import loadmat


def pca(data, n):
    data_mean = np.mean(data, axis=0)
    zeroCentred_data = data - data_mean
    nn = data.shape[1]
    covMat = np.dot(zeroCentred_data, zeroCentred_data.T) / (nn - 1)
    featValue, featVec = np.linalg.eig(covMat)
    index = np.argsort(featValue)
    n_index = index[-n:]
    n_featVec = featVec[:, n_index]
    low_dim_data = np.dot(n_featVec.T, zeroCentred_data)
    return data_mean, n_featVec, low_dim_data


def pca_fast(data, n):
    data_mean = np.mean(data, axis=0)
    zeroCentred_data = data - data_mean
    covMat = np.dot(zeroCentred_data.T, zeroCentred_data)
    featValue, featVec = np.linalg.eig(covMat)
    index = np.argsort(featValue)
    n_index = index[-n:]
    n_featVec = featVec[:, n_index]
    n_featValue = featValue[n_index]
    n_featVec = np.dot(np.dot(zeroCentred_data, n_featVec), np.linalg.inv(np.diag(n_featValue)))
    low_dim_data = np.dot(n_featVec.T, zeroCentred_data)
    return data_mean, n_featVec, low_dim_data


H = loadmat('Data/Data1/Data1/Data1_H.mat')['H']
M, N, J, K = H.shape
num = N * M // 64
Hjk_r = []
Hjk_m = []
Hr = H.real
Hm = H.imag
# decomposition of matrix dose not have impact on complexity calculation
for j in range(J):
    for k in range(K):
        Hjk_r.append(Hr[:, :, j, k])
        Hjk_m.append(Hm[:, :, j, k])

multiply_count = 0
add_count = 0
other = 0
mode = 'fast'
# compression and encode
num_eig_value = 3
mean_r = []
n_featVec_r = []
low_dim_data_r = []
mean_m = []
n_featVec_m = []
low_dim_data_m = []
for Blo_r, Blo_m in zip(Hjk_r, Hjk_m):
    if mode == 'org':
        mean, featVec_n, ld_data = pca(Blo_r, num_eig_value)
        mean_r.append(mean)
        n_featVec_r.append(featVec_n)
        low_dim_data_r.append(ld_data)
        mean, featVec_n, ld_data = pca(Blo_m, num_eig_value)
        mean_m.append(mean)
        n_featVec_m.append(featVec_n)
        low_dim_data_m.append(ld_data)
        # calculate computation complexity
        multiply_count = N * M * M + M * N * num_eig_value
        add_count = (N - 1) * M * M + (M - 1) * N * num_eig_value
        other += M * M * 25 + 2 * M * N + 25

    if mode == 'fast':
        mean, featVec_n, ld_data = pca_fast(Blo_r, num_eig_value)
        mean_r.append(mean)
        n_featVec_r.append(featVec_n)
        low_dim_data_r.append(ld_data)
        mean, featVec_n, ld_data = pca_fast(Blo_m, num_eig_value)
        mean_m.append(mean)
        n_featVec_m.append(featVec_n)
        low_dim_data_m.append(ld_data)
        # calculate computation complexity
        multiply_count = N * N * M + M * N * num_eig_value + M * N * num_eig_value + M * num_eig_value ** 2
        add_count = (M - 1) * N * N + (N - 1) * M * num_eig_value + (M - 1) * N * num_eig_value + \
                    num_eig_value * (num_eig_value - 1) * M
        other += N * N * 25 + 2 * M * N + 25 + N * num_eig_value * 25 * 2 + num_eig_value * 25

storage_complexity = 0
# calculate storage complexity
storage_complexity = 2 * (N * num_eig_value * len(low_dim_data_r))

storage_complexity = 32 * storage_complexity
compression_ratio = (64 * M * N * J * K) / storage_complexity

# decode and calculate error
rec_Hjk_r = []
rec_Hjk_m = []
count_block = 0

for m_r, m_m, f_r, f_m, l_r, l_m in zip(mean_r, mean_m, n_featVec_r, n_featVec_m, low_dim_data_r, low_dim_data_m):
    rec_Hjk_r.append((np.dot(f_r, l_r) + m_r))
    rec_Hjk_m.append((np.dot(f_m, l_m) + m_m))
    multiply_count = num_eig_value * N * M
    add_count = (num_eig_value - 1) * N * M

computation_complexity = multiply_count * 3 + add_count + other

# err_H

Hjk = np.array(Hjk_r) + 1j * np.array(Hjk_m)
rec_Hjk = np.array(rec_Hjk_r) + 1j * np.array(rec_Hjk_m)
err_n = []
err_d = []
for i in range(Hjk.shape[0]):
    err_n.append(np.linalg.norm(rec_Hjk[i, :, :] - Hjk[i, :, :]) ** 2)
    err_d.append(np.linalg.norm(Hjk[i, :, :]) ** 2)

err = 10 * log10(np.mean(err_n) / np.mean(err_d))

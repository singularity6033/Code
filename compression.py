import os
from math import log10, log
import cv2
import rle
import numpy as np
from scipy.fft import dct, idct
from zigzag.my_zigzag import zigzag, inverse_zigzag
from scipy.io import loadmat


def range_map(data, MIN, MAX):
    d_min = np.max(data)
    d_max = np.min(data)
    return MIN + (MAX - MIN) / (d_max - d_min) * (data - d_min)


def reconstruct_matrix(data):
    h = data.shape[0]
    return np.concatenate((data[0:int(h / 2), :], data[int(h / 2):, :]), axis=1)


def construct_quantization_table(data, n):
    q_table = np.ones((8, 8))
    data_temp = data.copy()
    data_compare = np.abs(data)
    nums = np.sort(data_compare, axis=None)
    nums = nums[:n]
    for nnn in nums:
        min_index = np.where(data_compare == nnn)
        q_table[min_index] = 0
    return q_table


H = loadmat('Data/Data1/Data1/Data1_H.mat')['H']
M, N, J, K = H.shape
num = N * M // 64
Hjk_r = []
Hjk_m = []
Block_r = []
Block_m = []
qBlock_r = []
qBlock_m = []
rle_r = []
rle_m = []
range_r = np.zeros((J * K * num, 2))
range_m = np.zeros((J * K * num, 2))
Hr = H.real
Hm = H.imag
Q = 1
# decomposition of matrix dose not have impact on complexity calculation
for j in range(J):
    for k in range(K):
        Hjk_r.append(Hr[:, :, j, k])
        Hjk_m.append(Hm[:, :, j, k])
        for b in range(num):
            temp_r = np.concatenate((Hr[:, 0 + b * 4 * M:2 * M + b * 4 * M, j, k],
                                     Hr[:, 2 * M + b * 4 * M:4 * M + b * 4 * M, j, k]), axis=0)
            temp_m = np.concatenate((Hm[:, 0 + b * 4 * M:2 * M + b * 4 * M, j, k],
                                     Hm[:, 2 * M + b * 4 * M:4 * M + b * 4 * M, j, k]), axis=0)
            range_r[j + k + b, 0] = (np.min(temp_r))
            range_r[j + k + b, 1] = (np.max(temp_r))
            range_m[j + k + b, 0] = (np.min(temp_m))
            range_m[j + k + b, 1] = (np.max(temp_m))
            Block_r.append(temp_r)
            Block_m.append(temp_m)

multiply_count = 0
add_count = 0

# 8*8 quantization table (JPEG)
# q_table = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
#                     [12, 12, 14, 19, 26, 58, 60, 55],
#                     [14, 13, 16, 24, 40, 57, 69, 56],
#                     [14, 17, 22, 29, 51, 87, 80, 62],
#                     [18, 22, 37, 56, 68, 109, 103, 77],
#                     [24, 35, 55, 64, 81, 104, 113, 92],
#                     [49, 64, 78, 87, 103, 121, 120, 101],
#                     [72, 92, 95, 98, 112, 100, 103, 99]])
# q_table = np.array([[1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1],
#                     [1, 1, 1, 1, 1, 1, 1, 1]])
q = np.ones((8, 8))
# compression and encode
for Blo_r, Blo_m in zip(Block_r, Block_m):
    q_table_r = construct_quantization_table(Blo_r, 1)
    qBlock_r.append(dct(Blo_r, norm='ortho'))
    zigzagBlock_r = zigzag(dct(Blo_r, norm='ortho') * q_table_r)
    qBlock_m.append(dct(Blo_m, norm='ortho'))
    zigzagBlock_m = zigzag(dct(Blo_m, norm='ortho') * q_table_m)
    rle_r.append(rle.encode(zigzagBlock_r))
    rle_m.append(rle.encode(zigzagBlock_m))

storage_complexity = 0
# calculate storage complexity
for element_r, element_m in zip(rle_r, rle_m):
    storage_complexity += len(element_r[0]) + len(element_r[1]) + len(element_m[0]) + len(element_m[1])

storage_complexity = 32 * storage_complexity
compression_ratio = (64 * M * N * J * K) / storage_complexity

# decode and calculate error
rec_Block_r = []
rec_Block_m = []
rec_Hjk_r = []
rec_Hjk_m = []
count_block = 0

for element_r, element_m in zip(rle_r, rle_m):
    rec_Block_r.append(idct(inverse_zigzag(rle.decode(
        element_r[0], element_r[1]), 8, 8), norm='ortho'))
    rec_Block_m.append(idct(inverse_zigzag(rle.decode(
        element_m[0], element_m[1]), 8, 8), norm='ortho'))

for j in range(J):
    for k in range(K):
        one_rec_H_r = []
        one_rec_H_m = []
        for b in range(num):
            one_rec_H_r.append(reconstruct_matrix(rec_Block_r[count_block]))
            one_rec_H_m.append(reconstruct_matrix(rec_Block_m[count_block]))
            count_block += 1
        rec_Hjk_r.append(np.concatenate((one_rec_H_r[0], one_rec_H_r[1], one_rec_H_r[2], one_rec_H_r[3]), axis=1))
        rec_Hjk_m.append(np.concatenate((one_rec_H_m[0], one_rec_H_m[1], one_rec_H_m[2], one_rec_H_m[3]), axis=1))

# err_H

Hjk = np.array(Hjk_r) + 1j * np.array(Hjk_m)
rec_Hjk = np.array(rec_Hjk_r) + 1j * np.array(rec_Hjk_m)
err_n = []
err_d = []
for i in range(Hjk.shape[0]):
    err_n.append(np.linalg.norm(rec_Hjk[i, :, :] - Hjk[i, :, :]) ** 2)
    err_d.append(np.linalg.norm(Hjk[i, :, :]) ** 2)

err = 10 * log10(np.mean(err_n) / np.mean(err_d))

from math import log10
import rle
import numpy as np
from scipy.fft import dct, idct
from zigzag.my_zigzag import zigzag, inverse_zigzag
from scipy.io import loadmat


def construct_quantization_Block(data, n):
    data_temp = data.copy()
    data_compare = np.abs(data)
    nums = np.sort(data_compare, axis=None)
    nums = nums[:n]
    for nnn in nums:
        min_index = np.where(data_compare == nnn)
        data_temp[min_index] = 0
    return data_temp


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
cc = 0
# decomposition of matrix dose not have impact on complexity calculation
for j in range(J):
    for k in range(K):
        Hjk_r.append(Hr[:, :, j, k])
        Hjk_m.append(Hm[:, :, j, k])
        temp_r = Hr[:, :, j, k]
        temp_m = Hm[:, :, j, k]
        # range_r[cc, 0] = (np.min(temp_r))
        # range_r[cc, 1] = (np.max(temp_r))
        # range_m[cc, 0] = (np.min(temp_m))
        # range_m[cc, 1] = (np.max(temp_m))
        Block_r.append(temp_r)
        Block_m.append(temp_m)
        cc = cc + 1

# para related to computation complexity
multiply_count = 0
add_count = 0
zero_para = 20

# compression and encode
for Blo_r, Blo_m in zip(Block_r, Block_m):
    Blo_r = construct_quantization_Block(dct(Blo_r, norm='ortho'), zero_para)
    Blo_m = construct_quantization_Block(dct(Blo_m, norm='ortho'), zero_para)
    multiply_count += (N * M * 2) * 2
    add_count += (N * (M - 1) + M * (N - 1)) * 2
    zigzagBlock_r = zigzag(Blo_r)
    zigzagBlock_m = zigzag(Blo_m)
    add_count += zero_para * 2
    rle_r.append(rle.encode(zigzagBlock_r))
    rle_m.append(rle.encode(zigzagBlock_m))

storage_complexity = 0
# calculate storage complexity
for element_r, element_m in zip(rle_r, rle_m):
    storage_complexity += len(element_r[0]) + len(element_m[0])

storage_complexity = 32 * storage_complexity
compression_ratio = (64 * M * N * J * K) / storage_complexity

# decode and calculate error
rec_Block_r = []
rec_Block_m = []
rec_Hjk_r = []
rec_Hjk_m = []
count_block = 0

for element_r, element_m in zip(rle_r, rle_m):
    rec_Hjk_r.append(idct(inverse_zigzag(rle.decode(
        element_r[0], element_r[1]), 4, 64), norm='ortho'))
    rec_Hjk_m.append(idct(inverse_zigzag(rle.decode(
        element_m[0], element_m[1]), 4, 64), norm='ortho'))
    multiply_count += (N * M * 2) * 2
    add_count += (N * (M - 1) + M * (N - 1)) * 2

# for j in range(J):
#     for k in range(K):
#         rec_Hjk_r.append(range_map(rec_Block_r[count_block], range_r[count_block, 0], range_r[count_block, 1]))
#         rec_Hjk_m.append(range_map(rec_Block_m[count_block], range_m[count_block, 0], range_m[count_block, 1]))
#         count_block += 1


# err_H

Hjk = np.array(Hjk_r) + 1j * np.array(Hjk_m)
rec_Hjk = np.array(rec_Hjk_r) + 1j * np.array(rec_Hjk_m)
err_n = []
err_d = []
for i in range(Hjk.shape[0]):
    err_n.append(np.linalg.norm(rec_Hjk[i, :, :] - Hjk[i, :, :]) ** 2)
    err_d.append(np.linalg.norm(Hjk[i, :, :]) ** 2)

err = 10 * log10(np.mean(err_n) / np.mean(err_d))

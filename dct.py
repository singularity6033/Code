import pickle
from math import log10
import rle
import numpy as np
from scipy.fft import dct, idct
from zigzag.my_zigzag import zigzag, inverse_zigzag
from scipy.io import loadmat


def construct_quantization_Block(data, nn):
    if nn == 0:
        return data
    data_temp = data.copy()
    data_compare = np.abs(data)
    nums = np.sort(data_compare, axis=None)
    nums = nums[:nn]
    for nnn in nums:
        min_index = np.where(data_compare == nnn)
        data_temp[min_index] = 0
    return data_temp


err = [[] for i in range(6)]
computation_complexity = [[] for i in range(6)]
storage_complexity = [[] for i in range(6)]
for file_id in range(6):
    print("[INFO] processing data {}/{}...".format(file_id + 1, 6))
    for n in range(100):
        H = loadmat('Data' + str(file_id + 1) + '_H.mat')['H']
        M, N, J, K = H.shape
        Hjk_r = []
        Hjk_m = []
        Hr = H.real
        Hm = H.imag
        cc = 0
        # decomposition of matrix dose not have impact on complexity calculation
        for j in range(J):
            for k in range(K):
                Hjk_r.append(Hr[:, :, j, k])
                Hjk_m.append(Hm[:, :, j, k])
                cc = cc + 1

        # para related to computation complexity
        multiply_count = 0
        add_count = 0
        zero_para = n

        # compression and encode
        rle_r = []
        rle_m = []

        for Blo_r, Blo_m in zip(Hjk_r, Hjk_m):
            Blo_r = construct_quantization_Block(dct(Blo_r, norm='ortho'), zero_para)
            Blo_m = construct_quantization_Block(dct(Blo_m, norm='ortho'), zero_para)
            multiply_count += (N * M * 2) * 2
            add_count += (N * (M - 1) + M * (N - 1)) * 2
            zigzagBlock_r = zigzag(Blo_r)
            zigzagBlock_m = zigzag(Blo_m)
            rle_r.append(rle.encode(zigzagBlock_r))
            rle_m.append(rle.encode(zigzagBlock_m))

        sc = 0
        # calculate storage complexity
        for element_r, element_m in zip(rle_r, rle_m):
            sc += len(element_r[0]) + len(element_m[0])

        storage_complexity[file_id].append(32 * sc)
        compression_ratio = (64 * M * N * J * K) / sc

        # decode and calculate error
        rec_Hjk_r = []
        rec_Hjk_m = []
        count_block = 0

        for element_r, element_m in zip(rle_r, rle_m):
            rec_Hjk_r.append(idct(inverse_zigzag(rle.decode(
                element_r[0], element_r[1]), M, N), norm='ortho'))
            rec_Hjk_m.append(idct(inverse_zigzag(rle.decode(
                element_m[0], element_m[1]), M, N), norm='ortho'))
            multiply_count += (N * M + M * N - 2 * zero_para) * 2
            add_count += (N * (M - 1) + M * (N - 1) - 2 * zero_para) * 2

        computation_complexity[file_id].append(multiply_count * 3 + add_count)

        # err_H
        Hjk = np.array(Hjk_r) + 1j * np.array(Hjk_m)
        rec_Hjk = np.array(rec_Hjk_r) + 1j * np.array(rec_Hjk_m)
        err_n = []
        err_d = []
        for i in range(Hjk.shape[0]):
            err_n.append(np.linalg.norm(rec_Hjk[i, :, :] - Hjk[i, :, :]) ** 2)
            err_d.append(np.linalg.norm(Hjk[i, :, :]) ** 2)
        if 10 * log10(np.mean(err_n) / np.mean(err_d)) > -30:
            break
        else:
            err[file_id].append(10 * log10(np.mean(err_n) / np.mean(err_d)))

print("[INFO] saving label encoder...")
f = open('err.pickle', "wb")
f.write(pickle.dumps(err))
f.close()
f = open('computation_complexity.pickle', "wb")
f.write(pickle.dumps(computation_complexity))
f.close()
f = open('storage_complexity.pickle', "wb")
f.write(pickle.dumps(storage_complexity))
f.close()


err_w = [[] for i in range(6)]
computation_complexity_w = [[] for i in range(6)]
storage_complexity_w = [[] for i in range(6)]
for file_id in range(6):
    for n in range(100):
        print("[INFO] processing data {}/{}...".format(file_id + 1, 6))
        W = loadmat('Data' + str(file_id + 1) + '_W.mat')['W']
        N, L, J, K = W.shape
        Wjk_r = []
        Wjk_m = []
        Wr = W.real
        Wm = W.imag
        cc = 0
        # decomposition of matrix dose not have impact on complexity calculation
        for j in range(J):
            for k in range(K):
                Wjk_r.append(Wr[:, :, j, k])
                Wjk_m.append(Wm[:, :, j, k])
                cc = cc + 1

        # para related to computation complexity
        multiply_count = 0
        add_count = 0
        zero_para = n

        # compression and encode
        rle_r = []
        rle_m = []

        for Blo_r, Blo_m in zip(Wjk_r, Wjk_m):
            Blo_r = construct_quantization_Block(dct(Blo_r, norm='ortho'), zero_para)
            Blo_m = construct_quantization_Block(dct(Blo_m, norm='ortho'), zero_para)
            multiply_count += (L * N * 2) * 2
            add_count += (L * (N - 1) + N * (L - 1)) * 2
            zigzagBlock_r = zigzag(Blo_r)
            zigzagBlock_m = zigzag(Blo_m)
            rle_r.append(rle.encode(zigzagBlock_r))
            rle_m.append(rle.encode(zigzagBlock_m))

        sc = 0
        # calculate storage complexity
        for element_r, element_m in zip(rle_r, rle_m):
            sc += len(element_r[0]) + len(element_m[0])

        storage_complexity_w[file_id].append(32 * sc)
        compression_ratio = (64 * L * N * J * K) / sc

        # decode and calculate error
        rec_Wjk_r = []
        rec_Wjk_m = []
        count_block = 0

        for element_r, element_m in zip(rle_r, rle_m):
            rec_Wjk_r.append(idct(inverse_zigzag(rle.decode(
                element_r[0], element_r[1]), N, L), norm='ortho'))
            rec_Wjk_m.append(idct(inverse_zigzag(rle.decode(
                element_m[0], element_m[1]), N, L), norm='ortho'))
            multiply_count += (L * N + N * L - 2 * zero_para) * 2
            add_count += (L * (N - 1) + N * (L - 1) - 2 * zero_para) * 2

        computation_complexity_w[file_id].append(multiply_count * 3 + add_count)

        # err_W
        Wjk = np.array(Wjk_r) + 1j * np.array(Wjk_m)
        rec_Wjk = np.array(rec_Wjk_r) + 1j * np.array(rec_Wjk_m)
        err_n = []
        err_d = []
        for i in range(Wjk.shape[0]):
            err_n.append(np.linalg.norm(rec_Wjk[i, :, :] - Wjk[i, :, :]) ** 2)
            err_d.append(np.linalg.norm(Wjk[i, :, :]) ** 2)
        if 10 * log10(np.mean(err_n) / np.mean(err_d)) > -30:
            break
        else:
            err_w[file_id].append(10 * log10(np.mean(err_n) / np.mean(err_d)))

print("[INFO] saving label encoder...")
f = open('err_w.pickle', "wb")
f.write(pickle.dumps(err_w))
f.close()
f = open('computation_complexity_w.pickle', "wb")
f.write(pickle.dumps(computation_complexity_w))
f.close()
f = open('storage_complexity_w.pickle', "wb")
f.write(pickle.dumps(storage_complexity_w))
f.close()

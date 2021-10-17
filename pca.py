import pickle
from math import log10
import numpy as np
from scipy.io import loadmat
import time


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
    covMat = np.cov(zeroCentred_data, rowvar=False)
    featValue, featVec = np.linalg.eig(covMat)
    index = np.argsort(featValue)
    n_index = index[-n:]
    n_featVec = featVec[:, n_index]
    low_dim_data = np.dot(zeroCentred_data, n_featVec)
    return data_mean, n_featVec, low_dim_data


time_start = time.time()
mode = 'org'

err = []
computation_complexity = []
storage_complexity = []
num_eig_value = 2

for file_id in range(6):
    print("[INFO] processing data {}/{}...".format(file_id + 1, 6))
    H = loadmat('Data' + str(file_id + 1) + '_H.mat')['H']
    M, N, J, K = H.shape
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
    # compression and encode
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

    sc = 0
    # calculate storage complexity
    sc = 2 * (N * num_eig_value * len(low_dim_data_r))
    storage_complexity.append(32 * sc)
    # compression_ratio = (64 * M * N * J * K) / storage_complexity

    # decode and calculate error
    rec_Hjk_r = []
    rec_Hjk_m = []
    count_block = 0

    for m_r, m_m, f_r, f_m, l_r, l_m in zip(mean_r, mean_m, n_featVec_r, n_featVec_m, low_dim_data_r,
                                            low_dim_data_m):
        if mode == 'org':
            rec_Hjk_r.append((np.dot(f_r, l_r) + m_r))
            rec_Hjk_m.append((np.dot(f_m, l_m) + m_m))
        else:
            rec_Hjk_r.append(np.dot(l_r, f_r.T) + m_r)
            rec_Hjk_m.append(np.dot(l_m, f_m.T) + m_m)
        multiply_count = num_eig_value * N * M
        add_count = (num_eig_value - 1) * N * M

    computation_complexity.append(multiply_count * 3 + add_count + other)

    # err_H

    Hjk = np.array(Hjk_r) + 1j * np.array(Hjk_m)
    rec_Hjk = np.array(rec_Hjk_r) + 1j * np.array(rec_Hjk_m)
    err_n = []
    err_d = []
    for i in range(Hjk.shape[0]):
        err_n.append(np.linalg.norm(rec_Hjk[i, :, :] - Hjk[i, :, :]) ** 2)
        err_d.append(np.linalg.norm(Hjk[i, :, :]) ** 2)
    err.append(10 * log10(np.mean(err_n) / np.mean(err_d)))

print("[INFO] saving label encoder...")
f = open('err_h_pca_o.pickle', "wb")
f.write(pickle.dumps(err))
f.close()
f = open('computation_complexity_h_pca_o.pickle', "wb")
f.write(pickle.dumps(computation_complexity))
f.close()
f = open('storage_complexity_h_pca_o.pickle', "wb")
f.write(pickle.dumps(storage_complexity))
f.close()

for file_id in range(6):
    print("[INFO] processing data {}/{}...".format(file_id + 1, 6))
    W = loadmat('Data' + str(file_id + 1) + '_W.mat')['W']
    N, L, J, K = W.shape
    Wjk_r = []
    Wjk_m = []
    Wr = W.real
    Wm = W.imag
    # decomposition of matrix dose not have impact on complexity calculation
    for j in range(J):
        for k in range(K):
            Wjk_r.append(Wr[:, :, j, k])
            Wjk_m.append(Wm[:, :, j, k])

    multiply_count = 0
    add_count = 0
    other = 0
    # compression and encode
    mean_r = []
    n_featVec_r = []
    low_dim_data_r = []
    mean_m = []
    n_featVec_m = []
    low_dim_data_m = []
    for Blo_r, Blo_m in zip(Wjk_r, Wjk_m):
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
            multiply_count = L * N * N + N * L * num_eig_value
            add_count = (L - 1) * N * N + (N - 1) * L * num_eig_value
            other += N * N * 25 + 2 * N * L + 25

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
            multiply_count = L * L * N + N * L * num_eig_value + N * L * num_eig_value + N * num_eig_value ** 2
            add_count = (N - 1) * L * L + (L - 1) * N * num_eig_value + (N - 1) * L * num_eig_value + \
                        num_eig_value * (num_eig_value - 1) * N
            other += L * L * 25 + 2 * N * L + 25 + L * num_eig_value * 25 * 2 + num_eig_value * 25

    sc = 0
    # calculate storage complexity
    sc = 2 * (L * num_eig_value * len(low_dim_data_r))
    storage_complexity.append(32 * sc)
    # compression_ratio = (64 * N * L * J * K) / storage_complexity

    # decode and calculate error
    rec_Wjk_r = []
    rec_Wjk_m = []
    count_block = 0

    for m_r, m_m, f_r, f_m, l_r, l_m in zip(mean_r, mean_m, n_featVec_r, n_featVec_m, low_dim_data_r,
                                            low_dim_data_m):
        if mode == 'org':
            rec_Wjk_r.append((np.dot(f_r, l_r) + m_r))
            rec_Wjk_m.append((np.dot(f_m, l_m) + m_m))
        else:
            rec_Wjk_r.append(np.dot(l_r, f_r.T) + m_r)
            rec_Wjk_m.append(np.dot(l_m, f_m.T) + m_m)
        multiply_count = num_eig_value * L * N
        add_count = (num_eig_value - 1) * L * N

    computation_complexity.append(multiply_count * 3 + add_count + other)

    # err_W

    Wjk = np.array(Wjk_r) + 1j * np.array(Wjk_m)
    rec_Wjk = np.array(rec_Wjk_r) + 1j * np.array(rec_Wjk_m)
    err_n = []
    err_d = []
    for i in range(Wjk.shape[0]):
        err_n.append(np.linalg.norm(rec_Wjk[i, :, :] - Wjk[i, :, :]) ** 2)
        err_d.append(np.linalg.norm(Wjk[i, :, :]) ** 2)
    err.append(10 * log10(np.mean(err_n) / np.mean(err_d)))
time_end = time.time()
print("[INFO] saving label encoder...")
f = open('time_w_pca_o.pickle', "wb")
f.write(pickle.dumps(time_end - time_start))
f.close()
f = open('err_w_pca_f.pickle', "wb")
f.write(pickle.dumps(err))
f.close()
f = open('computation_complexity_w_pca_f.pickle', "wb")
f.write(pickle.dumps(computation_complexity))
f.close()
f = open('storage_complexity_w_pca_f.pickle', "wb")
f.write(pickle.dumps(storage_complexity))
f.close()
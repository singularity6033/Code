import pickle

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

storage_complexity = pickle.loads(open('storage_complexity.pickle', "rb").read())
storage_complexity_w = pickle.loads(open('storage_complexity_w.pickle', "rb").read())

storage_complexity_h_pca_f = pickle.loads(open('storage_complexity_h_pca_f.pickle', "rb").read())
storage_complexity_w_pca_f = pickle.loads(open('storage_complexity_w_pca_f.pickle', "rb").read())
storage_complexity_h_pca_o = pickle.loads(open('storage_complexity_h_pca_o.pickle', "rb").read())
storage_complexity_w_pca_o = pickle.loads(open('storage_complexity_w_pca_o.pickle', "rb").read())

computation_complexity = pickle.loads(open('computation_complexity.pickle', "rb").read())
computation_complexity_w = pickle.loads(open('computation_complexity_w.pickle', "rb").read())

computation_complexity_h_pca_f = pickle.loads(open('computation_complexity_h_pca_f.pickle', "rb").read())
computation_complexity_h_pca_o = pickle.loads(open('computation_complexity_h_pca_o.pickle', "rb").read())
computation_complexity_w_pca_f = pickle.loads(open('computation_complexity_w_pca_f.pickle', "rb").read())
computation_complexity_w_pca_o = pickle.loads(open('computation_complexity_w_pca_o.pickle', "rb").read())

err = pickle.loads(open('err.pickle', "rb").read())
err_w = pickle.loads(open('err_w.pickle', "rb").read())

err_h_pca_f = pickle.loads(open('err_h_pca_f.pickle', "rb").read())
err_h_pca_o = pickle.loads(open('err_h_pca_o.pickle', "rb").read())
err_w_pca_f = pickle.loads(open('err_w_pca_f.pickle', "rb").read())
err_w_pca_o = pickle.loads(open('err_w_pca_o.pickle', "rb").read())

time_w_pca_f = pickle.loads(open('time_w_pca_f.pickle', "rb").read())
time_h_pca_f = pickle.loads(open('time_h_pca_f.pickle', "rb").read())
time_w_pca_o = pickle.loads(open('time_w_pca_o.pickle', "rb").read())
time_h_pca_o = pickle.loads(open('time_h_pca_o.pickle', "rb").read())

# plt.figure(1)
# plt.style.use("ggplot")
# plt.plot(range(len(computation_complexity[0])), computation_complexity[0], marker='o')
# plt.plot(range(len(computation_complexity[1])), computation_complexity[1], marker='v')
# plt.plot(range(len(computation_complexity[2])), computation_complexity[2], marker='x')
# plt.plot(range(len(computation_complexity[3])), computation_complexity[3], marker='*')
# plt.plot(range(len(computation_complexity[4])), computation_complexity[4], marker='')
# plt.plot(range(len(computation_complexity[5])), computation_complexity[5], marker='.')
# plt.grid()
# # plt.ylim([0, 1])
# plt.title("Accuracy of training set")
# plt.xlabel("Number of neurons in the hidden layer")
# plt.ylabel("Accuracy")
# # plt.legend(['Method 1', 'Method 2', 'Method 3', 'Method 4'], loc="best")
# plt.show()
# # plt.savefig('rbf1.jpg')

# plt.figure(1)
# plt.style.use("ggplot")
# (fig, ax) = plt.subplots(3, 2, figsize=(26, 30))
# count = 0
# loop over the loss names
# for i in range(3):
#     for j in range(2):
#         # plot the loss for both the training and validation data
#         title = "Data{}".format(count)
#         ax[i, j].set_title(title, fontweight='bold', fontsize=20)
#         ax[i, j].set_xlabel("Number of 0s in Reconstructed DCT Coefficient Matrix #", fontweight='bold', fontsize=20)
#         ax[i, j].set_ylabel("Computational Complexity (x 1e6)", fontweight='bold', fontsize=20)
#         lns1 = ax[i, j].plot(range(len(computation_complexity[count])),
#                              [cc / 1e6 for cc in computation_complexity[count]],
#                              marker='o', label='Computational Complexity')
#         ax[i, j].grid(False)
#         ax[i, j].tick_params(labelsize=20)
#         ax2 = ax[i, j].twinx()
#         lns2 = ax2.plot(range(len(storage_complexity[count])), [sc / 1e7 for sc in storage_complexity[count]],
#                         'b', label='Storage Complexity')
#         ax2.set_ylabel('Storage Complexity (x 1e7 bits)', fontweight='bold', fontsize=20)
#         ax2.grid(False)
#         ax2.tick_params(labelsize=20)
#         lns = lns1 + lns2
#         labs = [ll.get_label() for ll in lns]
#         ax[i, j].legend(lns, labs, loc=0, fontsize=20)
#         count += 1
# for i in range(3):
#     for j in range(2):
#         # plot the loss for both the training and validation data
#         title = "Data{}".format(count)
#         ax[i, j].set_title(title, fontweight='bold', fontsize=20)
#         ax[i, j].set_xlabel("Number of 0s in Reconstructed DCT Coefficient Matrix #", fontweight='bold', fontsize=20)
#         ax[i, j].set_ylabel("Saved bits (x 1e6)", fontweight='bold', fontsize=20)
#         ax[i, j].plot(range(len(storage_complexity[count])),
#                       [(64 * 4 * 64 * 384 * 4 - sc) / 1e6 for sc in storage_complexity[count]],
#                       label='H', marker='v')
#         ax[i, j].plot(range(len(storage_complexity_w[count])),
#                       [(64 * 2 * 64 * 384 * 4 - sc) / 1e6 for sc in storage_complexity_w[count]],
#                       'b', label='W')
#         ax[i, j].grid(False)
#         ax[i, j].tick_params(labelsize=20)
#         ax[i, j].legend(['H', 'W'], loc=0, fontsize=20)
#         count += 1
# fig.tight_layout(pad=2)
# plt.savefig('output/dct_2.png')
# plt.savefig('output/dct_2.eps')
# for i in range(3):
#     for j in range(2):
#         # plot the loss for both the training and validation data
#         title = "Data{}".format(count)
#         ax[i, j].set_title(title, fontweight='bold', fontsize=20)
#         ax[i, j].set_xlabel("Number of 0s in Reconstructed DCT Coefficient Matrix #", fontweight='bold', fontsize=20)
#         ax[i, j].set_ylabel("Error (dB)", fontweight='bold', fontsize=20)
#         ax[i, j].plot(range(len(err[count])), err[count],
#                       label='H', marker='*')
#         ax[i, j].plot(range(len(err_w[count])), err_w[count], 'b', label='W')
#         ax[i, j].grid(False)
#         ax[i, j].tick_params(labelsize=20)
#         ax[i, j].legend(['H', 'W'], loc=0, fontsize=20)
#         ax[i, j].set_ylim(bottom=-350, top=0)
#         count += 1
# fig.tight_layout(pad=2)
# # plt.show()
# plt.savefig('output/dct_err.png')
# plt.savefig('output/dct_err.eps')
# plt.figure(1)
# plt.style.use("ggplot")
# labels = ['Normal PCA', 'Fast PCA']
# x = range(len(labels))  # the label locations
# width = 0.3  # the width of the bars
# fig, ax = plt.subplots()
# rects1 = ax.bar([xx - width/2 for xx in x],
#                 [(64 * 4 * 4 * 384 * 64 - storage_complexity_h_pca_o[0]) / 1e6,
#                  (64 * 4 * 4 * 384 * 64 - storage_complexity_h_pca_f[0]) / 1e6],
#                 width, label='H')
# rects2 = ax.bar([xx + width/2 for xx in x],
#                 [(64 * 2 * 4 * 384 * 64 - storage_complexity_w_pca_o[0]) / 1e6,
#                  (64 * 2 * 4 * 384 * 64 - storage_complexity_w_pca_f[0]) / 1e6],
#                 width, label='W')
# ax.set_ylabel('(x 1e6)', fontsize=12)
# ax.set_title('Saved bits', fontsize=12)
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.grid(False)
# ax.tick_params(labelsize=12)
# ax.legend(loc=1, fontsize=12)
# ax.bar_label(rects1, padding=3, fontsize=12)
# ax.bar_label(rects2, padding=3, fontsize=12)
# fig.tight_layout()
# # plt.show()
# plt.savefig('output/pca_4.png')
# plt.savefig('output/pca_4.eps')

plt.figure(1)
plt.style.use("ggplot")
labels = ['H', 'W']
x = range(len(labels))  # the label locations
width = 0.3  # the width of the bars
fig, ax = plt.subplots(1, 2, figsize=(13, 7))
rects1 = ax[0].bar([xx - width / 2 for xx in x],
                   [min(min(computation_complexity)) / 1e6,
                    min(min(computation_complexity_w)) / 1e6],
                   width, label='DCT')
rects2 = ax[0].bar([xx + width / 2 for xx in x],
                   [computation_complexity_h_pca_o[0] / 1e6,
                    computation_complexity_w_pca_f[0] / 1e6],
                   width, label='PCA')
ax[0].set_ylabel('(x 1e6)', fontsize=12)
ax[0].set_title('Computational Complexity', fontsize=12)
ax[0].set_xticks(x)
ax[0].set_xticklabels(labels)
ax[0].grid(False)
ax[0].tick_params(labelsize=12)
ax[0].legend(loc=1, fontsize=12)
ax[0].bar_label(rects1, padding=3, fontsize=12)
ax[0].bar_label(rects2, padding=3, fontsize=12)

rects1 = ax[1].bar([xx - width / 2 for xx in x],
                   [(64 * 4 * 4 * 384 * 64) / min(min(storage_complexity)),
                    (64 * 2 * 4 * 384 * 64) / min(min(storage_complexity_w))],
                   width, label='DCT')
rects2 = ax[1].bar([xx + width / 2 for xx in x],
                   [(64 * 4 * 4 * 384 * 64) / storage_complexity_h_pca_o[0],
                    (64 * 2 * 4 * 384 * 64) / storage_complexity_w_pca_o[0]],
                   width, label='PCA')
ax[1].set_ylabel('Compression Ratio', fontsize=12)
ax[1].set_title('Storage Efficiency', fontsize=12)
ax[1].set_xticks(x)
ax[1].set_xticklabels(labels)
ax[1].grid(False)
ax[1].tick_params(labelsize=12)
ax[1].legend(loc=1, fontsize=12)
ax[1].bar_label(rects1, padding=3, fontsize=12)
ax[1].bar_label(rects2, padding=3, fontsize=12)
fig.tight_layout()
plt.savefig('output/comp.png')
plt.savefig('output/comp.eps')

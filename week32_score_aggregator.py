# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 06:41:04 2019

@author: PITAHAYA
"""

import os
import numpy as np

folder = os.getcwd()+'/good/'
files = os.listdir(folder)
precs = []
recs =[]
for i in files:
  if 'recall.txt' in i:
    recs.append(i)
  elif 'precision.txt' in i:
    precs.append(i)

precisions = np.zeros((len(precs), 9, 4))
for idx, i in enumerate(precs):
  precisions[idx] = np.loadtxt(folder+i)

recalls = np.zeros((len(recs), 9, 4))
for idx, i in enumerate(recs):
  recalls[idx] = np.loadtxt(folder+i)

precision_medians = np.nanmedian(precisions, axis=0)
precision_5pct = precision_medians - np.nanpercentile(precisions, 5, axis=0)
precision_95pct = np.nanpercentile(precisions, 95, axis=0) - precision_medians
# precision_stds = np.nanstd(precisions, axis=0)
n_precision = len(precs) - np.sum(np.isnan(precisions), axis=0)

recall_medians = np.nanmedian(recalls, axis=0)
recall_5pct = recall_medians - np.nanpercentile(recalls, 5, axis=0)
recall_95pct = np.nanpercentile(recalls, 95, axis=0) - recall_medians
# recall_stds = np.nanstd(recalls, axis=0)
n_recall = len(precs) - np.sum(np.isnan(recalls), axis=0)

precisions_good = precisions.copy()
recalls_good = recalls.copy()

# =============================================================================
# folder = os.getcwd()+'/all_20190924_scores/'
# files = os.listdir(folder)
# precs = []
# recs =[]
# for i in files:
#   if 'recall' in i:
#     recs.append(i)
#   else:
#     precs.append(i)
#
# precisions_good = precisions.copy()
# recalls_good = recalls.copy()
#
# precisions = np.zeros((len(precs), 9, 4))
# for idx, i in enumerate(precs):
#   precisions[idx] = np.loadtxt(folder+i)
#
# recalls = np.zeros((len(recs), 9, 4))
# for idx, i in enumerate(recs):
#   recalls[idx] = np.loadtxt(folder+i)
#
# precision_medians2 = np.nanmedian(precisions, axis=0)
# precision_5pct2 = precision_medians2 - np.nanpercentile(precisions, 5, axis=0)
# precision_95pct2 = np.nanpercentile(precisions, 95, axis=0) - precision_medians2
# # precision_stds = np.nanstd(precisions, axis=0)
# n_precision2 = len(precs) - np.sum(np.isnan(precisions), axis=0)
#
# recall_medians2 = np.nanmedian(recalls, axis=0)
# recall_5pct2 = recall_medians2 - np.nanpercentile(recalls, 5, axis=0)
# recall_95pct2 = np.nanpercentile(recalls, 95, axis=0) - recall_medians2
# # recall_stds = np.nanstd(recalls, axis=0)
# n_recall2 = len(precs) - np.sum(np.isnan(recalls), axis=0)
#
# new_rows = [7]
# for r in new_rows:
#   n_precision[r] = n_precision2[r]
#   n_recall[r] = n_recall2[r]
#   precision_medians[r] = precision_medians2[r]
#   recall_medians[r] = recall_medians2[r]
#   precision_5pct[r] = precision_5pct2[r]
#   precision_95pct[r] = precision_95pct2[r]
#   recall_5pct[r] = recall_5pct2[r]
#   recall_95pct[r] = recall_95pct2[r]
# =============================================================================


# =============================================================================
# # wanted = [0,1,2,3,4,5,6,7,8]
# wanted = [0, 3, 7]
# recall_medians = recall_medians[wanted]
# recall_5pcts = recall_5pct[wanted]
# recall_95pcts = recall_95pct[wanted]
# # recall_stds = recall_stds[wanted]
# precision_medians = precision_medians[wanted]
# precision_5pcts = precision_5pct[wanted]
# precision_95pcts = precision_95pct[wanted]
# # precision_stds = precision_stds[wanted]
#
# np.savetxt('recall_medians.txt', recall_medians)
# np.savetxt('recall_5pcts.txt', recall_5pcts)
# np.savetxt('recall_95pcts.txt', recall_95pcts)
# # np.savetxt('recall_stds.txt', recall_stds)
# np.savetxt('precision_medians.txt', precision_medians)
# np.savetxt('precision_5pcts.txt', precision_5pcts)
# np.savetxt('precision_95pcts.txt', precision_95pcts)
# # np.savetxt('precision_stds.txt', precision_stds)
#
# #%%
# import matplotlib.pyplot as plt
# import matplotlib
# matplotlib.rcParams.update({'errorbar.capsize': 5})
#
# x = np.array(wanted)+1
#
# plt.figure()
# plt.errorbar(x, precision_medians.T[1], [precision_5pcts.T[1], precision_95pcts.T[1]], label='Noise, 5-95pctile')
# plt.errorbar(x, precision_medians.T[2], [precision_5pcts.T[2], precision_95pcts.T[2]], label='Fundamental, 5-95pctile')
# plt.errorbar(x, precision_medians.T[3], [precision_5pcts.T[3], precision_95pcts.T[3]], label='1st-order, 5-95pctile')
# plt.xlabel('# Stations')
# plt.ylabel('Precision: TruePos / (TruePos + FalsePos)')
# plt.legend()
# plt.ylim(0, 1)
# plt.savefig('precision.pdf', bbox='tight')
# plt.show()
#
# plt.figure()
# plt.errorbar(x, recall_medians.T[1], [recall_5pcts.T[1], recall_95pcts.T[1]], label='Noise, 5-95pctile')
# plt.errorbar(x, recall_medians.T[2], [recall_5pcts.T[2], recall_95pcts.T[2]], label='Fundamental, 5-95pctile')
# plt.errorbar(x, recall_medians.T[3], [recall_5pcts.T[3], recall_95pcts.T[3]], label='1st-order, 5-95pctile')
# plt.xlabel('# Stations')
# plt.ylabel('Recall: TruePos / (TruePos + FalseNeg)')
# plt.legend(loc='lower right')
# plt.ylim(0, 1)
# plt.savefig('recall.pdf', bbox='tight')
# plt.show()
#
# #%% histogram time
#
# nbins = 30
# # bad_rows = np.array(new_rows) + 1
#
# bad_rows = []
# plt.figure(figsize=(6,18))
# for i in [1,4,8]:
#   plt.subplot(int(str(91)+str(i)))
#   if i not in bad_rows:
#     plt.hist(precisions_good[:,i-1,1], bins=nbins)
#   else:
#     plt.hist(precisions[:,i-1,1], bins=nbins)
#   plt.ylabel(str(i) + ' Prec. Noise')
#
# plt.figure(figsize=(6,18))
# for i in [1,4,8]:
#   plt.subplot(int(str(91)+str(i)))
#   if i not in bad_rows:
#     plt.hist(precisions_good[:,i-1,2], bins=nbins)
#   else:
#     plt.hist(precisions[:,i-1,2], bins=nbins)
#   plt.ylabel(i)
#   plt.ylabel(str(i) + ' Prec. Fund.')
#
# plt.figure(figsize=(6,18))
# for i in [1,4,8]:
#   plt.subplot(int(str(91)+str(i)))
#   if i not in bad_rows:
#     plt.hist(precisions_good[:,i-1,3], bins=nbins)
#   else:
#     plt.hist(precisions[:,i-1,3], bins=nbins)
#   plt.ylabel(i)
#   plt.ylabel(str(i) + ' Prec. 1st-order.')
#
# #%% recalls
# plt.figure(figsize=(6,18))
# for i in [1,4,8]:
#   plt.subplot(int(str(91)+str(i)))
#   if i not in bad_rows:
#     plt.hist(recalls_good[:,i-1,1], bins=nbins)
#   else:
#     plt.hist(recalls[:,i-1,1], bins=nbins)
#   plt.ylabel(str(i) + ' Recall Noise')
#
# plt.figure(figsize=(6,18))
# for i in [1,4,8]:
#   plt.subplot(int(str(91)+str(i)))
#   if i not in bad_rows:
#     plt.hist(recalls_good[:,i-1,2], bins=nbins)
#   else:
#     plt.hist(recalls[:,i-1,2], bins=nbins)
#   plt.ylabel(i)
#   plt.ylabel(str(i) + ' Recall Fund.')
#
# plt.figure(figsize=(6,18))
# for i in [1,4,8]:
#   plt.subplot(int(str(91)+str(i)))
#   if i not in bad_rows:
#     plt.hist(recalls_good[:,i-1,3], bins=nbins)
#   else:
#     plt.hist(recalls[:,i-1,3], bins=nbins)
#   plt.ylabel(i)
#   plt.ylabel(str(i) + ' Recall 1st-order.')
# =============================================================================


#%% top 50% chart
best_n = len(precisions)//2
precisions = precisions[:,0:8,:]#[:,[0,3,7],:]
recalls = recalls[:,0:8,:]

a = set(np.argsort(precisions[:,0,-1])[-5:])
b = set(np.argsort(recalls[:,0,-1])[-5:])
print(a)
print(b)

wanted_idx = list(a.intersection(b))[-1]
print(wanted_idx)

print(precs[wanted_idx])

precisions_top10 = np.zeros((best_n, 8, 4))
recalls_top10 = np.zeros((best_n, 8, 4))
for i in range(1,8+1):
  precisions_top10[:,i-1:i,:] = precisions[np.argsort(precisions[:,i-1,1], axis=0)[-best_n:]][:,i-1:i,:]
  recalls_top10[:,i-1:i,:] = recalls[np.argsort(recalls[:,i-1,1], axis=0)[-best_n:]][:,i-1:i,:]

precisions = precisions_top10
recalls = recalls_top10

precision_medians = np.nanmedian(precisions, axis=0)
precision_5pct = precision_medians - np.nanpercentile(precisions, 5, axis=0)
precision_95pct = np.nanpercentile(precisions, 95, axis=0) - precision_medians
# precision_stds = np.nanstd(precisions, axis=0)
n_precision = len(precs) - np.sum(np.isnan(precisions), axis=0)

recall_medians = np.nanmedian(recalls, axis=0)
recall_5pct = recall_medians - np.nanpercentile(recalls, 5, axis=0)
recall_95pct = np.nanpercentile(recalls, 95, axis=0) - recall_medians
# recall_stds = np.nanstd(recalls, axis=0)
n_recall = len(precs) - np.sum(np.isnan(recalls), axis=0)

precisions_good = precisions.copy()
recalls_good = recalls.copy()

# =============================================================================
# folder = os.getcwd()+'/all_20190924_scores/'
# files = os.listdir(folder)
# precs = []
# recs =[]
# for i in files:
#   if 'recall' in i:
#     recs.append(i)
#   else:
#     precs.append(i)
#
# precisions_good = precisions.copy()
# recalls_good = recalls.copy()
#
# precisions = np.zeros((len(precs), 9, 4))
# for idx, i in enumerate(precs):
#   precisions[idx] = np.loadtxt(folder+i)
#
# recalls = np.zeros((len(recs), 9, 4))
# for idx, i in enumerate(recs):
#   recalls[idx] = np.loadtxt(folder+i)
#
# precision_medians2 = np.nanmedian(precisions, axis=0)
# precision_5pct2 = precision_medians2 - np.nanpercentile(precisions, 5, axis=0)
# precision_95pct2 = np.nanpercentile(precisions, 95, axis=0) - precision_medians2
# # precision_stds = np.nanstd(precisions, axis=0)
# n_precision2 = len(precs) - np.sum(np.isnan(precisions), axis=0)
#
# recall_medians2 = np.nanmedian(recalls, axis=0)
# recall_5pct2 = recall_medians2 - np.nanpercentile(recalls, 5, axis=0)
# recall_95pct2 = np.nanpercentile(recalls, 95, axis=0) - recall_medians2
# # recall_stds = np.nanstd(recalls, axis=0)
# n_recall2 = len(precs) - np.sum(np.isnan(recalls), axis=0)
#
# new_rows = [7]
# for r in new_rows:
#   n_precision[r] = n_precision2[r]
#   n_recall[r] = n_recall2[r]
#   precision_medians[r] = precision_medians2[r]
#   recall_medians[r] = recall_medians2[r]
#   precision_5pct[r] = precision_5pct2[r]
#   precision_95pct[r] = precision_95pct2[r]
#   recall_5pct[r] = recall_5pct2[r]
#   recall_95pct[r] = recall_95pct2[r]
# =============================================================================


wanted = [0,1,2,3,4,5,6,7]
recall_medians = recall_medians[wanted]
recall_5pcts = recall_5pct[wanted]
recall_95pcts = recall_95pct[wanted]
# recall_stds = recall_stds[wanted]
precision_medians = precision_medians[wanted]
precision_5pcts = precision_5pct[wanted]
precision_95pcts = precision_95pct[wanted]
# precision_stds = precision_stds[wanted]

np.savetxt('recall_medians-20200123.txt', recall_medians)
np.savetxt('recall_5pcts-20200123.txt', recall_5pcts)
np.savetxt('recall_95pcts-20200123.txt', recall_95pcts)
# np.savetxt('recall_stds.txt', recall_stds)
np.savetxt('precision_medians-20200123.txt', precision_medians)
np.savetxt('precision_5pcts-20200123.txt', precision_5pcts)
np.savetxt('precision_95pcts-20200123.txt', precision_95pcts)
# np.savetxt('precision_stds.txt', precision_stds)

#%%
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'errorbar.capsize': 5})

x = np.array([1,2,3,4,5,6,7,8])#np.array(wanted)+1

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

plt.figure()
plt.errorbar(x, precision_medians.T[1], [precision_5pcts.T[1], precision_95pcts.T[1]], label='Noise, 5-95pctile')
plt.errorbar(x, precision_medians.T[2], [precision_5pcts.T[2], precision_95pcts.T[2]], label='Fundamental, 5-95pctile')
plt.errorbar(x, precision_medians.T[3], [precision_5pcts.T[3], precision_95pcts.T[3]], label='1st-order, 5-95pctile')
plt.xlabel('# Stations')
plt.ylabel('Precision: TruePos / (TruePos + FalsePos)')
plt.legend()
plt.ylim(0.6, 1.2)
plt.savefig('precision-20200123.pdf', bbox_inches='tight', transparent=True)
plt.show()

plt.figure()
plt.errorbar(x, recall_medians.T[1], [recall_5pcts.T[1], recall_95pcts.T[1]], label='Noise, 5-95pctile')
plt.errorbar(x, recall_medians.T[2], [recall_5pcts.T[2], recall_95pcts.T[2]], label='Fundamental, 5-95pctile')
plt.errorbar(x, recall_medians.T[3], [recall_5pcts.T[3], recall_95pcts.T[3]], label='1st-order, 5-95pctile')
plt.xlabel('# Stations')
plt.ylabel('Recall: TruePos / (TruePos + FalseNeg)')
plt.legend(loc='lower right')
plt.ylim(0.4, 1.2)
plt.savefig('recall-20200123.pdf', bbox_inches='tight', transparent=True)
plt.show()

print("samples:", best_n)
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 14:41:41 2019

@author: PITAHAYA
"""


#%% imports
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageOps

#%% save the filename lists for train and test
with open('files_lb.txt', 'r') as f:
  files = f.readlines()
train_idx = np.load('train_idx_1000.npy')
test_idx = np.load('test_idx_1000.npy')

train_idx = np.sort(train_idx)
test_idx = np.sort(test_idx)

train_files = []
test_files = []
for i in train_idx:
  train_files.append(files[i][:-5])
for i in test_idx:
  test_files.append(files[i][:-5])

with open('train_filenames.txt', 'w') as f:
  for i in train_files:
    f.write(i + '\n')
with open('test_filenames.txt', 'w') as f:
  for i in test_files:
    f.write(i + '\n')

#%% one station
print(datetime.now())
import os
savepath = 'ascii/one_lgb_pred/'
path = '2020-01-21-033402/LB_models/1/'
files_lb = os.listdir(path)
realfiles_lb = files_lb
# realfiles_lb = []
# for i in range(len(files_lb)):
#   if '-preds.png' in files_lb[i]:# and 'preds-preds' not in files_lb[i]:
#     realfiles_lb.append(files_lb[i])
N = len(realfiles_lb)

Y = np.zeros((N, 64, 64, 3))
for i in tqdm(range(N)):
  Y[i] = np.asarray(Image.open(path + realfiles_lb[i]))

# flatten Y into onehot
Y[np.where((Y==[255,255,255]).all(axis=3))] = [255,0,0] # white to red
Y = np.argmax(Y, axis=-1) # to integer onehot

# saving
savex = np.array(np.repeat(np.arange(1,64+1), 64), dtype=int) # for saving purposes
savey = np.array(np.tile(np.arange(1, 64+1), 64), dtype=int)

for i in tqdm(range(N)):
  tempY = np.zeros((1, 64, 64)) # make 1 subplots for each "square"
  tempY[0] = Y[i][:, :]

  tempY = tempY.reshape((1, 64*64)) # make into a one-line array

  np.savetxt(savepath+realfiles_lb[i][:-12]+'.txt', np.array([savex, savey, tempY[0]]).T, fmt='%d')

#%% multi station
print(datetime.now())
import os
savepath = 'ascii/multi_lgb_pred/'
path = '2020-01-21-033402/LB_models/8/'
files_lb = os.listdir(path)
realfiles_lb = files_lb
# realfiles_lb = []
# for i in range(len(files_lb)):
#   if '-preds.png' in files_lb[i]:# and 'preds-preds' not in files_lb[i]:
#     realfiles_lb.append(files_lb[i])
N = len(realfiles_lb)

Y = np.zeros((N, 64, 64, 3))
for i in tqdm(range(N)):
  Y[i] = np.asarray(Image.open(path + realfiles_lb[i]))

# flatten Y into onehot
Y[np.where((Y==[255,255,255]).all(axis=3))] = [255,0,0] # white to red
Y = np.argmax(Y, axis=-1) # to integer onehot

# saving
savex = np.array(np.repeat(np.arange(1,64+1), 64), dtype=int) # for saving purposes
savey = np.array(np.tile(np.arange(1, 64+1), 64), dtype=int)

for i in tqdm(range(N)):
  tempY = np.zeros((1, 64, 64)) # make 1 subplots for each "square"
  tempY[0] = Y[i][:, :]

  tempY = tempY.reshape((1, 64*64)) # make into a one-line array

  np.savetxt(savepath+realfiles_lb[i][:-12]+'.txt', np.array([savex, savey, tempY[0]]).T, fmt='%d')

#%% multi station input 64
print(datetime.now())
import os
savepath = 'ascii/64_input/'
# path = '2019-12-16-131337/LB_models/8/'
# files_lb = os.listdir(path)
# realfiles_lb = files_lb
# realfiles_lb = []
# for i in range(len(files_lb)):
#   if '-preds.png' in files_lb[i]:# and 'preds-preds' not in files_lb[i]:
#     realfiles_lb.append(files_lb[i])
with open('test_filenames.txt', 'r') as f:
  realfiles_lb = f.readlines()
N = len(realfiles_lb)

test_idx = np.sort(np.load('test_idx_1000.npy'))
Y = np.load('trainX_LB.npy')[test_idx,:,:,0]

# saving
savex = np.array(np.repeat(np.arange(1,64+1), 64), dtype=int) # for saving purposes
savey = np.array(np.tile(np.arange(1, 64+1), 64), dtype=int)

for i in tqdm(range(N)):
  tempY = np.zeros((1, 64, 64)) # make 1 subplots for each "square"
  tempY[0] = Y[i][:, :]

  tempY = tempY.reshape((1, 64*64)) # make into a one-line array

  np.savetxt(savepath+realfiles_lb[i][:-5]+'.txt', np.array([savex, savey, tempY[0]]).T, fmt='%d')


#%% multi station input 101
# =============================================================================
# print(datetime.now())
# import os
# savepath = 'ascii/101_input/'
# path = '2019-09-30-084600/LB_models/9/'
# files_lb = os.listdir(path)
# realfiles_lb = files_lb
# # realfiles_lb = []
# # for i in range(len(files_lb)):
# #   if '-preds.png' in files_lb[i]:# and 'preds-preds' not in files_lb[i]:
# #     realfiles_lb.append(files_lb[i])
# N = len(realfiles_lb)
#
# Y = np.load('ascii/trainX_LB_101.npy')[:1500,:,:,0]
# Y = Y.reshape((1500, 41*101))
#
# # saving
# savex = np.array(np.repeat(np.arange(1,41+1), 101), dtype=int) # for saving purposes
# savey = np.array(np.tile(np.arange(1, 101+1), 41), dtype=int)
#
# for i in tqdm(range(N)):
#   np.savetxt(savepath+realfiles_lb[i][:-12]+'.txt', np.array([savex, savey, Y[i]]).T, fmt='%d')
#
# # for i in tqdm(range(N)):
# #   tempY = np.zeros((1, 64, 64)) # make 1 subplots for each "square"
# #   tempY[0] = Y[i][:, :]
#
# #   tempY = tempY.reshape((1, 64*64)) # make into a one-line array
#
# #   np.savetxt(savepath+realfiles_lb[i][:-12]+'.txt', np.array([savex, savey, tempY[0]]).T, fmt='%d')
# =============================================================================


#%% multi station ground truth
print(datetime.now())
import os
savepath = 'ascii/multi_truth/'
path = 'actuals/'
files_lb = os.listdir(path)
realfiles_lb = files_lb
# for i in range(len(files_lb)):
#   if '-actual.png' in files_lb[i]:# and 'preds-preds' not in files_lb[i]:
#     realfiles_lb.append(files_lb[i])
N = len(realfiles_lb)

Y = np.zeros((N, 64, 64, 3))
for i in tqdm(range(N)):
  Y[i] = np.asarray(Image.open(path + realfiles_lb[i]))

# flatten Y into 5340 * (41*101)
Y[np.where((Y==[255,255,255]).all(axis=3))] = [255,0,0] # white to red
Y = np.argmax(Y, axis=-1) # to integer onehot

# saving
savex = np.array(np.repeat(np.arange(1,64+1), 64), dtype=int) # for saving purposes
savey = np.array(np.tile(np.arange(1, 64+1), 64), dtype=int)

for i in tqdm(range(N)):
  tempY = np.zeros((1, 64, 64)) # make 1 subplots for each "square"
  tempY[0] = Y[i][:, :]

  tempY = tempY.reshape((1, 64*64)) # make into a one-line array

  np.savetxt(savepath+realfiles_lb[i][:-12]+'.txt', np.array([savex, savey, tempY[0]]).T, fmt='%d')

#%%
# =============================================================================
# #%% cnn predictions with LB test data (1500)
# print(datetime.now())
# import os
# savepath = 'ascii/cnn_lgb_pred/'
# path = '../week6/testtrained_images/'
# files_lb = os.listdir(path)
# realfiles_lb = []
# for i in range(len(files_lb)):
#   if True:#'-actual.png' in files_lb[i]:# and 'preds-preds' not in files_lb[i]:
#     realfiles_lb.append(files_lb[i])
# N = len(realfiles_lb)
#
# Y = np.zeros((N, 41,101,3))
# for i in tqdm(range(N)):
#   Y[i] = np.asarray(Image.open(path + realfiles_lb[i]))
#
# # flatten Y into 1500 * (41*101)
# Y[np.where((Y==[255,255,255]).all(axis=3))] = [255,0,0] # white to red
# Y = np.argmax(Y, axis=-1) # to integer onehot
# Y = Y.reshape((N, 41*101))
#
# # saving
# savex = np.array(np.repeat(np.arange(1,41+1), 101), dtype=int) # for saving purposes
# savey = np.array(np.tile(np.arange(1, 101+1), 41), dtype=int)
#
# for i in tqdm(range(N)):
#   np.savetxt(savepath+realfiles_lb[i][1:-14]+'.txt', np.array([savex, savey, Y[i]]).T, fmt='%d')
#
# #%% cnn predictions with only synthetic data (5340)
# print(datetime.now())
# import os
# savepath = 'ascii/cnn_pred/'
# path = '../week6/lb_pred_images/'
# files_lb = os.listdir(path)
# realfiles_lb = []
# for i in range(len(files_lb)):
#   if '-preds.png' in files_lb[i] and 'preds-preds' not in files_lb[i]:
#     realfiles_lb.append(files_lb[i])
# N = len(realfiles_lb)
#
# Y = np.zeros((N, 41,101,3))
# for i in tqdm(range(N)):
#   Y[i] = np.asarray(Image.open(path + realfiles_lb[i]))
#
# # flatten Y into 1500 * (41*101)
# Y[(np.max(Y, axis=-1)<255)] = 255 # "if you aren't 100% sure, then you are noise"
# Y[np.where((Y==[255,255,255]).all(axis=3))] = [255,0,0] # white to red
# Y = np.argmax(Y, axis=-1) # to integer onehot
# Y = Y.reshape((N, 41*101))
#
# # saving
# savex = np.array(np.repeat(np.arange(1,41+1), 101), dtype=int) # for saving purposes
# savey = np.array(np.tile(np.arange(1, 101+1), 41), dtype=int)
#
# for i in tqdm(range(N)):
#   np.savetxt(savepath+'img_dsp_'+realfiles_lb[i][15:-10]+'.txt', np.array([savex, savey, Y[i]]).T, fmt='%d')
# =============================================================================


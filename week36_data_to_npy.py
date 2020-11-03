# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:02:16 2019

@author: PITAHAYA
"""

#%% imports
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.utils import to_categorical
from datetime import datetime
from PIL import Image, ImageOps
import os

#%% data train
# N = 10000
filelenx = 64
fileleny = 64

#%% data imports
print(datetime.now())
# path = 'imgsynwlabel/'
# # train_savepath = 'train_images_wk8/img_disp_'
# # test_savepath = 'test_images_wk8/img_disp_'

# trainX_synth = np.zeros((N, filelenx, fileleny, 9), dtype='uint8')
# trainY_synth = np.zeros((N, filelenx, fileleny, 9*3), dtype='uint8')

# for i in tqdm(range(1, N+1)):
  # x_img = np.zeros((filelenx, fileleny, 9), dtype='uint8') # 9 stations per image
  # y_img = np.zeros((filelenx, fileleny, 9*3), dtype='uint8') # 3 channels for each RGB
  # imported = np.loadtxt(path+'img_disp_'+str(i).rjust(5,'0')+'.dat',
                        # delimiter='\t')
  # x_img[:,:,0] = imported.T[2].reshape(filelenx, fileleny)
  # y_img[:,:,0*3:(0*3 + 3)] = to_categorical(imported.T[3].reshape(filelenx, fileleny))
  # for j in range(1,8+1):
    # imported = np.loadtxt(path+'img_disp_'+str(i).rjust(5,'0')+'_'+str(j)+'.dat',
                        # delimiter='\t')
    # x_img[:,:,j] = imported.T[2].reshape(filelenx, fileleny)
    # y_img[:,:,j*3:(j*3 + 3)] = to_categorical(imported.T[3].reshape(filelenx, fileleny))

  # trainX_synth[i-1:i] = x_img
  # trainY_synth[i-1:i] = y_img
  # # np.save(train_savepath + str(i).rjust(5,'0'), x_img.astype('uint8'))
  # # np.save(test_savepath + str(i).rjust(5,'0'), y_img.astype('uint8'))

# np.save('trainX_synth.npy', trainX_synth)
# np.save('trainY_synth.npy', trainY_synth)

# trainY_synth_short = trainY_synth[:,:,:,0:3]
# np.save('trainY_synth_short.npy', trainY_synth_short)

#%% LB data generation
file = 'LongBeachRealData_SurroundingStalist_9.dat'

train_datapath = 'lb_train_wk36/'
test_datapath = 'lb_test_wk36/'
filenames = np.loadtxt(file, delimiter=' ', dtype='str')

with open("files_lb.txt", "r") as text_file:
  files = text_file.readlines()
for i in range(len(files)):
  files[i] = files[i][:-6]

N = len(filenames)
trainX = np.zeros((N, filelenx, fileleny, 9), dtype='uint8')
trainY = np.zeros((N, filelenx, fileleny, 3), dtype='uint8')

for i in tqdm(range(N)):
  assert files[i] == filenames[i][0][:-4]
  for s in range(9):
    trainX[i:i+1, :, :, s] = np.asarray(ImageOps.invert(Image.open(train_datapath+filenames[i][s][:-4]+'.png')))
    if s==0:
      trainY[i:i+1, :, :, s*3:s*3+3] = np.asarray(ImageOps.invert(Image.open(test_datapath+filenames[i][s][:-4]+'.png')))

np.save('trainX_LB.npy', trainX)
np.save('trainY_LB.npy', trainY)

# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 10:27:12 2019

@author: PITAHAYA
"""

#%% imports
import os
import sys
import torch
from torch.nn import MaxPool2d, Conv2d, ConvTranspose2d, ELU, Dropout
from pytorchtools import EarlyStopping
import matplotlib.pyplot as plt
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=sys.argv[-1]
device = torch.device('cuda')

checkpoint_filename = 'checkpoint'+str(sys.argv[-1])+'.pt'

os.chdir(os.getcwd())

import numpy as np
from tqdm import tqdm
from datetime import datetime
import time
from PIL import Image#, ImageOps

#%% data train params
N = 1000 # number of training samples (limited by the train_idx_xxxx.npy file)
imgsizex = 64
imgsizey = 64
n_epochs = 500
patience = 3

foldername = 'actuals/'
  
with open("files_lb.txt", "r") as text_file:
  files_lb = text_file.readlines()
for i in range(len(files_lb)):
  files_lb[i] = files_lb[i][:-1]

#%% LB
#X = np.load('../week17/trainX_LB.npy')
Y = np.load('trainY_LB.npy')
print(Y.shape)

# y_pred = model.eval(test_loader).transpose((0,2,3,1))

# convert to image
def squash(old_min, old_max, new_min, new_max, val):
  return((val-old_min)/old_max * (new_max-new_min) + new_min)

y_img = np.round(squash(0,1, 0,255, Y)).astype(np.uint8) # predictions
# print(y_img[0])

y_img[np.where((y_img==[255,0,0]).all(axis=3))] = [255,255,255] # red -> white

# %% filtering!
# y_img[(np.max(y_img, axis=-1)<255)] = 255 # "if you aren't 100% sure, then you are noise"
y_img = y_img.astype(np.uint8)
y_mids = y_img.copy()

for j in range(len(Y)):
  if j==0:
    print(y_mids[j].dtype)
  Image.fromarray(y_mids[j]).save(foldername + files_lb[j][:-4] + '-mid.png')
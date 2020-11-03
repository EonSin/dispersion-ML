# -*- coding: utf-8 -*-
"""
Created on Sun May 12 12:01:53 2019

@author: PITAHAYA
"""

#%% imports
import numpy as np
from tqdm import tqdm
from tensorflow.python.keras.utils import to_categorical
from datetime import datetime
from time import time
from PIL import Image, ImageOps

#%% data train
filelenx = 64
fileleny = 64

#%% long beach actual data
import os
print(datetime.now())
path = 'LB_ascii/'

# files = os.listdir(path)
with open("files_lb.txt", "r") as text_file:
  files = text_file.readlines()
for i in range(len(files)):
  files[i] = files[i][:-6] + '.txt'

train_savepath = 'lb_train_wk36/'
test_savepath = 'lb_test_wk36/'

#for i in tqdm(range(1, N+1)):
for filename in tqdm(files):
  imported = np.loadtxt(path + filename, delimiter='\t')
  x_img = imported.T[2].reshape(filelenx, fileleny)
  img = ImageOps.invert(Image.fromarray(x_img.astype('uint8'), 'L'))
  img.save(train_savepath + filename[:-4] + '.png')

  y_img = to_categorical(imported.T[3].reshape(filelenx, fileleny))
  label_img = ImageOps.invert(Image.fromarray(y_img.astype('uint8')))
  label_img.save(test_savepath + filename[:-4] +'.png')
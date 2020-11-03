# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 07:14:41 2019

@author: PITAHAYA
"""

#%% imports
import numpy as np
from tqdm import tqdm
from datetime import datetime
from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

path = 'ascii/64_input/'
files_lb = os.listdir(path)
filename = 'img_dsp_LGB_1034-5083.txt'
title = filename[8:-4]# files_lb[0][8:-4]

in_threshold = 30

#%% transform back to (T, g)
"""
The conversion from original period (T) to x index (x) is as follows,
        if (T>=0.5&&T<2.2)  % T= 0.5, 0.55, 0.6, 0.65, … , 2.1, 2.15

                x=round((T-0.5)/0.05+1);

        elseif (T>=2.2&&T<2.9) % T= 2.2, 2.3, …, 2.7, 2.8

                x=round(35+(T-2.2)/0.1);

        elseif (T>=2.9&&T<3) % T= 2.9, T= 2.95

                x=round(42+(T-2.9)/0.05);

        elseif (T>=3&&T<=5) % T=3, 3.1, 3.2, …, 4.8, 4.9, 5.

                x=round(44+(T-3)/0.1);

The conversion from original group velocity (g) to y index (y) is as follows,

gdt=g-T*0.1-0.35 (detrend).

y=round((gdt+0.3)/0.0127+1); (y from -0.3 to 0.5)

For your problem, you want g and T with given x and y. """
def transform(Z):
  X = Z.copy()
  X[X[:,0]<=34,0] = (X[X[:,0]<=34,0] - 1) * 0.05 + 0.5
  X[np.logical_and(X[:,0]>=35,X[:,0]<=41),0] = (X[np.logical_and(X[:,0]>=35,X[:,0]<=41),0] - 35) * 0.1 + 2.2
  X[np.logical_and(X[:,0]>=42,X[:,0]<=43),0] = (X[np.logical_and(X[:,0]>=42,X[:,0]<=43),0] - 42) * 0.05 + 2.9
  X[X[:,0]>=44,0] = (X[X[:,0]>=44,0] - 44) * 0.1 + 3

  X[:,1] = (X[:,1] - 1) * 0.0127 - 0.3
  X[:,1] = X[:,1] + 0.35 + 0.1*X[:,0] # already transformed X[:,0]

  return X

#%% plotting function
def plot(X, Y, title, savename, raw=False):
  X = transform(X)
  Y = transform(Y)
  Y0 = Y[Y[:,2]==1]
  Y1 = Y[Y[:,2]==2]
  X0 = X[Y[:,2]==1]
  X1 = X[Y[:,2]==2]
  #% plotting, truth 101
  plt.figure()
  plt.scatter(X.T[0], X.T[1], s=np.maximum(0, X.T[2]-in_threshold), facecolors='none', edgecolors='black', label='Input Data', linewidth=0.3)
  if not raw:
    plt.scatter(Y0.T[0], Y0.T[1], c='r', edgecolors='black',label='Fundamental Picks', linewidth=.5, s=np.maximum(0, X0.T[2]-in_threshold))
    plt.scatter(Y1.T[0], Y1.T[1], c='b', edgecolors='black',label='First-order Picks', linewidth=.5, s=np.maximum(0, X1.T[2]-in_threshold))
  plt.xlim(0.5, 5)
  plt.ylim(0.3, 1.0)
  # plt.xlim(0.5, 3)
  # plt.ylim(0, 1)
  plt.legend()
  plt.xlabel('Period (s)')
  plt.ylabel('Group Velocity (km/s)')
  plt.title(title)
  plt.savefig(savename, bbox_inches='tight', transparent=True)
  plt.show()

#%% 64 truth
X = np.loadtxt('ascii/64_input/' + filename)
plot(X,
      np.loadtxt('ascii/multi_truth/' + filename),
      'Ground Truth: ' + title, 'truth_64.pdf')
plot(X,
      np.loadtxt('ascii/one_lgb_pred/' + filename),
      'CNN (sim2real 1 station) Picks: ' + title, 'cnn_sim2real_1.pdf')
plot(X,
      np.loadtxt('ascii/multi_lgb_pred/' + filename),
      'CNN (sim2real 8 station) Picks: ' + title, 'cnn_sim2real_8.pdf')

plot(X,
      np.loadtxt('ascii/multi_lgb_pred/' + filename),
      'Input (X): ' + title, 'input.pdf', raw=True)

#%% just plot inputs
img = X[:,2].reshape((64,64)).astype('uint8')
img = ImageOps.invert(Image.fromarray(img))
img.save('grey_input_for_ppt.png')

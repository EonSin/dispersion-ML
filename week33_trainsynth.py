# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 23:10:55 2019

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
N = 10000
imgsizex = 64
imgsizey = 64
n_epochs = 500
patience = 3

with open("../week10/files_lb.txt", "r") as text_file:
  files_lb = text_file.readlines()
for i in range(len(files_lb)):
  files_lb[i] = files_lb[i][:-1]

#%% data imports
now = datetime.now()
timename = str(now.microsecond) #datetime.strftime(now, "%Y-%m-%d-%H%M%S")

print(now.microsecond)
torch.manual_seed(now.microsecond)
np.random.seed(2019)

#%% synth train
X_synth = np.load('../week17/trainX_synth.npy')
Y_synth = np.load('../week17/trainY_synth_short.npy')

class NumpyDataset(torch.utils.data.Dataset):
  def __init__(self, data, target, transform=None):
    print(data.shape)
    self.data = torch.from_numpy(data.transpose((0, 3, 1, 2))).float()
    # self.target = torch.from_numpy(target.transpose((0, 3, 1, 2))).float()
    self.target = torch.from_numpy(np.argmax(target, axis=-1)).float()

  def __getitem__(self, index):
    x = self.data[index]
    y = self.target[index]
    return x, y

  def __len__(self):
    return len(self.data)

class Model():
  def __init__(self, network, optimizer, model_path, model_name):
    self.network = network
    self.optimizer = optimizer
    self.model_path = model_path
    self.model_name = model_name

  def train(self, train_loader, val_loader, patience, n_epochs):
    from torch.autograd import Variable
    early_stopping = EarlyStopping(patience=patience, verbose=True, filename=checkpoint_filename)

    loss = torch.nn.CrossEntropyLoss()#BCEWithLogitsLoss()#CrossEntropyLoss()
    training_start_time = time.time()
    avg_train_loss = []
    avg_val_loss = []

    for epoch in range(n_epochs):
      epoch_train_loss = 0
      epoch_val_loss = 0
      epoch_data_count = 0

      self.network.train()
      for i, data in enumerate(train_loader, 0):
        # Get inputs/outputs and wrap in variable object
        inputs, y_true = data
        inputs, y_true = Variable(
            inputs.to(device)), Variable(
            y_true.to(device, dtype=torch.int64))

        # Set gradients for all parameters to zero
        self.optimizer.zero_grad()

        # Forward pass
        outputs = self.network(inputs)
        #y_true = y_true[:,None,:]

        # Backward pass
        loss_value = loss(outputs, y_true)
        loss_value.backward()

        # Update parameters
        self.optimizer.step()
        epoch_train_loss += loss_value.item()
        epoch_data_count += len(y_true)
      epoch_train_loss /= float(epoch_data_count)
      avg_train_loss.append(epoch_train_loss)

      epoch_data_count = 0
      self.network.eval()
      with torch.no_grad():
        for inputs, y_true in val_loader:

          # Wrap tensors in Variables
          inputs, y_true = Variable(
              inputs.to(device)), Variable(
              y_true.to(device, dtype=torch.int64))

          # Forward pass only
          val_outputs = self.network(inputs)
          val_outputs = torch.sigmoid(val_outputs)
          val_loss = loss(val_outputs, y_true)
          epoch_val_loss += val_loss.item()
          epoch_data_count += len(y_true)

      epoch_val_loss /= float(epoch_data_count)
      print("Validation loss = {:.4e}".format(epoch_val_loss))
      avg_val_loss.append(epoch_val_loss)

      early_stopping(epoch_val_loss, self.network)
      if early_stopping.early_stop:
        print("Early stopping, epoch", epoch)
        break

    self.network.load_state_dict(torch.load(checkpoint_filename))
    # save model to disk under <modelpath>/<modelname>_<minloss>
    torch.save(self, '%s/%s_%s_.h5' % (self.model_path, self.model_name, np.min(avg_val_loss)))

    print(
      "Training finished, took {:.2f}s".format(
          time.time() -
          training_start_time))
    return avg_train_loss, avg_val_loss

  def eval(self, test_loader):
    from torch.autograd import Variable
    testing_start_time = time.time()
    y_pred_all = []

    self.network.eval()
    with torch.no_grad():
      for inputs, _ in test_loader: # y values are NOT USED when testing
        # Wrap tensors in Variables
        inputs = Variable(inputs.to(device))

        # Forward pass only
        val_outputs = self.network(inputs)
        val_outputs = torch.sigmoid(val_outputs)

        # Make predictions
        y_pred = torch.zeros(val_outputs.data.size()).to(device, dtype=torch.int64)

        y_pred = val_outputs#.argmax(axis=1)

        y_pred_all.append(y_pred.cpu().numpy())

    y_pred_all = np.concatenate(y_pred_all)

    print(
        "Testing finished, took {:.2f}s".format(
            time.time() -
            testing_start_time))

    return(y_pred_all)

class UNet(torch.nn.Module):
  def __init__(self, num_channels=9, num_classes=3):
    super(UNet, self).__init__()

    self.elu = ELU()
    self.maxpool = MaxPool2d(kernel_size=2, stride=2)
    self.dropout1 = Dropout(0.1)
    self.dropout2 = Dropout(0.2)
    self.dropout3 = Dropout(0.3)

    self.conv11 = Conv2d(num_channels, 16, kernel_size=3, padding=1)
    self.conv12 = Conv2d(16, 16, kernel_size=3, padding=1)

    self.conv21 = Conv2d(16, 32, kernel_size=3, padding=1)
    self.conv22 = Conv2d(32, 32, kernel_size=3, padding=1)

    self.conv31 = Conv2d(32, 64, kernel_size=3, padding=1)
    self.conv32 = Conv2d(64, 64, kernel_size=3, padding=1)

    self.conv41 = Conv2d(64, 128, kernel_size=3, padding=1)
    self.conv42 = Conv2d(128, 128, kernel_size=3, padding=1)

    self.conv51 = Conv2d(128, 256, kernel_size=3, padding=1)
    self.conv52 = Conv2d(256, 256, kernel_size=3, padding=1)

    self.uconv6 = ConvTranspose2d(256, 128, kernel_size=2, stride=2)
    self.conv61 = Conv2d(256, 128, kernel_size=3, padding=1)
    self.conv62 = Conv2d(128, 128, kernel_size=3, padding=1)

    self.uconv7 = ConvTranspose2d(128, 64, kernel_size=2, stride=2)
    self.conv71 = Conv2d(128, 64, kernel_size=3, padding=1)
    self.conv72 = Conv2d(64, 64, kernel_size=3, padding=1)

    self.uconv8 = ConvTranspose2d(64, 32, kernel_size=2, stride=2)
    self.conv81 = Conv2d(64, 32, kernel_size=3, padding=1)
    self.conv82 = Conv2d(32, 32, kernel_size=3, padding=1)

    self.uconv9 = ConvTranspose2d(32, 16, kernel_size=2, stride=2)
    self.conv91 = Conv2d(32, 16, kernel_size=3, padding=1)
    self.conv92 = Conv2d(16, 16, kernel_size=3, padding=1)

    self.conv93 = Conv2d(16, num_classes, kernel_size=1, padding=0)

  def forward(self, x):
    x = self.conv11(x)
    x = self.elu(x)
    x = self.dropout1(x)
    x = self.conv12(x)
    x1d = self.elu(x)
    x = self.maxpool(x1d)

    x = self.conv21(x)
    x = self.elu(x)
    x = self.dropout1(x)
    x = self.conv22(x)
    x2d = self.elu(x)
    x = self.maxpool(x2d)

    x = self.conv31(x)
    x = self.elu(x)
    x = self.dropout2(x)
    x = self.conv32(x)
    x3d = self.elu(x)
    x = self.maxpool(x3d)

    x = self.conv41(x)
    x = self.elu(x)
    x = self.dropout2(x)
    x = self.conv42(x)
    x4d = self.elu(x)
    x = self.maxpool(x4d)

    x = self.conv51(x)
    x = self.elu(x)
    x = self.dropout3(x)
    x = self.conv52(x)
    x5d = self.elu(x)

    x6u = self.uconv6(x5d)
    x = torch.cat((x4d, x6u), 1)
    x = self.conv61(x)
    x = self.elu(x)
    x = self.dropout2(x)
    x = self.conv62(x)
    x = self.elu(x)

    x7u = self.uconv7(x)
    x = torch.cat((x3d, x7u), 1)
    x = self.conv71(x)
    x = self.elu(x)
    x = self.dropout2(x)
    x = self.conv72(x)
    x = self.elu(x)

    x8u = self.uconv8(x)
    x = torch.cat((x2d, x8u), 1)
    x = self.conv81(x)
    x = self.elu(x)
    x = self.dropout1(x)
    x = self.conv82(x)
    x = self.elu(x)

    x9u = self.uconv9(x)
    x = torch.cat((x1d, x9u), 1)
    x = self.conv91(x)
    x = self.elu(x)
    x = self.dropout1(x)
    x = self.conv92(x)
    x = self.elu(x)

    x = self.conv93(x)
    return x

for i in tqdm(range(1, 0, -1)):#tqdm([8,4,1]):
  print(i, 'neighbours-----', datetime.now())
  X_synth = X_synth[:,:,:,0:i]

  dataset = NumpyDataset(X_synth, Y_synth)

  n_samples = len(dataset)
  n_test = int(0.1*n_samples)
  print(X_synth.shape, Y_synth.shape)
  print(n_samples, n_test)

  indices = list(range(n_samples))

  validation_idx = np.random.choice(indices, size=n_test, replace=False)
  train_idx = list(set(indices) - set(validation_idx))
  np.save(validation_idx, 'validation_idx.npy')
  np.save(np.array(train_idx), 'train_idx.npy')

  from torch.utils.data.sampler import SubsetRandomSampler
  train_sampler = SubsetRandomSampler(train_idx)
  validation_sampler = SubsetRandomSampler(validation_idx)

  train_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=32,
      shuffle=False,
      sampler=train_sampler,
  )
  val_loader = torch.utils.data.DataLoader(
      dataset,
      batch_size=256,
      shuffle=False,
      sampler=validation_sampler
  )

  unet = UNet(num_channels=i, num_classes=3).to(device)
  optimizer = torch.optim.Adam(unet.parameters())

  model = Model(unet, optimizer, model_path='synth_models/'+str(i),
                model_name='synth-stacked-'+timename)
  train_loss, valid_loss = model.train(train_loader, val_loader, patience, n_epochs)

  if i==1:
    y_pred = model.eval(val_loader).transpose((0,2,3,1))
    
    N = len(y_pred)
    y_pred = np.argmax(y_pred, axis=-1)
    y_pred = np.reshape(y_pred, (N, 64*64))
    
    y_actual = Y_synth[validation_idx]
    y_actual = np.argmax(y_actual, axis=-1)
    y_actual = np.reshape(y_actual, (N, 64*64))

    precision = np.zeros(3)
    recall = np.zeros(3)

    truepos = np.zeros((3, N))
    trueneg = np.zeros((3, N))
    falsepos = np.zeros((3, N))
    falseneg = np.zeros((3, N))

    for k in range(3):
      if (y_pred!=0).sum() < 10:
        precision[k] = np.nan
        recall[k] = np.nan
      else:
        truepos[k] = np.bitwise_and(y_actual==y_pred, y_pred==k).sum(axis=-1)
        trueneg[k] = np.bitwise_and(y_actual==y_pred, y_pred!=k).sum(axis=-1)
        falsepos[k] = np.bitwise_and(y_actual!=y_pred, y_pred==k).sum(axis=-1)
        falseneg[k] = np.bitwise_and(y_actual!=y_pred, y_pred!=k).sum(axis=-1)
        precision[k] = np.sum(truepos[k]) / np.sum(truepos[k] + falsepos[k])
        recall[k] = np.sum(truepos[k]) / np.sum(truepos[k] + falseneg[k])
    
    print(precision)
    print(recall)
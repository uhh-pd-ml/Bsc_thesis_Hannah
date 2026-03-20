#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto encoder and training
"""


# =============================================================================
# I M P O R T S
# =============================================================================

import argparse

import numpy as np
import matplotlib.pyplot as plt

import h5py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmeta.utils.data import Dataset
from torch.utils.data import DataLoader


# =============================================================================
# C O N S T A N T S   A N D   A R G P A R S E
# =============================================================================

parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs',
                    help='Number of epochs',
                    type=int,
                    default=20)
parser.add_argument('-l', '--learningrate',
                    help='Learning rate',
                    type=float,
                    default=0.001)
parser.add_argument('-d', '--datapath',
                    help='Path to full dataset.',
                    type=str)
parser.add_argument('-s', '--savepath',
                    help='Directory to save outputs.',
                    type=str)
parser.add_argument('-p', '--pixels',
                    help='Number of pixels per side.',
                    type=int,
                    default='33')
args = parser.parse_args()

_N_EPOCHS = args.epochs
_LEARNING_RATE = args.learningrate
_DATA_PATH = args.datapath
_SAVE_PATH = args.savepath
_N_PIX = args.pixels

_PARAMS = {'batch_size': 256,
           'shuffle': False}
_DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# =============================================================================
# C L A S S E S
# =============================================================================

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.e1a = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.e1b = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.e1c = nn.MaxPool2d(3) # 33 -> 11

        self.e2a = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.e2b = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.e2c = nn.Conv2d(5, 1, kernel_size=3, padding=1)

        self.e3a = nn.Linear(11*11, 100)
        self.e3b = nn.Linear(100, 32)
        
        self.d1a = nn.Linear(32, 100)
        self.d1b = nn.Linear(100, 11*11)
        
        self.d2a = nn.Conv2d(1, 5, kernel_size=3, padding=1)
        self.d2b = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.d2c = nn.Upsample(scale_factor=3) # 11 -> 33

        self.d3a = nn.Conv2d(5, 5, kernel_size=3, padding=1)
        self.d3b = nn.Conv2d(5, 1, kernel_size=3, padding=1)

    def encode(self,x):

        x = x.view(-1, 1, _N_PIX, _N_PIX)

        x = F.elu(self.e1a(x))
        x = F.elu(self.e1b(x))
        x = self.e1c(x)

        x = F.elu(self.e2a(x))
        x = F.elu(self.e2b(x))
        x = F.elu(self.e2c(x))

        x = x.view(-1, 11*11)

        x = F.relu(self.e3a(x))
        x = F.relu(self.e3b(x))

        return x

    def decode(self, z):

        z = F.relu(self.d1a(z))
        z = F.relu(self.d1b(z))

        z = z.view(-1, 1, 11, 11)

        z = F.elu(self.d2a(z))
        z = F.elu(self.d2b(z))
        z = self.d2c(z)

        z = F.elu(self.d3a(z))
        z = F.elu(self.d3b(z))

        return z

    def forward(self, x):

        z = self.encode(x)
        r = self.decode(z)

        r = r.view(-1, _N_PIX*_N_PIX)

        return r


class H5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_file = h5py.File(h5_path, 'r')
        self.j1_images = self.h5_file['j1_images']
        # self.j2_images = self.h5_file['j2_images']

    def __len__(self):
        return len(self.j1_images)

    def __getitem__(self, index):
        dataset = self.j1_images[index]
        data = torch.from_numpy(dataset[:])

        return data


# =============================================================================
# F U N C T I O N S
# =============================================================================

def loss_function(x, y):
    x = x.view(-1, _N_PIX*_N_PIX)
    MSE = (x - y)**2
    return MSE.sum()


def save_validation_images_2d(array_of_images):
    fig, axs = plt.subplots(5, _N_EPOCHS+1, figsize=(30, 8))

    images_no = 0
    for epoch in range(-1, _N_EPOCHS):
        for i in range(5):
            axs[i, epoch+1].imshow(array_of_images[images_no], cmap='hot')
            if i == 0:
                if epoch == -1:
                    axs[i, epoch+1].set_title('Input')
                else:
                    axs[i, epoch+1].set_title('Epoch {}'.format(epoch))
            images_no += 1

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                        hspace=0.4, wspace=0.3)
    plt.savefig('validation_images.png', dpi=100)


def save_validation_images_1d(array_of_images):
    fig, axs = plt.subplots(5, _N_EPOCHS+1, figsize=(30, 8))

    images_no = 0
    for epoch in range(-1, _N_EPOCHS):
        for i in range(5):
            x = range(_N_PIX)
            y = array_of_images[images_no].sum(axis=0)
            axs[i, epoch+1].plot(x, y)
            axs[i, epoch+1].set_ylim(0, 0.8)
            if i == 0:
                if epoch == -1:
                    axs[i, epoch+1].set_title('Input')
                else:
                    axs[i, epoch+1].set_title('Epoch {}'.format(epoch))
            images_no += 1

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.05, top=0.9,
                        hspace=0.4, wspace=0.3)
    plt.savefig('validation_images.png', dpi=100)


def main():

    # @ whoever works with this after me:
    # Adjust the splitting of the dataset into training/testing/validation to your
    # dataset's size. I had quite good results with approx. 80:10:10.
    # Dont forget to adjust the normalization of the losses, too!
    # I was too lazy to automate this but should be quite easy... :D

    dataset_train = H5Dataset(_DATA_PATH)
    dataset_train.j1_images = dataset_train.j1_images[:202898]

    dataloader_train = DataLoader(dataset_train, **_PARAMS)

    dataset_test = H5Dataset(_DATA_PATH)
    dataset_test.j1_images = dataset_test.j1_images[202898:228260]

    dataset_validate = H5Dataset(_DATA_PATH)
    dataset_validate.j1_images = dataset_validate.j1_images[228260:]

    train_losses, test_losses, validate_losses = np.array([]), np.array([]), np.array([])

    for epoch in range(_N_EPOCHS):
    
        # training
        num_batches = len(dataloader_train)
        train_loss = 0
    
        for batch in dataloader_train:
            _OPTIMIZER.zero_grad()

            batch = batch.to(_DEVICE).float()

            pred = _MODEL(batch)
            loss = loss_function(batch, pred)
            train_loss += loss.item()
            loss.backward()
            _OPTIMIZER.step()

        train_loss /= (num_batches*64)

        # testing
        with torch.no_grad():
            data = torch.from_numpy(dataset_test.j1_images).to(_DEVICE).float()
            pred = _MODEL(data)
            test_loss = loss_function(data, pred).item()
            test_loss /= 25362
    
        # validation
        with torch.no_grad():
            data = torch.from_numpy(dataset_validate.j1_images).to(_DEVICE).float()
            pred = _MODEL(data)
            validate_loss = loss_function(data, pred).item()
            validate_loss /= 25362
        # _SCHEDULER.step(validate_loss)

        # saving
        train_losses = np.append(train_losses, train_loss)
        test_losses = np.append(test_losses, test_loss)
        validate_losses = np.append(validate_losses, validate_loss)
    
        torch.save(_MODEL.state_dict(), '{}/{}.pt'.format(_SAVE_PATH, epoch))


    #save losses
    np.save('{}/losses_validate.npy'.format(_SAVE_PATH), validate_losses)
    np.save('{}/losses_test.npy'.format(_SAVE_PATH), test_losses)
    np.save('{}/losses_train.npy'.format(_SAVE_PATH), train_losses)


    # plot losses
    plt.figure()
    plt.plot(range(_N_EPOCHS), train_losses, label='training loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('{}/losses_train.png'.format(_SAVE_PATH))

    plt.figure()
    plt.plot(range(_N_EPOCHS), validate_losses, label='validation loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.savefig('{}/losses_validate.png'.format(_SAVE_PATH))

    print(np.argmin(validate_losses))


# =============================================================================
# M A I N
# =============================================================================

if __name__ == '__main__':

    _MODEL = AutoEncoder().to(_DEVICE)
    _OPTIMIZER = torch.optim.Adam(_MODEL.parameters(), lr=_LEARNING_RATE)

    # If you want to use a scheduler, uncomment the following
    # and line 257: _SCHEDULER.step(validate_loss):

    # _SCHEDULER = torch.optim.lr_scheduler.ReduceLROnPlateau(_OPTIMIZER, patience=5)

    # execute main function
    main()

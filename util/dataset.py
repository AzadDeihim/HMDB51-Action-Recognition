import os
import torch
from torch.utils import data
from torchvision import transforms
from PIL import Image
import numpy as np
import re
import random
import shutil
import numpy as np
import cv2
import os
from os import listdir
from os.path import isfile, join


class TrainDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.labels = kwargs['labels']
        self.data_path = kwargs['data_path']

    def __len__(self):
        'Denotes the total number of samples'
        return len(os.listdir(self.data_path + 'train_data/'))

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.load(self.data_path + 'train_data/' + str(index) + '.pt')
        y = torch.load(self.data_path + 'train_labels/' + str(index) + '.pt')
        return X, y, index


class TestDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.labels = kwargs['labels']
        self.data_path = kwargs['data_path']

    def __len__(self):
        'Denotes the total number of samples'
        return len(os.listdir(self.data_path + 'test_data/'))

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.load(self.data_path + 'test_data/' + str(index) + '.pt')
        y = torch.load(self.data_path + 'test_labels/' + str(index) + '.pt')
        return X, y, index


class ValDataset(data.Dataset):

    def __init__(self, **kwargs):
        self.labels = kwargs['labels']
        self.data_path = kwargs['data_path']

    def __len__(self):
        'Denotes the total number of samples'
        return len(os.listdir(self.data_path + 'val_data/'))

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.load(self.data_path + 'val_data/' + str(index) + '.pt')
        y = torch.load(self.data_path + 'val_labels/' + str(index) + '.pt')
        return X, y, index




class TrainDatasetMask(data.Dataset):

    def __init__(self, **kwargs):
        self.labels = kwargs['labels']
        self.data_path = kwargs['data_path']

    def __len__(self):
        'Denotes the total number of samples'
        return len(os.listdir(self.data_path + 'train_data/'))

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.load(self.data_path + 'train_mask/' + str(index) + '.pt')
        y = torch.load(self.data_path + 'train_labels_mask/' + str(index) + '.pt')
        return X, y, index


class TestDatasetMask(data.Dataset):

    def __init__(self, **kwargs):
        self.labels = kwargs['labels']
        self.data_path = kwargs['data_path']

    def __len__(self):
        'Denotes the total number of samples'
        return len(os.listdir(self.data_path + 'test_data/'))

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.load(self.data_path + 'test_mask/' + str(index) + '.pt')
        y = torch.load(self.data_path + 'test_labels_mask/' + str(index) + '.pt')
        return X, y, index


class ValDatasetMask(data.Dataset):

    def __init__(self, **kwargs):
        self.labels = kwargs['labels']
        self.data_path = kwargs['data_path']

    def __len__(self):
        'Denotes the total number of samples'
        return len(os.listdir(self.data_path + 'val_data/'))

    def __getitem__(self, index):
        'Generates one sample of data'
        # Load data and get label
        X = torch.load(self.data_path + 'val_mask/' + str(index) + '.pt')
        y = torch.load(self.data_path + 'val_labels_mask/' + str(index) + '.pt')
        return X, y, index



import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
import sys

from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from models import *

import torch.nn as nn
import torch.nn.functional as F
import torch

# GPU Setting
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor

Enc1 = Encoder(in_channels=3, dim = 64, n_downsample = 2, n_residual= 3, style_dim= 8).to(device) # dim, n_downsample, n_residual, style_dim
Dec1 = Decoder(out_channels=3, dim = 64, n_upsample = 2, n_residual= 3, style_dim= 8).to(device)
Enc2 = Encoder(in_channels=3, dim = 64, n_downsample = 2, n_residual= 3, style_dim= 8).to(device)
Dec2 = Decoder(out_channels=3, dim = 64, n_upsample = 2, n_residual= 3, style_dim= 8).to(device)

Enc1.load_state_dict(torch.load("output/pkl/Enc1_70.pkl"))
Dec1.load_state_dict(torch.load("output/pkl/Dec1_70.pkl"))
Enc2.load_state_dict(torch.load("output/pkl/Enc2_70.pkl"))
Dec2.load_state_dict(torch.load("output/pkl/Dec2_70.pkl"))

Enc1.eval()
Enc2.eval()
Dec1.eval()
Dec2.eval()

# Dataset

image = os.listdir('dataset/testA/')
print(len(image))
class MUNITData(Dataset):
    def __init__(self,trainA_path,transform):
        self.trainA_path = trainA_path
        self.transform = transform
        
    def __len__(self):
        return len(image)
    
    def trainA(self,trainA_path):
        trainA = Image.open('dataset/testA/'+trainA_path).convert('RGB')
        trainA = trainA.resize((420,296))
        trainA = self.transform(trainA)
        return trainA
    
    
    def __getitem__(self,index):
        trainA = self.trainA(self.trainA_path[index])
            
        return {'trainA': trainA}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# DataSet, DataLoader
Dataset = MUNITData(trainA_path=image,transform=transform)
dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1,
                                          shuffle=True, num_workers=0,drop_last=True)

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(dataloader))
    img_samples = None
    for img1 in imgs["trainA"]:
    # Create copies of image
        X1 = img1.unsqueeze(0).repeat(8, 1, 1, 1)
        X1 = Variable(X1.type(Tensor))
        # Get random style codes
        s_code = Variable(torch.randn(8,8, 1, 1).cuda())
        s_code = Variable(Tensor(s_code))
        # Generate samples
        c_code_1, _ = Enc1(X1)
        X12 = Dec2(c_code_1, s_code)
        # Concatenate samples horisontally
        X12 = torch.cat([x for x in X12.data.cpu()], -1)
        img_sample = torch.cat((img1, X12), -1).unsqueeze(0)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)

    img_save_dir = os.path.join('output', 'testimg')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)

    save_image(img_samples, "%s/%s.png" % ('output/testimg',batches_done), nrow=5, normalize=True)


for i in range(0,87):
    batches_done = i
    sample_images(batches_done)
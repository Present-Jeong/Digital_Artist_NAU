# https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/munit/munit.py
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
os.environ["CUDA_VISIBLE_DEVICES"]="3"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Tensor = torch.cuda.FloatTensor

# Dataset

image = os.listdir('dataset/trainA/')
label = os.listdir('dataset/trainB/')

print(len(image),len(label))
# dataset
class MUNITData(Dataset):
    def __init__(self,trainA_path,trainB_path,transform):
        self.trainA_path = trainA_path
        self.trainB_path = trainB_path
        self.transform = transform
        
    def __len__(self):
        return len(label)
    
    def trainA(self,trainA_path):
        # trainA = sitk.ReadImage('dataset/cezanne2photo/trainA/'+trainA_path)
        trainA = Image.open('dataset/trainA/'+trainA_path).convert('RGB')
        # IMG_SIZE = 512
        trainA = trainA.resize((420,296))
        # trainA = sitk.GetArrayFromImage(trainA)
        # IMG_SIZE = 256
        # trainA = cv2.resize(trainA,(IMG_SIZE,IMG_SIZE))
        # trainA = trainA.astype('float')
        trainA = self.transform(trainA)
        return trainA
    
    def trainB(self,trainB_path):
        trainB = Image.open('dataset/trainB/'+trainB_path).convert('RGB')
        # IMG_SIZE = 512
        trainB = trainB.resize((420,296))
        # trainB = sitk.ReadImage('dataset/cezanne2photo/trainB/'+trainB_path)
        # trainB = sitk.GetArrayFromImage(trainB)
        # trainB = cv2.resize(trainB,(IMG_SIZE,IMG_SIZE))
        # trainB = trainB.astype('float')
        trainB = self.transform(trainB)
        return trainB
    
    def __getitem__(self,index):
        trainA = self.trainA(self.trainA_path[index])
        trainB = self.trainB(self.trainB_path[index])
            
        return {'trainA': trainA,
                'trainB': trainB}

# Transform
transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.RandomCrop(512), 
    transforms.RandomHorizontalFlip(),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# DataSet, DataLoader
Dataset = MUNITData(trainA_path=image,trainB_path=label,transform=transform)
dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1,
                                          shuffle=True, num_workers=0,drop_last=True)

print(Dataset[40]['trainA'].shape)

criterion_recon = torch.nn.L1Loss().to(device)

# Loss weights
lambda_gan = 1
lambda_id = 10
lambda_style = 1
lambda_cont = 1
lambda_cyc = 0

# Initialize encoders, generators and discriminators
Enc1 = Encoder(in_channels=3, dim = 64, n_downsample = 2, n_residual= 3, style_dim= 8).to(device) # dim, n_downsample, n_residual, style_dim
Dec1 = Decoder(out_channels=3, dim = 64, n_upsample = 2, n_residual= 3, style_dim= 8).to(device)
Enc2 = Encoder(in_channels=3, dim = 64, n_downsample = 2, n_residual= 3, style_dim= 8).to(device)
Dec2 = Decoder(out_channels=3, dim = 64, n_upsample = 2, n_residual= 3, style_dim= 8).to(device)

D1 = MultiDiscriminator().to(device)
D2 = MultiDiscriminator().to(device)

Enc1.apply(weights_init_normal)
Dec1.apply(weights_init_normal)
Enc2.apply(weights_init_normal)
Dec2.apply(weights_init_normal)
D1.apply(weights_init_normal)
D2.apply(weights_init_normal)

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(Enc1.parameters(), Dec1.parameters(), Enc2.parameters(), Dec2.parameters()),
    lr=2e-4,
    betas=(0.5, 0.999),)
optimizer_D1 = torch.optim.Adam(D1.parameters(), lr=2e-4, betas=(0.5, 0.999))
optimizer_D2 = torch.optim.Adam(D2.parameters(), lr=2e-4, betas=(0.5, 0.999))

n_epochs = 300
decay_epoch = 100

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
)
lr_scheduler_D1 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D1, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
)
lr_scheduler_D2 = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D2, lr_lambda=LambdaLR(n_epochs, 0, decay_epoch).step
)

def sample_images(batches_done):
    """Saves a generated sample from the validation set"""
    imgs = next(iter(dataloader))
    img_samples = None
    for img1, img2 in zip(imgs["trainA"], imgs["trainB"]):
        # Create copies of image
        X1 = img1.unsqueeze(0).repeat(8, 1, 1, 1)
        X1 = Variable(X1.type(Tensor))
        # Get random style codes
        s_code = np.random.uniform(-1, 1, (8,8))
        s_code = Variable(Tensor(s_code))
        # Generate samples
        c_code_1, _ = Enc1(X1)
        X12 = Dec2(c_code_1, s_code)
        # Concatenate samples horisontally
        X12 = torch.cat([x for x in X12.data.cpu()], -1)
        img_sample = torch.cat((img1, X12), -1).unsqueeze(0)
        # Concatenate with previous samples vertically
        img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
    
    img_save_dir = os.path.join('output', 'image')
    if not os.path.exists(img_save_dir):
        os.makedirs(img_save_dir)
    save_image(img_samples, "%s/%s.png" % ('output/image', batches_done), nrow=5, normalize=True)


#  Training

# Adversarial ground truths

valid = 1
fake = 0

save_dir = "./"
loss_arr = []
prev_time = time.time()
for epoch in range(n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        X1 = batch['trainA'].to(torch.float).to(device)
        X2 = batch['trainB'].to(torch.float).to(device)
        # Sampled style codes
        style_1 = Variable(torch.randn(X1.size(0), 8, 1, 1).type(Tensor))
        style_2 = Variable(torch.randn(X1.size(0), 8, 1, 1).type(Tensor))

        # -------------------------------
        #  Train Encoders and Generators
        # -------------------------------

        optimizer_G.zero_grad()

        # Get shared latent representation
        c_code_1, s_code_1 = Enc1(X1)
        c_code_2, s_code_2 = Enc2(X2)

        # Reconstruct images
        X11 = Dec1(c_code_1, s_code_1)
        X22 = Dec2(c_code_2, s_code_2)

        # Translate images
        X21 = Dec1(c_code_2, style_1)
        X12 = Dec2(c_code_1, style_2)

        # Cycle translation
        c_code_21, s_code_21 = Enc1(X21)
        c_code_12, s_code_12 = Enc2(X12)
        X121 = Dec1(c_code_12, s_code_1) if lambda_cyc > 0 else 0
        X212 = Dec2(c_code_21, s_code_2) if lambda_cyc > 0 else 0

        # Losses
        loss_GAN_1 = lambda_gan * D1.compute_loss(X21, valid)
        loss_GAN_2 = lambda_gan * D2.compute_loss(X12, valid)
        loss_ID_1 = lambda_id * criterion_recon(X11, X1)
        loss_ID_2 = lambda_id * criterion_recon(X22, X2)
        loss_s_1 = lambda_style * criterion_recon(s_code_21, style_1)
        loss_s_2 = lambda_style * criterion_recon(s_code_12, style_2)
        loss_c_1 = lambda_cont * criterion_recon(c_code_12, c_code_1.detach())
        loss_c_2 = lambda_cont * criterion_recon(c_code_21, c_code_2.detach())
        loss_cyc_1 = lambda_cyc * criterion_recon(X121, X1) if lambda_cyc > 0 else 0
        loss_cyc_2 = lambda_cyc * criterion_recon(X212, X2) if lambda_cyc > 0 else 0

        # Total loss
        loss_G = (
            loss_GAN_1
            + loss_GAN_2
            + loss_ID_1
            + loss_ID_2
            + loss_s_1
            + loss_s_2
            + loss_c_1
            + loss_c_2
            + loss_cyc_1
            + loss_cyc_2
        )

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator 1
        # -----------------------

        optimizer_D1.zero_grad()

        loss_D1 = D1.compute_loss(X1, valid) + D1.compute_loss(X21.detach(), fake)

        loss_D1.backward()
        optimizer_D1.step()

        # -----------------------
        #  Train Discriminator 2
        # -----------------------

        optimizer_D2.zero_grad()

        loss_D2 = D2.compute_loss(X2, valid) + D2.compute_loss(X12.detach(), fake)

        loss_D2.backward()
        optimizer_D2.step()

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] ETA: %s"
            % (epoch, n_epochs, i, len(dataloader), (loss_D1 + loss_D2).item(), loss_G.item(), time_left)
        )

        # If at sample interval save image
        if batches_done % 500 == 0:
            sample_images(batches_done)

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D1.step()
    lr_scheduler_D2.step()
        
    dataset_name = 'output/pkl'
    pkl_save_dir = os.path.join('output', 'pkl')
    if not os.path.exists(pkl_save_dir):
        os.makedirs(pkl_save_dir)
    
    if epoch % 10 == 0:
        # Save model checkpoints
        torch.save(Enc1.state_dict(), "%s/Enc1_%d.pkl" % (dataset_name, epoch))
        torch.save(Dec1.state_dict(), "%s/Dec1_%d.pkl" % (dataset_name, epoch))
        torch.save(Enc2.state_dict(), "%s/Enc2_%d.pkl" % (dataset_name, epoch))
        torch.save(Dec2.state_dict(), "%s/Dec2_%d.pkl" % (dataset_name, epoch))
        torch.save(D1.state_dict(), "%s/D1_%d.pkl" % (dataset_name, epoch))
        torch.save(D2.state_dict(), "%s/D2_%d.pkl" % (dataset_name, epoch))
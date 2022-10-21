import cv2
import tensorflow_datasets as tfds
import tensorflow as tf
import os
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torch
from torch.utils.data import dataloader
from tqdm import tqdm
from torchvision import transforms, datasets
from torchvision.transforms import ToPILImage
import torch.nn as nn
import torch.utils as utils
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import torchvision.utils as v_utils
import torchvision.datasets as dset
from torch.optim import lr_scheduler
from torch.utils.data import Dataset, DataLoader
import SimpleITK as sitk
import torchvision.transforms as transforms
import torch.nn.functional as F

import argparse
import itertools
import dicom2nifti

plt.imshow('dataset/trainA/1195540_90.npy',cmap='gray')

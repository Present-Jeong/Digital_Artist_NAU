import argparse
import sys
import os
from PIL import Image

import torchvision.transforms as transforms
import torchvision.utils as v_utils
from torch.utils.data import Dataset,DataLoader
from torch.autograd import Variable
import torch

from CycleGAN import Generator
import random

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

###### Definition of variables ######
# Networks
netG_A2B = Generator(3, 3).to(device)
netG_B2A = Generator(3, 3).to(device)

# Load state dicts
netG_A2B.load_state_dict(torch.load('output/pkl/netG_A2B.pkl'))
netG_B2A.load_state_dict(torch.load('output/pkl/netG_B2A.pkl'))

# Set model's test mode
netG_A2B.eval()
netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
input_A = Tensor(1, 3, 296, 420) # (batchsize, output channel, size, size)
input_B = Tensor(1, 3, 296, 420)

# Dataset loader

image = os.listdir('dataset/cezanne2photo/testA/')
label = os.listdir('dataset/cezanne2photo/trainB')
print(len(image),len(label))
# for i in range(0,10000):
#     random.shuffle(label)
#     if 'busan' == label[0]:
#         break
#     else :
#         continue
# label = label[:58]
# print(len(label))
# print(label)
class CycleGanData_test(Dataset):
    def __init__(self,testA_path,testB_path,transform):
        self.testA_path = testA_path
        self.testB_path = testB_path
        self.transform = transform
        
    def __len__(self):
        return len(image)
    
    def testA(self,testA_path):
        testA = Image.open('dataset/cezanne2photo/testA/'+testA_path).convert('RGB')
        testA = testA.resize((420,296))
        testA = self.transform(testA)
        return testA
    
    def testB(self,testB_path):
        testB = Image.open('dataset/cezanne2photo/trainB/'+testB_path).convert('RGB')
        testB = testB.resize((420,296))
        testB = self.transform(testB)
        return testB
    
    def __getitem__(self,index):
        testA = self.testA(self.testA_path[index])
        testB = self.testB(self.testB_path[index])
            
        return {'testA': testA,
                'testB': testB}

transform = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Resize((256,256), Image.BICUBIC),
    transforms.Normalize(mean=(0.5,), std=(0.5,)),
])

# DataSet, DataLoader
Dataset = CycleGanData_test(testA_path=image,testB_path=label,transform=transform)
# print(Dataset[2]['trainA'])

dataloader = torch.utils.data.DataLoader(Dataset, batch_size=1,
                                          shuffle=True, num_workers=0,drop_last=True)

###### Testing######

# Create output dirs if they don't exist
if not os.path.exists('output/A_image'):
    os.makedirs('output/A_image')
if not os.path.exists('output/B_image'):
    os.makedirs('output/B_image')

for i, batch in enumerate(dataloader):
    # Set model input
    real_A = Variable(input_A.copy_(batch['testA']))
    real_B = Variable(input_B.copy_(batch['testB']))

    # Generate output
    fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    # Save image files
    batch_tensorA = torch.cat((real_A, fake_B), dim=2)
    batch_tensorB = torch.cat((real_B, fake_A), dim=2)
    
    grid_imgA = v_utils.make_grid(batch_tensorA) # padding = 1, nrow = 4
    grid_imgB = v_utils.make_grid(batch_tensorB)

    v_utils.save_image(grid_imgA, 'output/A_image/%04d.png' % (i+1))
    v_utils.save_image(grid_imgB, 'output/B_image/%04d.png' % (i+1))

    sys.stdout.write('\rGenerated images %04d of %04d'%(i+1, len(dataloader)))
sys.stdout.write('\n')

###################################
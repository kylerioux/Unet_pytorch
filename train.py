# this python file contains code to train the U-net
# modifying code from https://medium.com/analytics-vidhya/pytorch-implementation-of-semantic-segmentation-for-single-class-from-scratch-81f96643c98c

# visualization library
import cv2
from matplotlib import pyplot as plt
# data storing library
import numpy as np
import pandas as pd
# torch libraries
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader, Dataset, sampler
# architecture and data split library
from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp #-- this was to get u-net architecture
from random import *
# augmenation library
#from albumentations import (HorizontalFlip, ShiftScaleRotate, Normalize, Resize, Compose, GaussNoise)
#from albumentations.pytorch import ToTensor

from torchvision.transforms import (Normalize, RandomHorizontalFlip,ToTensor,Compose)

# others
import os
import pdb
import time
import warnings
import random
from tqdm import tqdm_notebook as tqdm
import concurrent.futures
# warning print supression
warnings.filterwarnings("ignore")

from UNet import UNet # get the U-net model 

# image directories
#train_img_dir = 'images/processed_images/train'
#train_img_masks_dir = 'images/processed_images/train_masks'

#kaggle csv
df=pd.read_csv('kaggle_data/train_masks_kaggle.csv')

# kaggle locations of images
train_img_dir='kaggle_data/carvana_my_unet/proc/train-572-grayscale'
train_img_masks_dir='kaggle_data/carvana_my_unet/proc/train_masks-388'



#read csv file of image names
#df=pd.read_csv('image_names.csv')

#mean, std = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225) # for rbg images, related to imagenet
mean, std = 0,255 # for grayscale images

# during traning eval phase make a list of transforms to be used.
# inputs "phase", mean, std
# outputs list of transformations
# def get_transform(phase,mean,std):
#     list_trans=[]
#     if phase=='train':
#         list_trans.extend([RandomHorizontalFlip(p=0.5)])
#     list_trans.extend([Normalize(mean=mean,std=std),ToTensor()])  #normalizing the data & then converting to tensors
#     list_trans=Compose(list_trans)
#     return list_trans

import torchvision.transforms.functional as TF
from PIL import Image

# when dataloader requests samples using index it fetches input image and target mask,
# applys transformation and returns it
class CityDataset(Dataset):
    def __init__(self, df, train_img_dir, train_img_masks_dir, mean, std, phase):
        #self.fname = df['images'].values.tolist()
        self.fname = df['img'].values.tolist() #kaggle one
        self.train_img_dir = train_img_dir
        self.train_img_masks_dir = train_img_masks_dir
        self.mean = mean
        self.std = std
        self.phase = phase
        self.isTraining = 0.
        #self.transform = get_transform(phase,mean,std)
        # self.transform = Compose([
        #         RandomHorizontalFlip(p=0.5),
        #         ToTensor(),
        #         Normalize(mean=mean, std=std)
        #     ])
        # self.transform2 = Compose([
        #         ToTensor(),
        #         Normalize(mean=mean, std=std)
        #     ])
    def transform(self,image,mask,isTraining):
        # Random horizontal flipping
        if random.random() > 0.5 and isTraining==1.:
            image = TF.hflip(image)
            mask = TF.hflip(mask)
            
        # Transform to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        image = TF.normalize(image,mean=mean, std=std)
        #mask = TF.normalize(mask,mean=mean, std=std)

        return image, mask

    def __getitem__(self, idx):
        name = self.fname[idx] # this is the [img] colmn in the csv file
        
        img_name_path = os.path.join(self.train_img_dir,name)
        mask_name_path = img_name_path.split('.')[0].replace('train-572-grayscale','train_masks-388')+'_mask_mask.png' #kaggle dirs
        #mask_name_path = os.path.join(self.train_img_masks_dir,name)


        img = cv2.imread(img_name_path, cv2.IMREAD_GRAYSCALE) #added to make this grayscale similar to below line
        #img = cv2.imread(img_name_path) #non grayscale (rgb) version 
        mask = cv2.imread(mask_name_path, cv2.IMREAD_GRAYSCALE)

        im_pil = Image.fromarray(img) #convert to PIL image instead of opencv
        mask_pil = Image.fromarray(mask)
        
        if(self.phase=='train'):
            isTraining = 1.

        img_aug,mask_aug = self.transform(im_pil,mask_pil,isTraining)

        
            
        torch.set_printoptions(profile="full")
        print("mask after aug is: ")
        print(mask_aug)
        torch.set_printoptions(profile="default")



        #augmentation = self.transform(image=img, mask=mask)
        #img_aug = augmentation['image'] #[1,572,572] type:Tensor
        #mask_aug = augmentation['mask'] #[1,572,572] type:Tensor

        #print(mask_aug.shape)

        return img_aug, mask_aug

    def __len__(self):
        return len(self.fname)

#divide data into train and val and return the dataloader depending upon train or val phase.
def CityDataloader(df,train_img_dir,train_img_masks_dir,mean,std,phase,batch_size,num_workers):
    if(phase=='train'):
        df_train, df_valid = train_test_split(df, test_size=0.2, random_state=69)
        df = df_train if phase == 'train' else df_valid
        for_loader = CityDataset(df, train_img_dir, train_img_masks_dir, mean, std, phase)
        dataloader = DataLoader(for_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True)

    else:
        df_valid = df
        for_loader = CityDataset(df, train_img_dir, train_img_masks_dir, mean, std, phase)
        dataloader = DataLoader(for_loader,batch_size=batch_size,num_workers=num_workers,pin_memory=True)

    return dataloader

#dice scores
def dice_score(pred,targs):
    pred = (pred>0).float()
    return 2. * (pred*targs).sum() / (pred+targs).sum()

#initialize a empty list when Scores is called, append the list with dice scores
#for every batch, at the end of epoch calculates mean of the dice scores
class Scores:
    def __init__(self, phase, epoch):
        self.base_dice_scores = []

    def update(self, targets, outputs):
        probs = outputs
        dice = dice_score(probs, targets)
        self.base_dice_scores.append(dice)

    def get_metrics(self):
        dice = np.mean(self.base_dice_scores)         
        return dice

#return dice score for epoch when called
def epoch_log(epoch_loss, measure):
    #logging the metrics at the end of an epoch
    dices = measure.get_metrics()    
    dice = dices                       
    print("Loss: %0.4f |dice: %0.4f" % (epoch_loss, dice))
    return dice

class Trainer(object):
    def __init__(self,model):
        self.num_workers = 4
        self.batch_size = {'train':1, 'val':1}
        self.accumulation_steps = 4//self.batch_size['train']
        self.lr=1e-3
        self.num_epochs = 100
        self.phases = ['train','val']
        self.best_loss = float('inf')
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model.to(self.device)
        cudnn.benchmark = True
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer,mode='min',patience=3,verbose=True)
        #print("before dataloader called")
        self.dataloaders = {phase: CityDataloader(df, train_img_dir,train_img_masks_dir, mean, std,
                                                phase=phase,batch_size=self.batch_size[phase],
                                                num_workers=self.num_workers) for phase in self.phases}

        self.losses = {phase:[] for phase in self.phases}
        self.dice_score = {phase:[] for phase in self.phases}

    def forward(self,inp_images,tar_mask):
        inp_images = inp_images.to(self.device)
        tar_mask = tar_mask.to(self.device)
        #inp_images = inp_images.unsqueeze(0) # adding dimension for batch (s/b 1,1,572,572)
        pred_mask = self.net(inp_images)
        # so pred mask is tensor [1,2,388,388], target mask is [1,1,388,388]
        # target mask is the one from my preprocessing, pred is what came out of my Unet
        print("pred mask is: ")
        print(pred_mask)
        loss = self.criterion(pred_mask,tar_mask)
        return loss, pred_mask

    def iterate(self,epoch,phase):
        measure = Scores(phase, epoch)
        start = time.strftime("%H:%M:%S")
        print (f"Starting epoch: {epoch} | phase:{phase} | ':{start}")
        batch_size = self.batch_size[phase]
        self.net.train(phase=="train")
        dataloader = self.dataloaders[phase]
        running_loss = 0.0
        total_batches = len(dataloader)
        self.optimizer.zero_grad()
        for itr,batch in enumerate(dataloader):
            images,mask_target = batch
            loss, pred_mask = self.forward(images,mask_target)
            loss = loss/self.accumulation_steps
            if phase == 'train':
                loss.backward()
                if (itr+1) % self.accumulation_steps ==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            running_loss += loss.item()
            pred_mask = pred_mask.detach().cpu()
            measure.update(mask_target,pred_mask)
        epoch_loss = (running_loss*self.accumulation_steps)/total_batches
        #dice = epoch_log(phase, epoch, epoch_loss, measure, start)
        dice = epoch_log(epoch_loss, measure)
        self.losses[phase].append(epoch_loss)
        self.dice_score[phase].append(dice)
        torch.cuda.empty_cache()
        return epoch_loss
    def start(self):
        for epoch in range (self.num_epochs):
            self.iterate(epoch,"train")
            state = {
                "epoch": epoch,
                "best_loss": self.best_loss,
                "state_dict": self.net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
            }
            with torch.no_grad():
                val_loss = self.iterate(epoch,"val")
                self.scheduler.step(val_loss)
            if val_loss < self.best_loss:
                print("******** New optimal weights found, saving state ********")
                state["best_loss"] = self.best_loss = val_loss
                torch.save(state, "./model_office_100_defaults_epoch_10xlr2_myunet_binary.pth")
            print ()

def main():
    #print("in main of train")
    
    # model2 = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    # model_trainer2 = Trainer(model2)
    # model_trainer2.start()
    
    model = UNet()
    
    model_trainer = Trainer(model)
    model_trainer.start()
    
    #image = torch.rand((1,1,572,572))#creating a test image
    #print(model(image))

if __name__=="__main__": 
    main()

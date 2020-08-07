#python file contains code to train the U-net
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

#loss.requres_grad = True ##RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn

#kaggle csv
# df=pd.read_csv('kaggle_data/train_masks_kaggle.csv')

# kaggle locations of images
# train_img_dir='kaggle_data/train-128'
# train_img_masks_dir='kaggle_data/train_masks-128'



# image directories
train_img_dir = 'images/processed_images/train'
train_img_masks_dir = 'images/processed_images/train_masks'

#read csv file of image names
df=pd.read_csv('image_names_specific_seg.csv')

#mean, std = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225) # for rbg images, related to imagenet
mean = 0
std = 255

torch.set_printoptions(threshold=50000) # for debugging, allows to print more of tensors


from PIL import Image
import torchvision.transforms.functional as TF
isTraining=0

# when dataloader requests samples using index it fetches input image and target mask,
# applys transformation and returns it
class CityDataset(Dataset):
    def __init__(self, df, train_img_dir, train_img_masks_dir, mean, std, phase):
        #self.fname = df['images'].values.tolist()
        self.fname = df['img'].values.tolist() #kaggle one - their csv has this column name
        self.train_img_dir = train_img_dir
        self.train_img_masks_dir = train_img_masks_dir
        self.mean = mean
        self.std = std
        self.phase = phase
        self.isTraining=isTraining
        #self.transform = get_transform(phase,mean,std)

    def transform(self,image,mask,isTraining):
        #flip image if training and at 50% chance
        if random.random ()>0.5 and isTraining==1.:
            image = TF.hflip(image)       
            mask = TF.hflip(mask)

        #convert to tensor
        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)
        mask=mask*256
        mask=(mask*10**0).round()/(10**0)

        image = TF.normalize(image, mean=mean, std=std)

        return image, mask


    def __getitem__(self, idx):
        name = self.fname[idx]
        img_name_path = os.path.join(self.train_img_dir,name)
        #mask_name_path=img_name_path.split('.')[0].replace('train-128','train_masks-128')+'_mask.png' #kaggle dirs
        mask_name_path = os.path.join(self.train_img_masks_dir,name)

        img = cv2.imread(img_name_path, cv2.IMREAD_GRAYSCALE) #added to make this grayscale similar to below line
        #img = cv2.imread(img_name_path) #non grayscale (rgb) version 
        mask = cv2.imread(mask_name_path, cv2.IMREAD_GRAYSCALE)
       
        #convert to PIL image
        im_pil = Image.fromarray(img)
        mask_pil = Image.fromarray(mask)

        if(self.phase=='train'):
            isTraining = 1.

        else:
            isTraining = 0.
        img_aug, mask_aug = self.transform(im_pil,mask_pil,isTraining)
        
        torch.set_printoptions(profile="full")
        torch.set_printoptions(profile="default")
        
        img_aug,mask_aug = self.transform(im_pil,mask_pil,isTraining)

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
    pred = torch.argmax(pred,1) #extract mask values
    targs = targs.squeeze(1)#get rid of channels value to match pred shape
    correctlyClassified = torch.eq(pred,targs)
    correctlyClassified= correctlyClassified.numpy()
    correctlyClassified = np.count_nonzero(correctlyClassified) #count the pixels which were classified correctly
    numerator = 2. * correctlyClassified #total correctly classified pixels * 2
    denom = (pred.shape[1]*pred.shape[2] + targs.shape[1]*targs.shape[2]) #total number of pixels in both masks
    dice = numerator/denom
    
    return dice

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
        self.lr=5e-3
        self.num_epochs = 100
        self.phases = ['train','val']
        self.best_loss = float('inf')
        self.device = torch.device("cuda:0")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        self.net = model.to(self.device)
        cudnn.benchmark = True
        #self.criterion = torch.nn.BCEWithLogitsLoss()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.net.parameters(),lr=self.lr)
        self.scheduler = ReduceLROnPlateau(self.optimizer,mode='min',patience=3,verbose=True)
        self.dataloaders = {phase: CityDataloader(df, train_img_dir,train_img_masks_dir, mean, std,
                                                phase=phase,batch_size=self.batch_size[phase],
                                                num_workers=self.num_workers) for phase in self.phases}

        self.losses = {phase:[] for phase in self.phases}
        self.dice_score = {phase:[] for phase in self.phases}

    def forward(self,inp_images,tar_mask):
        inp_images = inp_images.to(self.device)
        tar_mask = tar_mask.to(self.device)
        #print(inp_images.shape)
        #print(tar_mask.shape)

        #inp_images = inp_images.unsqueeze(0) # adding dimension for batch (s/b 1,1,572,572)
        pred_mask = self.net(inp_images)
        # target mask is the one from my preprocessing, pred is what came out of my Unet
          
        tar_mask=tar_mask.long()#change it to Long type
        tar_mask = tar_mask.squeeze(1) #get ris of one dim of target mask to calculate loss
        
        #pred_mask=pred_mask.squeeze(0)
        #tar_mask=tar_mask.squeeze(0)
   
        loss = self.criterion(pred_mask,tar_mask) #changed from pred_mask to my_tensor
        return loss, pred_mask #changed from pred_mask to my_tensor
       
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
            #pred_mask = pred_mask.detach().cuda() #cuda expected cpu
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
                torch.save(state, "./model_office_12seg_2.pth")
            print ()

def main():
    #print("in main of train")
    
    # model2 = smp.Unet("resnet18", encoder_weights="imagenet", classes=1, activation=None)
    # model_trainer2 = Trainer(model2)
    # model_trainer2.start()
    
    model = UNet()
    model = model.cuda()#cuda expected but got cpu
    model_trainer = Trainer(model)
    model_trainer.start()
    
    #image = torch.rand((1,1,572,572))#creating a test image
    #print(model(image))

if __name__=="__main__": 
    main()

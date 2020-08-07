#this is to do inference on models

import segmentation_models_pytorch as smp #-- this was to get u-net architecture
import torch
#import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from tqdm import tqdm_notebook as tqdm # just a progress bar- uses jupyter etc
import os

from train import CityDataloader # get the U-net model 

#mean, std = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225) # for rbg images, related to imagenet
mean,std = (0,255)
from UNet import UNet # get the U-net model 

df = pd.read_csv('image_names_specific_seg.csv') # need a csv to do inference

# kaggle locations of images
train_img_dir = 'test_segment_specific/train'
train_img_masks_dir = 'test_segment_specific/train_masks'

ckpt_path = 'model_office_12seg.pth'

device = torch.device("cuda")

inference_image = os.listdir( train_img_dir ) 
inference_image_fullpath = train_img_dir+"/"+inference_image[0]
img = mpimg.imread(inference_image_fullpath)

if __name__=="__main__":  
    test_dataloader = CityDataloader(df,train_img_dir,train_img_masks_dir,mean,std,'val',1,4)
    #model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
    model = UNet()
    model.to(device)
    #torch.no_grad() # this disallows gradient descent/weight change
    model.eval() # sets to evaluation mode (not training)
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    # start prediction
    predictions = []
    fig, (ax1,ax2,ax3) = plt.subplots(1,3,figsize=(15,15))
    fig.suptitle('predicted_mask//original_mask//original image')
    for i, batch in enumerate(test_dataloader):
        
        images,mask_target = batch

        #batch_preds = torch.sigmoid(model(images.to(device))) # make prediction by passing image to model
        #pred = net()
        batch_preds = model(images.to(device))

        #pred_mask = torch.argmax(pred_mask,1) #extract mask values
        #batch_preds = torch.argmax(batch_preds,1)

        batch_preds = batch_preds.detach().cpu()
        batch_preds = torch.argmax(batch_preds,1) #extract mask values


        #batch_preds = (batch_preds>0).float()
        #batch_preds = batch_preds.detach().cpu().numpy()

        ax1.imshow(np.squeeze(batch_preds),cmap='gray')
        ax2.imshow(np.squeeze(mask_target),cmap='gray')
        
        # get original image on plot
        #images = images.squeeze(0)
        #test = images.numpy()
        #test = np.rollaxis(test, 0, 3)  
        ax3.imshow(np.squeeze(images),cmap='gray')

        #ax3.imshow(images)

        plt.show()
        break

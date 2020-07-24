#this is to do inference on models

import segmentation_models_pytorch as smp #-- this was to get u-net architecture
import torch
#import torch.nn as nn
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm_notebook as tqdm # just a progress bar- uses jupyter etc
import concurrent.futures

from train import CityDataloader # get the U-net model 

mean, std = (0.485, 0.456, 0.406),(0.229, 0.224, 0.225) # for rbg images, related to imagenet

df=pd.read_csv('kaggle_data/train_masks_kaggle.csv') # kaggle csv

# kaggle locations of images
train_img_dir='kaggle_data/train-128'
train_img_masks_dir='kaggle_data/train_masks-128'

ckpt_path='model_office.pth'

device = torch.device("cuda")

if __name__=="__main__":  
    test_dataloader=CityDataloader(df,train_img_dir,train_img_masks_dir,mean,std,'val',1,4)
    model = smp.Unet("resnet18", encoder_weights=None, classes=1, activation=None)
    model.to(device)
    #torch.no_grad() # this disallows gradient descent/weight change
    model.eval() # sets to evaluation mode (not training)
    state = torch.load(ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state["state_dict"])

    # start prediction
    predictions = []
    fig, (ax1,ax2,ax3)=plt.subplots(1,3,figsize=(15,15))
    fig.suptitle('predicted_mask//original_mask')
    for i, batch in enumerate(test_dataloader):
        
        images,mask_target = batch

        batch_preds = torch.sigmoid(model(images.to(device)))
        batch_preds = batch_preds.detach().cpu().numpy()
        ax1.imshow(np.squeeze(batch_preds),cmap='gray')
        ax2.imshow(np.squeeze(mask_target),cmap='gray')

        images=images.squeeze(0)
        test=images.numpy()

        test = np.rollaxis(test, 0, 3)  

        ax3.imshow(test)

        plt.show()
        break
    
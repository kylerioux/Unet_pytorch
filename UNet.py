import torch
import torch.nn as nn
import numpy as np
import math

# this python file contains the network architecture of a U-net

def double_conv(in_c,out_c):
    #function representing the double convolutions  present in UNet architecture
    #on both the encoder and decoder sides

    def init_weights(m):
        if(type(m)==nn.Conv2d):
            print(m.in_channels)
            torch.nn.init.normal_(tensor=m.weight,std = math.sqrt(2/(m.in_channels*3*3)))

    conv = nn.Sequential(
        #create a container for our layers we will repeat
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),

        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True)
    )
    conv.apply(init_weights)
    return conv

def crop_image(largerTensor,smallerTensor):
    #here we are cropping square tensors. The size of tensors is (batch size, channels, height, width) so sizes [2]=[3]
    target_size = smallerTensor.size()[2]
    larger_size = largerTensor.size()[2]
    size_change = larger_size - target_size
    half_size_change = size_change//2 # this is integer division- rounds down always 
    return largerTensor[:,:,half_size_change:larger_size-half_size_change,half_size_change:larger_size-half_size_change]


class UNet(nn.Module):
    def __init__(self):
        super(UNet,self).__init__()

        #inititalize our variables with our custom function and pytorch functions
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.down_conv_1 = double_conv(1,64)
        self.down_conv_2 = double_conv(64,128)
        self.down_conv_3 = double_conv(128,256)
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

        # each up conv(trans conv part) halves feature channels and doubles the spatial size
        self.trans_conv_2x2_1 = nn.ConvTranspose2d(in_channels=1024,out_channels=512,kernel_size=2,stride=2) 
        #wasnt able to find stride for transposed conv in paper--assume 2 is implied by halving feature channels/doubling spatial size
        torch.nn.init.normal_(tensor=self.trans_conv_2x2_1.weight,std = math.sqrt(2/(1024*2*2)))
        
        self.up_conv_1 = double_conv(1024,512) #input to this is the concatinated tensors
        
        self.trans_conv_2x2_2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        torch.nn.init.normal_(self.trans_conv_2x2_2.weight,std = math.sqrt(2/(512*2*2)))

        self.up_conv_2 = double_conv(512,256) 

        self.trans_conv_2x2_3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        torch.nn.init.normal_(self.trans_conv_2x2_3.weight,std = math.sqrt(2/(256*2*2)))
        
        self.up_conv_3 = double_conv(256,128)  

        self.trans_conv_2x2_4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        torch.nn.init.normal_(self.trans_conv_2x2_4.weight,std = math.sqrt(2/(128*2*2)))
        self.up_conv_4 = double_conv(128,64) #input to this is the concatinated tensors

        self.output = nn.Conv2d(in_channels=64,out_channels=1, kernel_size=1)#out_channels is number of segmentation labels

    def forward(self,image):
        #this is the encoder (forward pass)
        #image expected size is tuple: (batch size,channels, height, width)
        #this is channels first

        outp1 = self.down_conv_1(image) #pass to concat with decoder
        outp2 = self.max_pool_2x2(outp1)

        outp3 = self.down_conv_2(outp2) #pass to concat with decoder
        outp4 = self.max_pool_2x2(outp3)

        outp5 = self.down_conv_3(outp4) #pass to concat with decoder
        outp6 = self.max_pool_2x2(outp5)

        outp7 = self.down_conv_4(outp6) #pass to concat with decoder
        outp8 = self.max_pool_2x2(outp7)

        outp9 = self.down_conv_5(outp8) #pass straight to decoder

        #this is the decoder part 
        outp10 = self.trans_conv_2x2_1(outp9) #up conv here with output from encoder
        #here we have to crop output7 before concatenating with output 10  due to unequal sizes
        #this could be done other ways with padding but in paper they cropped larger of the two to match smaller
        outp7_cropped = crop_image(outp7,outp10)
        outp11 = self.up_conv_1(torch.cat((outp7_cropped,outp10),dim=1)) #concatenate cropped outp7 with 10 and double conv it
        
        
        outp12 = self.trans_conv_2x2_2(outp11)
        outp5_cropped = crop_image(outp5,outp12)
        outp13 = self.up_conv_2(torch.cat((outp5_cropped,outp12),dim=1)) 

        outp14 = self.trans_conv_2x2_3(outp13)
        outp3_cropped = crop_image(outp3,outp14)
        outp15 = self.up_conv_3(torch.cat((outp3_cropped,outp14),dim=1)) 

        outp16 = self.trans_conv_2x2_4(outp15)
        outp1_cropped = crop_image(outp1,outp16)
        outp17 = self.up_conv_4(torch.cat((outp1_cropped,outp16),dim=1)) 
        
        pre_flattened_outp = self.output(outp17)

        return pre_flattened_outp

        #flatten output- classify based on channel (label) with highest value
        #flattened_outp = torch.argmax(pre_flattened_outp.squeeze(),dim=0).detach().cpu().numpy() --can do this step in inference
        #note this is no longer a tensor.

        #print(np.unique(flattened_outp)) # this is to get unique values in an array

        
if __name__=="__main__":  
    #image = torch.rand((1,1,572,572))#creating a test image
    #print(image)
    #model = UNet()

    #print(model(image))
    print("in main of Unet")
    

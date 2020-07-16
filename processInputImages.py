from PIL import Image
import os, sys

path_preprocessed = "images/preprocessed" #path to preprocessed images
path_processed = "images/processed" #path to images once processed - will store here

dir_folders = os.listdir( path_preprocessed ) #directory containing preprocessed image folders (4)

for folder in dir_folders: #iterate through all 4 preprocessed image folders
    dir_images = os.listdir( path_preprocessed+'/'+folder ) #directory containing all images within each folder

    for image in dir_images: #iterate through all images in each folder
    
        if os.path.isfile(path_preprocessed+"/"+folder+"/"+image): #ensure path is an existing file
            open_im = Image.open(path_preprocessed+"/"+folder+"/"+image) #open image so it can be resized
            resized_image = open_im.resize((572,572)) #resize the image
            resized_image.save(path_processed+"/"+folder+"/"+image) #save it to the processed folder with the same name 


from PIL import Image
import os, sys

# The execution of this python code reads in images stored in the images/preprocessed_images folder
# and resizes all images (segmentations and source) to 572x572, the size accepted by the network.
# RGB source images are also converted to grayscale
# the processed images are saved with the same name in the images/processed_images folder

path_preprocessed = "images/preprocessed_images" # path to preprocessed images
path_processed = "images/processed_images" # path to images once processed - will store here

dir_folders = os.listdir( path_preprocessed ) # directory containing preprocessed image folders (4)

for folder in dir_folders: # iterate through all 4 preprocessed image folders
    dir_images = os.listdir( path_preprocessed+'/'+folder ) # directory containing all images within each folder

    # check if image we are looking at now is part of testing or training subset
    if("train" in folder):
        currentData = "train"
    else:
        currentData = "test"

    for image in dir_images: # iterate through all images in each folder
    
        if os.path.isfile(path_preprocessed+"/"+folder+"/"+image): # ensure path is an existing file
            open_im = Image.open(path_preprocessed+"/"+folder+"/"+image) # open image so it can be resized
            resized_image = open_im.resize((572,572)) # resize the image
            
            # save images in their respective folders depending on whether they are 
            if("regular" in folder): # only perform this on base images- not segmentation masks
                resized_image = resized_image.convert('L') # changes images to grayscale (if your data set is RGB) - if already grayscale shouldn't change anything
                resized_image.save(path_processed +"/"+ currentData +"/"+ image) # save it to the processed folder with the same name

            else: # this is for the segmentation masks
                resized_image.save(path_processed+ "/"+ currentData + "_masks" +"/"+ image) # save it to the processed folder with the same name

import os
import pandas as pd

BASE_DIR = 'kaggle_data/test_segment_specific/'
train_folder = BASE_DIR+'train-128/'
train_annotation = BASE_DIR+'train_masks-128/'

files_in_train = sorted(os.listdir(train_folder))
files_in_annotated = sorted(os.listdir(train_annotation))

images = [i for i in files_in_train if i in files_in_annotated] #checks that the file exists in both folders (with same name)

df = pd.DataFrame()
df['img'] = [str(x) for x in images]
df['labels'] = [str(x) for x in images]

df.to_csv('image_names_specific_seg.csv', header="images") # prints to csv named here
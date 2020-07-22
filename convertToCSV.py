#something like this - https://datascience.stackexchange.com/questions/49094/how-to-transform-a-folder-of-images-into-csv-file
import os
import pandas as pd

BASE_DIR = 'images/processed_images/'
train_folder = BASE_DIR+'train/'
train_annotation = BASE_DIR+'train_masks/'

files_in_train = sorted(os.listdir(train_folder))
files_in_annotated = sorted(os.listdir(train_annotation))

images = [i for i in files_in_train if i in files_in_annotated]

df = pd.DataFrame()
df['images'] = [str(x) for x in images]
df['labels'] = [str(x) for x in images]

df.to_csv('files_path.csv', header="images")
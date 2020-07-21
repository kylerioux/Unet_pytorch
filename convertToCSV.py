#something like this - https://datascience.stackexchange.com/questions/49094/how-to-transform-a-folder-of-images-into-csv-file
import os
import pandas as pd

BASE_DIR = 'images/'
train_folder = BASE_DIR+'train/'
train_annotation = BASE_DIR+'annotated_train_data/'

files_in_train = sorted(os.listdir(train_folder))
files_in_annotated = sorted(os.listdir(train_annotation))

images=[i for i in files_in_train if i in files_in_annotated]

df = pd.DataFrame()
df['images']=[train_folder+str(x) for x in images]
df['labels']=[train_annotation+str(x) for x in images]

pd.to_csv('files_path.csv', header=None)
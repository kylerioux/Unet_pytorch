#kaggle images processing - gif to png
import pathlib
from pathlib import Path
import concurrent.futures

from PIL import Image

PATH = Path('kaggle_data')

# convert masks from gif to png
(PATH/'train_masks_png').mkdir(exist_ok=True)
def convert_img(fn):
    fn = fn.name
    Image.open(PATH/'train_masks'/fn).save(PATH/'train_masks_png'/f'{fn[:-4]}.png') #opening and saving image

files = list((PATH/'train_masks').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(convert_img, files)  #uses multi thread for fast conversion

# we convert the high resolution image mask to 572*572 for starting for the masks.
(PATH/'train_masks-572').mkdir(exist_ok=True)
def resize_mask(fn):
    Image.open(fn).resize((572,572)).save((fn.parent.parent)/'train_masks-572'/fn.name)

files = list((PATH/'train_masks_png').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_mask, files)

# # # we convert the high resolution input image to 572*572
(PATH/'train-572').mkdir(exist_ok=True)
def resize_img(fn):
    Image.open(fn).resize((572,572)).save((fn.parent.parent)/'train-572'/fn.name)

files = list((PATH/'train').iterdir())
with concurrent.futures.ThreadPoolExecutor(8) as e: e.map(resize_img, files)
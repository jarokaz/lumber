from PIL import Image
from PIL import ImageDraw

from os import listdir
from os.path import isfile, join
import time
import numpy as np
import pandas as pd 


def clone_sound_samples(image, group, size, prefix, outputpath, file_type):        
    group = group.sort_values(['min_x', 'min_y'])
    group = group.reset_index(drop=True)
    cols = group['min_x'].nunique()
    rows = group['min_y'].nunique()
    prefix = prefix + '_' + group.iloc[0]['image'] + '_sound_'
    outputpath = join(outputpath, 'sound')
    filenumber = 1
    for col in range(cols-1):
        for row in range(rows-1):
            segments = [group.iloc[i]['label'] for i in [col*rows + row, (col+1)*rows + row, col*rows + row + 1 , (col+1)*rows + row + 1]]
            if segments.count('sound') == 4:
                x = group.iloc[col*rows + row]['min_x'] + 10
                y = group.iloc[col*rows + row]['min_y'] + 5
                im = image.crop((x, y, x + size, y + size))
                filename = join(outputpath, prefix + str(filenumber) + file_type)
                im.save(filename)
                filenumber += 1
        


def generate_sound_samples(df, prefix, inputpath, outputpath):
    clone_size = 100
    file_type = ".PNG"
    gb = df.groupby(df['image'])
    for name, group in gb:
        filepath = join(inputpath, name)
        image = Image.open(filepath)
        clone_sound_samples(image, group, clone_size,  prefix, outputpath, file_type)
        print("Generated {0} samples from image: {1}".format(prefix, name))
   


### Generate train data
inputpath="c:/repos/lumber/data/unprocessed/orig"
outputpath="c:/repos/lumber/data/training"
prefix = 'train'
index = "c:/repos/lumber/data/train_index.csv"
df_train = pd.read_csv(index)

print("Starting generation of training samples")
generate_sound_samples(df_train, prefix, inputpath, outputpath)


### Generate test data
inputpath="c:/repos/lumber/data/unprocessed/orig"
outputpath="c:/repos/lumber/data/testing"
prefix = 'test'
index = "c:/repos/lumber/data/test_index.csv"
df_test = pd.read_csv(index)

print("Starting generation of testing samples")
generate_sound_samples(df_train, prefix, inputpath, outputpath)

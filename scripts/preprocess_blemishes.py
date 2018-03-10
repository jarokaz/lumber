from PIL import Image
from os.path import isfile, join
import numpy as np
import pandas as pd 


def clone_blemishes(image, group, size, stride, prefix, outputpath, file_type):
    min_x = min(group['min_x'])
    min_y = min(group['min_y'])
    max_x = max(group['max_x'])
    max_y = max(group['max_y'])
    
    group = group[group['label'] != 'sound']

    if len(group) == 0:
        return

    prefix = prefix + '_' + group.iloc[0]['image'] + '_'
    filenumber = 1
    
    for row in group.iterrows():
        x1, y1, x2, y2 = row[1][2], row[1][1], row[1][4], row[1][3]
        x = x_start = max(min_x, x2 - size)
        y = y_start = max(min_y, y2 - size)
        x_end = min(x1, max_x - size)
        y_end = min(y1, max_y - size)
        folder = row[1][5]

        while x <= x_end:
            while y <= y_end:
                im = image.crop((x, y, x + size, y + size))
                filename = join(outputpath, folder, prefix + '_' + folder + '_' + str(filenumber) + file_type)
                im.save(filename)
                filenumber += 1
                y = y + stride
            y = y_start
            x = x + stride        


def generate_blemished_samples(df, prefix, inputpath, outputpath):
    clone_size = 100
    stride = 10
    file_type = ".PNG"
    gb = df.groupby(df['image'])
    for name, group in gb:
        filepath = join(inputpath, name)
        image = Image.open(filepath)
        clone_blemishes(image, group, clone_size, stride, prefix, outputpath, file_type)
        print("Generated {0} samples from image: {1}".format(prefix, name))
   



### Generate train data
inputpath="c:/repos/lumber/data/unprocessed/orig"
outputpath="c:/repos/lumber/data/training"
prefix = 'train'
index = "c:/repos/lumber/data/train_index.csv"
df_train = pd.read_csv(index)

print("Starting generation of training samples")
generate_blemished_samples(df_train, prefix, inputpath, outputpath)

### Generate test data
inputpath="c:/repos/lumber/data/unprocessed/orig"
outputpath="c:/repos/lumber/data/testing"
prefix = 'test'
index = "c:/repos/lumber/data/test_index.csv"
df_test = pd.read_csv(index)

print("Starting generation of testing samples")
generate_blemished_samples(df_train, prefix, inputpath, outputpath)


from PIL import Image

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
                filename = prefix + str(filenumber) + file_type
                filepath = join(outputpath, filename)
                im.save(filepath)
                logfile.write("{0}  {1}\n".format(filename, 'sound'))
                filenumber += 1
        

def generate_sound_samples(df, prefix, inputpath, outputpath):
    clone_size = 100
    file_type = ".PNG"
    gb = df.groupby(df['image'])
    for name, group in gb:
        filepath = join(inputpath, name)
        image = Image.open(filepath)
        clone_sound_samples(image, group, clone_size,  prefix, outputpath, file_type)
      
   

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
                filename = prefix + '_' + folder + '_' + str(filenumber) + file_type
                filepath = join(outputpath, folder, filename)
                im.save(filepath)
                logfile.write("{0}  {1}\n".format(filename, folder))
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
     
   

### COnfigure environment
inputpath="c:/repos/lumber/data/unprocessed/orig"
training_outputpath="c:/repos/lumber/data/training"
testing_outputpath="c:/repos/lumber/data/testing"
training_index = "c:/repos/lumber/data/train_index.csv"
testing_index = "c:/repos/lumber/data/test_index.csv"
training_prefix = "train"
testing_prefix = "test"


# Global logfile
path = "c:/repos/lumber/data/labels.txt"
logfile = open(path, "w")

# Read training index
df_train = pd.read_csv(training_index)

# Read testing index
df_test = pd.read_csv(testing_index)

### Generate training samples
print("Starting generation of blemished training samples")
generate_blemished_samples(df_train, training_prefix, inputpath, training_outputpath)

print("Starting generation of sound training samples")
generate_sound_samples(df_train, training_prefix, inputpath, training_outputpath)

### Generate testing samples
print("Starting generation of blemished testing samples")
generate_blemished_samples(df_test, testing_prefix, inputpath, testing_outputpath)

print("Starting generation of sound testing samples")
generate_sound_samples(df_test, testing_prefix, inputpath, testing_outputpath)

logfile.close()

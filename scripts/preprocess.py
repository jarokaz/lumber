from PIL import Image
from PIL import ImageDraw

from os import listdir
from os.path import isfile, join
import time
import numpy as np
import pandas as pd 


def is_overlap(rec1, rec2, margin):
    """
    Returns true if rectangles overlap; false otherwise
    
    Parameters
    ----------
    rec1: [min_x, min_y, max_x, max_y]
    rec2: [min_x, min_y, max_x, max_y]
    margin: overlap margin
    """

    overlap = False

    w = rec1[2] - rec1[0]
    h = rec1[3] - rec1[1]

    x1 = rec2[0] - rec1[0]
    y1 = rec2[1] - rec1[1]
    x2 = rec2[2] - rec1[0]
    y2 = rec2[3] - rec1[1]

    if (x1 > -margin) and (x1 < w) and (y1 > -margin) and (y1 < h) and (x2 < margin + w) and (y2 < margin + h):
       overlap = True

    return overlap
        

def clone_blemishes(image, group, size, stride, outputpath, suffix, file_number, file_type):
    min_x = min(group['min_x'])
    min_y = min(group['min_y'])
    max_x = max(group['max_x'])
    max_y = max(group['max_y'])
    
    group = group[group['label'] != 'sound']
    
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
                filename = join(outputpath, folder, suffix + '_' + str(file_number) + file_type)
                im.save(filename)
                file_number += 1
                y = y + stride
            y = y_start
            x = x + stride        
    return file_number


index = "c:/repos/lumber/data/unprocessed/manlabel.txt"
df = pd.read_csv(index, delim_whitespace=True, header=None, names=['image','min_y','min_x','max_y','max_x','label'])
gb = df.groupby(df['image']) 

file_type = ".PNG"
inputpath="c:/repos/lumber/data/unprocessed"
outputpath="c:/repos/lumber/data/processed"

clone_size = 100
stride = 10
file_number = 1

for name, group in gb:
    filepath = join(inputpath, name)
    image = Image.open(filepath)
    file_number = clone_blemishes(image, group, clone_size, stride, outputpath, name, file_number, file_type)
    print(name)
               

        

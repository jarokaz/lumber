from PIL import Image
from PIL import ImageDraw

from os import listdir
from os.path import isfile, join
import time
import numpy as np
import pandas as pd 


def overlay_grid(image, group):
    draw = ImageDraw.Draw(image)
    for row in group.iterrows():
        cor = [row[1][2], row[1][1], row[1][4], row[1][3]]
        draw.rectangle(cor)
    return image

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
        


index = "c:/repos/lumber/data/unprocessed/manlabel.txt"
df = pd.read_csv(index, delim_whitespace=True, header=None, names=['image','min_y','min_x','max_y','max_x','label'])
gb = df.groupby(df['image']) 

file_type = ".PNG"
prefix = 'train'
inputpath="c:/repos/lumber/data/unprocessed"
outputpath="c:/repos/lumber/data/processed"

clone_size = 100

for name, group in gb:
    filepath = join(inputpath, name)
    image = Image.open(filepath)
    clone_sound_samples(image, group, clone_size, prefix, outputpath, file_type)
    


               

        

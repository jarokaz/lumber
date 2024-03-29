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



index = "../../data/unprocessed/manlabel.txt"
df = pd.read_csv(index, delim_whitespace=True, header=None, names=['image','min_y','min_x','max_y','max_x','label'])
gb = df.groupby(df['image']) 

file_type = ".PNG"
inputpath="../../data/unprocessed"
outputpath="../../data/originals"

for name, group in gb:
    filepath = join(inputpath, name)
    image = Image.open(filepath)
    # overlay_grid(image, group)
    filename = join(outputpath, group.iloc[0]['image'] + file_type)
    image.save(filename)
    print("Processed image: {0}".format(filename))



    
    


        

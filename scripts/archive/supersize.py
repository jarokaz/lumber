from PIL import Image
import PIL

from os import listdir
from os.path import isfile, join
from os import listdir
import time
import numpy as np
import pandas as pd 


inputpath = "c:/repos/lumber/data/testing/edge_knot"
outputpath = "c:/repos/lumber/data/upscale"


files = [f for f in listdir(inputpath) if isfile(join(inputpath, f))]

for file in files:
    image = Image.open(join(inputpath, file))
    largeimage = image.resize((224, 224), resample=PIL.Image.BILINEAR)
    largeimage.save(join(outputpath, file))



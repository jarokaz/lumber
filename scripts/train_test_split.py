import numpy as np
import pandas as pd 

index = "c:/repos/lumber/data/unprocessed/manlabel.txt"
df = pd.read_csv(index, delim_whitespace=True, header=None, names=['image','min_y','min_x','max_y','max_x','label'])

images = df['image'].unique()

total = len(images)
split = 0.2
test = round(total * split)
train = total - test

images_test = images[0:test]
images_train = images[test:total] 

df_train = df[df.image.isin(images_train)]
df_test = df[~df.image.isin(images_train)]


path = 'c:/repos/lumber/data/train_index.csv'
df_train.to_csv(path, index=False, header=True)

path = 'c:/repos/lumber/data/test_index.csv'
df_test.to_csv(path, index=False, header=True)
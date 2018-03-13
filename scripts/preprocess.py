from PIL import Image
from os import listdir
from os.path import isfile, join
import time
import numpy as np
import pandas as pd 
import argparse

def clone_sound_samples(image, group, size, outputpath, file_type, logfile):        
    group = group.sort_values(['min_x', 'min_y'])
    group = group.reset_index(drop=True)
    cols = group['min_x'].nunique()
    rows = group['min_y'].nunique()
    label = 'sound'
    prefix = group.iloc[0]['image'] + '_' + label + '_'
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
                logfile.write("{0}  {1}\n".format(filename, label))
                filenumber += 1
        

def generate_sound_samples(df, inputpath, outputpath, logfile):
    clone_size = 112
    file_type = ".PNG"
    gb = df.groupby(df['image'])
    for name, group in gb:
        filepath = join(inputpath, name)
        image = Image.open(filepath)
        clone_sound_samples(image, group, clone_size, outputpath, file_type, logfile)
      
   

def clone_blemishes(image, group, size, stride,outputpath, file_type, logfile):
    min_x = min(group['min_x'])
    min_y = min(group['min_y'])
    max_x = max(group['max_x'])
    max_y = max(group['max_y'])
    
    group = group[group['label'] != 'sound']

    if len(group) == 0:
        return

    prefix = group.iloc[0]['image'] 
    filenumber = 1
    
    for row in group.iterrows():
        x1, y1, x2, y2 = row[1][2], row[1][1], row[1][4], row[1][3]
        x = x_start = max(min_x, x2 - size)
        y = y_start = max(min_y, y2 - size)
        x_end = min(x1, max_x - size)
        y_end = min(y1, max_y - size)
        label = row[1][5]

        while x <= x_end:
            while y <= y_end:
                im = image.crop((x, y, x + size, y + size))
                filename = prefix + '_' + label + '_' + str(filenumber) + file_type
                filepath = join(outputpath, filename)
                im.save(filepath)
                logfile.write("{0}  {1}\n".format(filename, label))
                filenumber += 1
                y = y + stride
            y = y_start
            x = x + stride        


def generate_blemished_samples(df, inputpath, outputpath, logfile):
    clone_size = 112
    stride = 10
    file_type = ".PNG"
    gb = df.groupby(df['image'])
    for name, group in gb:
        filepath = join(inputpath, name)
        image = Image.open(filepath)
        clone_blemishes(image, group, clone_size, stride, outputpath, file_type, logfile)
     


def main(input_index, output_labels, input_dir, output_dir):
    with open(output_labels, "w") as logfile:
        index = pd.read_csv(input_index)
        
        print(len(index))
        return
    
        print("Starting generation of blemished training samples")
        generate_blemished_samples(index, input_dir, output_dir, logfile)
        print("Starting generation of sound training samples")
        generate_sound_samples(index, input_dir, output_dir, logfile)


OUTPUT_LABELS = 'labels.txt'
        
if __name__ == '__main__':
  parser = argparse.ArgumentParser("Preprocessing lumber images")

  parser.add_argument(
      '--input-index',
      type=str,
      required = True,
      help='The file index for origininal images')
  parser.add_argument(
      '--input-dir',
      type=str,
      required = True,
      help='Directory with original unzipped images')
  parser.add_argument(
      '--output-index',
      type=str,
      required = True,
      help='The file index for the processed images')
  parser.add_argument(
      '--output-dir',
      type=str,
      required = True,
      help='Directory for processed .PNG files')

  args = parser.parse_args()
  print(args)
  main(args.input_index, args.output_index, args.input_dir, args.output_dir)



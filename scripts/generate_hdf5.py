import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image
import argparse
import argparse
from PIL import Image
import pandas as pd



             
def convert_to_hdf5(labels, inputpath, outputfile):
     with h5py.File(outputfile,  mode='w') as hdf5:
    
        images = hdf5.create_dataset("images", images_shape, np.uint8)
        labels = hdf5.create_dataset("labels", labels_shape, np.uint8)

        counter = 0
        for index, row in labels.iterrows():
            im = load_image(inputpath, row['image'])
            label = row['label']
            labels[i] = label 
            images[i] = np.array(im)

            counter += 1
            if counter%500 == 0:
                print("processed {0} images.".format(counter))
 
   
   

def main(inputpath, outputfile, allowed_labels):
    labels = pd.read_csv(join(inputpath, LABELS_FILE), delim_whitespace=True, header=None, names=['image', 'label'])
   
    labels = labels[labels['label'].isin(allowed_labels)] 
    labels['label']= labels['label'].apply(lambda x: allowed_labels.index(x))
    convert_to_hdf5(labels, inputpath, outputfile)
    
   

LABELS_FILE = 'labels.txt'
ALLOWED_LABELS = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']



if __name__ == '__main__':
  parser = argparse.ArgumentParser("HDF5 file generator")
  parser.add_argument(
      '--data-dir',
      type=str,
      required=True,
      help='Folder with images and labels file: labels.txt')
  parser.add_argument(
      '--output-file',
      type=str,
      required=True,
      help='Output file name')

  args = parser.parse_args()
  main(args.data_dir, args.output_file, ALLOWED_LABELS)

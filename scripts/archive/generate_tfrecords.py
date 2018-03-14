import numpy as np
from os import listdir
from os.path import isfile, join
import argparse
from PIL import Image
import pandas as pd
import tensorflow as tf



def load_image(path, name):
    im = Image.open(join(path, name))
    return np.array(im)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


    
def convert_to_tfrecord(labels, inputpath, outputfile):    
    with tf.python_io.TFRecordWriter(outputfile) as writer:
        counter = 0
        for index, row in labels.iterrows():
            im = load_image(inputpath, row['image'])
            example= tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'image': _bytes_feature(im.tostring()),
                        'label': _int64_feature(row['label'])
                    }))
            writer.write(example.SerializeToString())
            counter += 1
            if counter%500 == 0:
                print("Processed {0} images.".format(counter))
                      
 
   

def main(inputpath, outputfile, allowed_labels):
    labels = pd.read_csv(join(inputpath, LABELS_FILE), delim_whitespace=True, header=None, names=['image', 'label'])
   
    labels = labels[labels['label'].isin(allowed_labels)] 
    labels['label']= labels['label'].apply(lambda x: allowed_labels.index(x))
    convert_to_tfrecord(labels, inputpath, outputfile)
    
   

LABELS_FILE = 'labels.txt'
ALLOWED_LABELS = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']



if __name__ == '__main__':
  parser = argparse.ArgumentParser("TFRecords file generator")
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

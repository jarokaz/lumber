import numpy as np
import os
from os.path import isfile, join
import argparse
from PIL import Image
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

    
def load_image(path, name):
    im = Image.open(join(path, name))
    return np.array(im)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    
def convert_to_tfrecord(images, inputpath, outputfile):    
    
    with tf.python_io.TFRecordWriter(outputfile) as writer:
        for name, label in images:
            im = load_image(inputpath, name)
            example= tf.train.Example(
                features = tf.train.Features(
                    feature = {
                        'image': _bytes_feature(im.tostring()),
                        'label': _int64_feature(label)
                    }))
            writer.write(example.SerializeToString())
           
def extract_label(filename):
    label = filename[filename.find('_')+1 : filename.rfind('_')]
    return label
    
def prepare_image_label_list(image_folder):
    images = os.listdir(image_folder)
    images =[(image, ALLOWED_LABELS.index(extract_label(image))) for image in images if extract_label(image) in ALLOWED_LABELS]
    return images
    
def generate_tfrecords(snaphots_dir, output_dir, validation_percentage):
    
    development_folder = join(snaphots_dir, DEVELOPMENT_SUBFOLDER_NAME)
    development_images = prepare_image_label_list(development_folder)
    testing_folder = join(snaphots_dir, TESTING_SUBFOLDER_NAME)
    testing_images = prepare_image_label_list(testing_folder)
    
    print("Total number of development images is {0}".format(len(development_images)))
    print("Total number of testing images is {0}".format(len(testing_images)))
        
    # Split development set into training and validattion
    last_training_index = len(development_images)
    last_validation_index = round(last_training_index * validation_percentage/100)
    
    validation_images = development_images[0: last_validation_index]
    training_images = development_images[last_validation_index: last_training_index]
    
    print("Generating training TFRecords file using {0} images".format(len(training_images)))
    convert_to_tfrecord(training_images, development_folder, join(output_dir, TRAINING_FILE_NAME))
    
    print("Generating validation TFRecords file using {0} images".format(len(validation_images)))
    convert_to_tfrecord(validation_images, development_folder, join(output_dir, VALIDATION_FILE_NAME))
     
    print("Generating testing TFRecords file using {0} images".format(len(testing_images)))
    convert_to_tfrecord(testing_images, testing_folder, join(output_dir, TESTING_FILE_NAME))
   

    
    
TESTING_SUBFOLDER_NAME = 'testing'
DEVELOPMENT_SUBFOLDER_NAME = 'development'
SNAPSHOT_SIZE = 112
SNAPSHOT_FILETYPE = ".PNG"
SNAPSHOT_STRIDE = 10
ALLOWED_LABELS = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']
TRAINING_FILE_NAME = 'training.tfrecords'
VALIDATION_FILE_NAME = 'validation.tfrecords'
TESTING_FILE_NAME = 'testing.tfrecords'


    
def main(snapshots_dir, output_dir, validation_percentage):
   
    if not os.path.isdir(output_dir):
        print("Output directory does not exist !!!")
        return
    
    if not os.path.isdir(join(snapshots_dir, TESTING_SUBFOLDER_NAME)):
        print("Snapshots directory does not exist or folder structure incorrect !!!")
        return
    
    if not os.path.isdir(join(snapshots_dir, DEVELOPMENT_SUBFOLDER_NAME)):
        print("Snapshots directory does not exist or folder structure incorrect !!!")
        return
    
    if len(os.listdir(join(snapshots_dir, DEVELOPMENT_SUBFOLDER_NAME))) != 0:
        print("Snapshot folders not empty")
        return
        
    if len(os.listdir(join(snapshots_dir, TESTING_SUBFOLDER_NAME))) != 0:
        print("Snapshot folders not empty")
        return
    
    if len(os.listdir(output_dir)) != 0:
        print("TFRecords folders not empty")
        return
    
    if validation_percentage  > 50:
        print("{0} for validation? Think again...".format(validation_percentage))
        return
   
    print("TFRecords directory: {0}".format(output_dir))
    print("Percentage of images used for the training set: {0}".format(100 - validation_percentage))
    print("Percentage of images used for the validation set: {0}".format(validation_percentage))
    
    generate_tfrecords(snapshots_dir, output_dir, validation_percentage)


if __name__ == '__main__':
  parser = argparse.ArgumentParser("TFRecords file generator")

  parser.add_argument(
      '--output-dir',
      type=str,
      required=True,
      help='Directory for output training, validation, and testing files in TFRecords format')
    
  parser.add_argument(
      '--snapshots-dir',
      type=str,
      required=True,
      help='Directory for intermediate snapshots' )

  parser.add_argument(
      '--validation-percentage',
      type=int,
      default=10,
      help='Percentage of images to use as a validation set')
  

  args = parser.parse_args()
  main(args.snapshots_dir, args.output_dir, args.validation_percentage)

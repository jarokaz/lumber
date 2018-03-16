import numpy as np
import os
from os.path import isfile, join
import argparse
from PIL import Image
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split





def clone_sound_samples(image, group, size, outputpath, file_type):        
    group = group.sort_values(['min_x', 'min_y'])
    group = group.reset_index(drop=True)
    cols = group['min_x'].nunique()
    rows = group['min_y'].nunique()
    label = ALLOWED_LABELS[0]
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
                #logfile.write("{0}  {1}\n".format(filename, label))
                filenumber += 1
        

def generate_sound_samples(df, inputpath, outputpath):
    clone_size = SNAPSHOT_SIZE
    file_type = SNAPSHOT_FILETYPE
    gb = df.groupby(df['image'])
    for name, group in gb:
        filepath = join(inputpath, name)
        image = Image.open(filepath)
        clone_sound_samples(image, group, clone_size, outputpath, file_type)
      
   

def clone_blemishes(image, group, size, stride,outputpath, file_type):
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
                #logfile.write("{0}  {1}\n".format(filename, label))
                filenumber += 1
                y = y + stride
            y = y_start
            x = x + stride        


def generate_blemished_samples(df, inputpath, outputpath):
    clone_size = SNAPSHOT_SIZE
    stride = SNAPSHOT_STRIDE
    file_type = SNAPSHOT_FILETYPE
    gb = df.groupby(df['image'])
    for name, group in gb:
        filepath = join(inputpath, name)
        image = Image.open(filepath)
        clone_blemishes(image, group, clone_size, stride, outputpath, file_type)
     


        

def generate_snapshots(index_path, image_dir, snapshots_dir, testing_percentage):
    """
    Processes original images (in PPM format) into development and testing snapshots
    in PNG format.
    
    Args:
      index_path: path to the original label file
      testing_percentage: Integer percentage of images reserved for testing
      
    Returns:
      Creates two folders in snapshots_dir: development and testing. Puts processed snapshots
      (in .PNG format) into the folders using the following naming convention:
      <original_image_name>_<label>_<ID>.PNG
    """
    
    index = pd.read_csv(index_path,
                        delim_whitespace=True, 
                        header=None, 
                        names=['image','min_y','min_x','max_y','max_x','label'])
    assert len(index) !=0
    
    names = index['image'].unique()
    development_images, testing_images = train_test_split(names, 
                                                       test_size = testing_percentage/100)
    development = index[index.image.isin(development_images)]
    testing = index[index.image.isin(testing_images)]
    
    # Generate development snapshots
    print("Generating development snapshots with {0} original samples".format(len(development)))
    generate_sound_samples(development, image_dir, join(snapshots_dir, DEVELOPMENT_SUBFOLDER_NAME))
    generate_blemished_samples(development, image_dir, join(snapshots_dir, DEVELOPMENT_SUBFOLDER_NAME))
    print("Generated {0} development snapshots".format(len(os.listdir(join(snapshots_dir, DEVELOPMENT_SUBFOLDER_NAME)))))
    
    # Generate testing snapshots
    print("Generating testing snapshots with {0} original samples".format(len(testing)))
    generate_sound_samples(testing, image_dir, join(snapshots_dir, TESTING_SUBFOLDER_NAME))
    generate_blemished_samples(testing, image_dir, join(snapshots_dir, TESTING_SUBFOLDER_NAME))
    print("Generated {0} testing snapshots".format(len(os.listdir(join(snapshots_dir, TESTING_SUBFOLDER_NAME)))))
    

    
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
   

    
    
ORIGINAL_LABELS_FILENAME = 'manlabel.txt' 
SORTED_LABELS_FILENAME = 'labels.txt'
TESTING_SUBFOLDER_NAME = 'testing'
DEVELOPMENT_SUBFOLDER_NAME = 'development'
SNAPSHOT_SIZE = 112
SNAPSHOT_FILETYPE = ".PNG"
SNAPSHOT_STRIDE = 10
ALLOWED_LABELS = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']
TRAINING_FILE_NAME = 'training.tfrecords'
VALIDATION_FILE_NAME = 'validation.tfrecords'
TESTING_FILE_NAME = 'testing.tfrecords'


    
def main(image_dir, output_dir, snapshots_dir, validation_percentage, testing_percentage, action):
    if not os.path.isdir(image_dir):
        print("Image directory does not exist !!!")
        return
    
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
    
    if validation_percentage + testing_percentage > 50:
        print("Wrong split proportions. Think again...")
        return
    
    if not os.path.exists(join(image_dir, ORIGINAL_LABELS_FILENAME)):
        print("Missing manlabel.txt file from image directory")
        return
    
          
    print("Image directory: {0}".format(image_dir))
    
    if action == 'tfrecords':
      print("Only TFRecords will be generated")
      print("TFRecords directory: {0}".format(output_dir))
      print("Percentage of images used for the training set: {0}".format(100 - validation_percentage))
      print("Percentage of images used for the validation set: {0}".format(validation_percentage))
      generate_tfrecords(snapshots_dir, output_dir, validation_percentage)
    elif action == 'snapshots':
      print("Only snapshots will be generated")
      print("Percentage of unprocessd images used for the development set: {0}".format(100 - testing_percentage))
      print("Percentage of unprocessed images used for the testing set: {0}".format(testing_percentage))
      generate_snapshots(join(image_dir, ORIGINAL_LABELS_FILENAME), image_dir, snapshots_dir, testing_percentage)
    elif action == 'both':   
      print("Both snapshots and tfrecords will be generated")
      print("TFRecords directory: {0}".format(output_dir))
      print("Snapshots directory: {0}".format(snapshots_dir))
      print("Percentage of unprocessed images used for the development set: {0}".format(100 - testing_percentage))
      print("Percentage of unprocessed images used for the testing set: {0}".format(testing_percentage))
      print("Percentage of images used for the training set: {0}".format(100 - validation_percentage))
      print("Percentage of images used for the validation set: {0}".format(validation_percentage))
      generate_snapshots(join(image_dir, ORIGINAL_LABELS_FILENAME), image_dir, snapshots_dir, testing_percentage)
      generate_tfrecords(snapshots_dir, output_dir, validation_percentage)
    else:
      print("Wrong action specified")

    
    
    

LABELS_FILE = 'labels.txt'
ALLOWED_LABELS = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']



if __name__ == '__main__':
  parser = argparse.ArgumentParser("TFRecords file generator")
  parser.add_argument(
      '--image-dir',
      type=str,
      required=True,
      help='Folder with images and label file: labels.txt')
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
  parser.add_argument(
      '--testing-percentage',
      type=int,
      default=10,
      help='Percentage of images to use as a test set')
  parser.add_argument(
      '--action',
      type=str,
      default='tfrecords',
      help='Generate tf files: tfrecords; generate snapshots: snapshots; generate both: both')
  
  

  args = parser.parse_args()
  main(args.image_dir, args.output_dir, args.snapshots_dir, args.validation_percentage, args.testing_percentage, args.action)

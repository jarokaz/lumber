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
                      
 


def clone_sound_samples(image, group, size, outputpath, file_type):        
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
                #logfile.write("{0}  {1}\n".format(filename, label))
                filenumber += 1
        

def generate_sound_samples(df, inputpath, outputpath):
    clone_size = 112
    file_type = ".PNG"
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
    clone_size = 112
    stride = 10
    file_type = ".PNG"
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
    
   
    
    
    

    
    
ORIGINAL_LABELS_FILENAME = 'manlabel.txt' 
SORTED_LABELS_FILENAME = 'labels.txt'
TESTING_SUBFOLDER_NAME = 'testing'
DEVELOPMENT_SUBFOLDER_NAME = 'development'
SNAPSHOT_SIZE = 112
SNAPSHOT_FILETYPE = ".PNG"
SNAPSHOT_STRIDE = 10

    
def main(image_dir, output_dir, snapshots_dir, validation_percentage, testing_percentage):
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
    
    if validation_percentage + testing_percentage > 50:
        print("Wrong split proportions. Think again...")
        return
    
    if not os.path.exists(join(image_dir, ORIGINAL_LABELS_FILENAME)):
        print("Missing manlabel.txt file from image directory")
        return
    
    print("Image directory: {0}".format(image_dir))
    print("TFRecords directory: {0}".format(output_dir))      
    print("Percentage of images used for the validation set: {0}".format(validation_percentage))
    print("Percentage of images used for the testing set: {0}".format(testing_percentage))
    
    generate_snapshots(join(image_dir, ORIGINAL_LABELS_FILENAME), image_dir, snapshots_dir, testing_percentage)
       
    
    
    

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
      '--validation_percentage',
      type=int,
      default=10,
      help='Percentage of images to use as a validation set')
  parser.add_argument(
      '--testing_percentage',
      type=int,
      default=10,
      help='Percentage of images to use as a test set')
  
  

  args = parser.parse_args()
  main(args.image_dir, args.output_dir, args.snapshots_dir, args.validation_percentage, args.testing_percentage)

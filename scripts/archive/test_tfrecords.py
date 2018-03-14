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
                      
 
def load_tfrecords(file):
    record_iterator = tf.python_io.tf_record_iterator(path=file)
    
    images = []
    labels = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        image_feature = example.features.feature['image'].bytes_list.value[0]
        label_feature = example.features.feature['label'].int64_list.value[0]
        
        image = np.fromstring(image_feature, dtype=np.uint8)
           
        assert IMAGE_SHAPE[0]*IMAGE_SHAPE[1]*IMAGE_SHAPE[2] == image.shape[0]
        
        image = np.reshape(image, IMAGE_SHAPE)
        label = tf.keras.utils.to_categorical(label_feature, NUM_CLASSES)
        
        images.append(image)
        labels.append(label)
       
    images = np.asarray(images)
    labels = np.asarray(labels)
    
    return images, labels
    

def load_data_from_tfrecords(training_file, validation_file):
    
    x_train, y_train = load_tfrecords(training_file)
    x_test, y_test = load_tfrecords(validation_file)
    
    return x_train, y_train, x_test, y_test


 
LABELS_FILE = 'labels.txt'
INPUT_PATH = '../data/training'
ALLOWED_LABELS = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']
OUTPUT_PATH = '../data/tfrecords/training.tfrecords'
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7  

def main():
    allowed_labels = ALLOWED_LABELS

    labels = pd.read_csv(join(INPUT_PATH, LABELS_FILE), delim_whitespace=True, header=None, names=['image', 'label'])
   
    labels = labels[labels['label'].isin(allowed_labels)] 
    labels['label']= labels['label'].apply(lambda x: allowed_labels.index(x))


    dflabels = labels

    #convert_to_tfrecord(dflabels, INPUT_PATH, OUTPUT_PATH)
 
    images = []
    labels = []

    for index, row in dflabels.iterrows():
        im = load_image(INPUT_PATH, row['image'])
        label = row['label']
        images.append(im)
        labels.append(label)
   

    npimages = np.asarray(images)
    nplabels = np.asarray(labels)

    print("npimages and nplabels")
    print(npimages.shape)
    print(nplabels.shape)

    record_iterator = tf.python_io.tf_record_iterator(path=OUTPUT_PATH)
    images1 = []
    labels1 = []
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        
        image_feature = example.features.feature['image'].bytes_list.value[0]
        label_feature = example.features.feature['label'].int64_list.value[0]
        
        image = np.fromstring(image_feature, dtype=np.uint8)
        label = label_feature
           
        assert IMAGE_SHAPE[0]*IMAGE_SHAPE[1]*IMAGE_SHAPE[2] == image.shape[0]
        
        image = np.reshape(image, IMAGE_SHAPE)
        
        images1.append(image)
        labels1.append(label)


    npimages1 = np.asarray(images1)
    nplabels1 = np.asarray(labels1)

    print("npimages1 nplabels1")
    print(npimages1.shape)
    print(nplabels1.shape)

    print("Comparing ...")

    print(np.allclose(nplabels, nplabels1))
    print(np.allclose(npimages, npimages1))





main()

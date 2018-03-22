import tensorflow as tf

from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.xception import Xception
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import argparse
from time import strftime, time 
from os.path import join
import os


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

def create_test_file():
    images = [('st1035_sound_knot_52.PNG', 5), ('st1035_split_118.PNG', 3), ('st1050_sound_24.PNG', 0)]
    outputfile = '../data/test.tfrecords'
    inputpath = '../data/snapshots/testing'
    convert_to_tfrecord(images, inputpath, outputfile)
    
    
def scale_image(image):

    """Scales image pixesl between -1 and 1"""
    image = image / 127.5
    image = image - 1.
    return image

def convert_to_uint8(image):
    image = image + 1
    image = image * 127.5
    image = tf.cast(image, tf.uint8)
    return image

def _parse(example_proto, augment):

     
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image = tf.cast(image, tf.float32)
    image = scale_image(image)
    image = tf.reshape(image, IMAGE_SHAPE)
    
    if augment:
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_hue(image, max_delta=0.1)
        
    image = convert_to_uint8(image)
        
    label = parsed_features['label']
    label = tf.one_hot(label, NUM_CLASSES)

    return  image, label

  
def input_fn(file, train, batch_size=32, buffer_size=10000):
   
    if train:
        rep = None 
        augment = True
    else:
        rep = 1
        augment = False

    dataset = tf.data.TFRecordDataset(file)
    parse = lambda x: _parse(x, augment)
    dataset = dataset.map(parse)
    
    if train:
        dataset = dataset.shuffle(buffer_size)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(rep)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    
    return {"image": features}, labels

IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7

def main():

   
    file = '../data/test.tfrecords'
    
    dataset = tf.data.TFRecordDataset(file)
    parse = lambda x: _parse(x, True)
    dataset = dataset.map(parse)
    dataset = dataset.batch(1)
    dataset = dataset.repeat(None)
    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    
    with tf.Session() as sess:
        for i in range(200):
            results = sess.run(images)
            img = Image.fromarray(results[0])
            img.save('../data/images/img' + str(i) +'.PNG')
          
    
main()
        
        


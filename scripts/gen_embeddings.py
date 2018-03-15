import tensorflow as tf

from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import argparse
from time import strftime, time 
from os.path import join
import os


 



def VGG16convbase(image_shape, layer='last'):
  
    if layer == 'last':
        conv_base = VGG16(weights='imagenet',
                       include_top=False,
                       input_shape=image_shape)

    model = conv_base
    
    return model

def create_embeddings(images, model):
    images = scale(images)
    

    
def load_image(path, name):
    im = Image.open(join(path, name))
    return np.array(im)


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))
      
def scale_image(image):
    """Scales image pixesl between -1 and 1"""
    image = image / 127.5
    image = image - 1.
    return image

  
def convert_to_embeddings(file, input_dir, output_dir, model, batch_size=512 ):
    
    def parse_record(string_record):
        example = tf.train.Example()
        example.ParseFromString(string_record)

        image_feature = example.features.feature['image'].bytes_list.value[0]
        label_feature = example.features.feature['label'].int64_list.value[0]
        image = np.fromstring(image_feature, dtype=np.uint8)

        assert IMAGE_SHAPE[0]*IMAGE_SHAPE[1]*IMAGE_SHAPE[2] == image.shape[0]

        image = np.reshape(image, IMAGE_SHAPE)
        image = scale_image(image)
        label = label_feature
        return image, label
    
    def process_batch(batch, batch_size = batch_size):
       
        if len(batch) == batch_size:
            
            images = [item[0] for item in batch]
            images = np.array(images)
            embeddings = model.predict(images)
            new_shape = (embeddings.shape[0], EMBEDDING_SHAPE[0]*EMBEDDING_SHAPE[1]*EMBEDDING_SHAPE[2])
            embeddings = np.reshape(embeddings, new_shape)
    
            for i in range(embeddings.shape[0]):
                example= tf.train.Example(
                            features = tf.train.Features(
                                feature = {
                                    'embedding': _floats_feature(embeddings[i]),
                                    'label': _int64_feature(batch[i][1])
                                }))
                writer.write(example.SerializeToString())
                 
            batch = []
        return batch


        
   
    record_iterator = tf.python_io.tf_record_iterator(path=join(input_dir, file)) 
    outputfile =  'eb' +'_' + file 
    outputfile = join(output_dir, outputfile)
    with tf.python_io.TFRecordWriter(outputfile) as writer:
        
        batch = []
        for string_record in record_iterator:
            
            image, label = parse_record(string_record)
            batch.append((image, label))
            batch = process_batch(batch) 
               
        if len(batch) !=0:
            process_batch(batch, len(batch))

        



    
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7
INPUT_NAME = 'image'
OUTPUT_SUFFIX = 'embeddings.tfrecords'
EMBEDDING_SHAPE = (3, 3, 512)


def main(input_dir, output_dir):
    
    def parse_name(filename):
        last = filename.find('.')
        return filename[0: last]
   
  
    if not os.path.isdir(input_dir):
        print("Input directory {0} does not exist !!!".format(input_dir))
        return
    
    if not os.path.isdir(output_dir):
        print("Output directory {0} does not exist !!!".format(input_dir))
        return
    
    if len(os.listdir(output_dir)) != 0:
        print("Output directory not empty")
        return
    
    model = VGG16convbase(IMAGE_SHAPE, layer='last')
    for file in os.listdir(input_dir):
        print("Converting {0} from {1}".format(file, input_dir))
        convert_to_embeddings(file, input_dir, output_dir, model)
        
 
    
    
       
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Generating embeddings")
        
    parser.add_argument(
        '--input-dir',
        type=str,
        required = True,
        help='Input directory with image tfrecords')
    
    parser.add_argument(
        '--output-dir',
        type=str,
        required = True,
        help='Output directory for embedding tfrecords')
    
       
    args = parser.parse_args()
    
    main(args.input_dir, args.output_dir)

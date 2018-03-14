import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.estimator import model_to_estimator

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import argparse

import h5py

NUM_CLASSES = 7
IMAGE_SHAPE = (112, 112, 3)

def parse(serialized):
    # Define a dict with the schema reflecting the data in the TFRecords file
    features = \
        {
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.FixedLenFeature([], tf.int64)
        }
    
    # Parse the serialized data
    parsed_example = tf.parse_single_example(serialized=serialized,
                                             features=features)
    
    # Get the image as raw bytes
    image_raw = parsed_example['image']
    
    # Convert the raw bytes to tensorflow datatypes
    image = tf.decode_raw(image_raw, tf.uint8)
    image = tf.cast(image, tf.float32)
    image = image/255
    image = tf.reshape(image, IMAGE_SHAPE)
    print(image)
    
    # Get the label
    label = parsed_example['label']
    label = tf.one_hot(label, NUM_CLASSES)
    
    # Return the image and label as correct data types
    return image, label
    
    
def input_fn(filenames, train, batch_size=32, buffer_size=2048):
    # Create a TensorFlow Dataset-object which has functionality for reading and shuffling data 
    # from TFREcords files
    
    dataset = tf.data.TFRecordDataset(filenames=filenames)
    
    # Start building the pipeline
    # Parse
    dataset = dataset.map(parse)
    
    # Shuffle when training
    if train:
        # dataset = dataset.shuffle(buffer_size = buffer_size)
        # Allow infinite reading of the data
        num_repeat = None
    else:
        num_repeat = 1
        
    # Repeat the dataset the given number of times
    dataset = dataset.repeat(num_repeat)
    
    # Set the batch size
    dataset = dataset.batch(batch_size)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    images_batch, labels_batch = iterator.get_next()
    
    return {'image':images_batch}, labels_batch


def model_fn(image_shape, input_name):
    inputs = Input(shape=image_shape, name=input_name)
    x = Conv2D(32, (3, 3), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(7, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=y)

    model.compile(optimizer = Adadelta(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model




def main(mode, train_file, valid_file, ckpt_dir):
    print(train_file, valid_file, ckpt_dir)
    
    if mode == 'train':
        train_input_fn = lambda: input_fn(filenames=train_file, train=True)
        valid_input_fn = lambda: input_fn(filenames=valid_file, train=False)
        train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=5000)
        eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn)
        keras_model = model_fn(image_shape=IMAGE_SHAPE, input_name='image')
        keras_estimator = model_to_estimator(keras_model = keras_model, model_dir=ckpt_dir)
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)

    
    elif mode == 'train_evaluate':
        print('train_evaluate')
    elif mode == 'evaluate':
        print('evaluate')
    elif mode == 'predict':
        print('predict')
    else:
        print('else')

def input_fn1(images, labels, train, batch_size=32, buffer_size=2048):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
         
    # Repeat the dataset the given number of times
    if train == True:
        repeat = None
    else:
        repeat = 1
    dataset = dataset.repeat(repeat)
    
    # Set the batch size
    dataset = dataset.batch(batch_size)
    
    # Create an iterator
    iterator = dataset.make_one_shot_iterator()
    
    images_batch, labels_batch = iterator.get_next()
    
    return {'image':images_batch}, labels_batch


    
    

def main1(mode, train_file, valid_file, ckpt_dir):

    with h5py.File(train_file) as data:
      images = data['images'].value
      labels = data['labels'].value
      
      images = images[0:5000]
      labels = labels[0:5000]

      labels = tf.keras.utils.to_categorical(labels, 7)
      images = images/255


    assert images.shape[0] == labels.shape[0]


    train_input_fn = lambda: input_fn1(images, labels, train=True)
    keras_model = model_fn(image_shape=IMAGE_SHAPE, input_name='image')
    keras_estimator = model_to_estimator(keras_model = keras_model, model_dir=ckpt_dir)
    tf.logging.set_verbosity(tf.logging.INFO)
    
    keras_estimator.train(input_fn = train_input_fn, steps = 1000)
  

DEFAULT_TRAIN_FILE = '../data/tfrecords/training.tfrecords'
DEFAULT_VALIDATION_FILE = '../data/tfrecords/validation.tfrecords'

if __name__ == '__main__':
      parser = argparse.ArgumentParser("TFRecords file generator")
      parser.add_argument(
          '--mode',
          type=str,
          default='train',
          help='Training file')
      parser.add_argument(
          '--data',
          type=str,
          help='Training file')
      parser.add_argument(
          '--valid-file',
          type=str,
          help='Validation file')
      parser.add_argument(
          '--ckpt-dir',
          type=str,
          default='../checkpoints',
          help='Checkpoint dir')

      args = parser.parse_args()
      
      if args.mode == 'train':
          if args.data is None:
              train_data = DEFAULT_TRAIN_FILE
              test_data = DEFAULT_VALIDATION_FILE
      elif args.mode == 'validate':
          if args.data is None:
              data = DEFAULT_VALIDATION_FILE
      else:
          print("What to you want to do? train? validate?")

      mode = 'train'
      train = '../data/hdf5/training.h5'
      validate = '../data/hdf5/validation.h5'
      ckpt = '../tmp'


      main1(mode, train, validate, ckpt)

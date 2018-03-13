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

    
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7
INPUT_NAME = 'image'


def main(train_file, valid_file, ckpt_dir):

    train_input_fn = lambda: input_fn(filenames=train_file, train=True)
    valid_input_fn = lambda: input_fn(filenames=valid_file, train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=5000)
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn)

    keras_model = model_fn(image_shape=IMAGE_SHAPE, input_name=INPUT_NAME)
    keras_estimator = model_to_estimator(keras_model = keras_model, model_dir=ckpt_dir)

    tf.logging.set_verbosity(tf.logging.INFO)

    tf.estimator.train_and_evaluate(keras_estimator, train_spec, eval_spec)

   

if __name__ == '__main__':
      parser = argparse.ArgumentParser("TFRecords file generator")
      parser.add_argument(
          '--training',
          type=str,
          default = '../data/tfrecords/training.tfrecords',
          help='Training file')
      parser.add_argument(
          '--validation',
          type=str,
          default = '../data/tfrecords/validation.tfrecords',
          help='Validation file')
      parser.add_argument(
          '--ckpt',
          type=str,
          default='../checkpoints',
          help='Checkpoint dir')

      args = parser.parse_args()
      main(args.training, args.validation,  args.ckpt)

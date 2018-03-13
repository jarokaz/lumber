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

 
def parse(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image = image/255
    image = tf.reshape(image, IMAGE_SHAPE)
    label = parsed_features['label']
    label = tf.one_hot(label, NUM_CLASSES)

    return {'image': image}, label


  
def input_fn(file, train, batch_size=32, buffer_size=10000):
    if train:
        rep = None
    else:
        rep = 1 


    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(parse)
    dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(rep)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    
    
    return features, labels



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

    model.compile(optimizer = Adam(), 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model

    
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7
INPUT_NAME = 'image'


def main(train_file, valid_file, ckpt_dir):

    batch_size = 64
    print("Starting training: batch_size:{0}".format(batch_size))

    train_input_fn = lambda: input_fn(file=train_file, batch_size=batch_size, train=True)
    valid_input_fn = lambda: input_fn(file=valid_file, batch_size=batch_size, train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=15000)
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

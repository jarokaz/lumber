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

 
def _parse(example_proto, augment):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image = image/255
    image = tf.reshape(image, IMAGE_SHAPE)
    if augment:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_hue(image, max_delta=0.1)
    label = parsed_features['label']
    label = tf.one_hot(label, NUM_CLASSES)


    return  image, label


  
def input_fn(file, train, batch_size=32, buffer_size=10000, augment=False):
    
   
    dataset = tf.data.TFRecordDataset(file)
    
    parse = lambda x: _parse(x, augment)
    
    dataset = dataset.map(parse)
    if train:
        dataset = dataset.shuffle(buffer_size)
    
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(None if train else 1)

    iterator = dataset.make_one_shot_iterator()
    image, labels = iterator.get_next()
    
    return {'image': image}, labels


def simple_cnn_model_fn(image_shape, input_name, optimizer):
    
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

    model.compile(optimizer = optimizer, 
                  loss='categorical_crossentropy', 
                  metrics=['accuracy'])

    return model


def VGG16base(image_shape, input_name, optimizer):
  
    input = Input(shape=image_shape, name=input_name)
    conv_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=input)
    
    for layer in conv_base.layers:
        layer.trainable = False

    a = Flatten()(conv_base.output)
    a = Dense(256, activation='relu')(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=input, outputs=y)
    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    
    return model



    
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7
INPUT_NAME = 'image'


def main(model, train_file, valid_file, augment, ckpt_dir, opt, batch_size, max_steps,  lr):
   
    if not os.path.exists(train_file):
        print("Training file does not exist")
        return
    
    if not os.path.exists(valid_file):
        print("Validation file does not exist")
        return
    
    if not os.path.isdir(ckpt_dir):
        print("Checkpoint directory does not exist !!!")
        return
    
    if (augment != 1) and (augment != 0):
        print("Wrong augment value")
        return
    
    if opt == 'Adam':
        optimizer = Adam(lr = lr)
    else:
        print("Unsupported optimizer")
        return
   
    start_time = strftime('%d-%m-%H%M')
    ckpt_folder = join(ckpt_dir, model+'_'+start_time)
  
    if model == 'base':
        model_fn = simple_cnn_model_fn(image_shape=IMAGE_SHAPE, input_name=INPUT_NAME, optimizer = optimizer)      
    elif model == 'vgg16base':
        model_fn = VGG16base(image_shape=IMAGE_SHAPE, input_name=INPUT_NAME, optimizer=optimizer)
        
    else:
        print("Unsupported model")
        return
    
    
   
    
    estimator = model_to_estimator(keras_model = model_fn, model_dir=ckpt_folder)
    
    train_input_fn = lambda: input_fn(file=train_file, batch_size=batch_size, train=True, augment = augment)
    valid_input_fn = lambda: input_fn(file=valid_file, batch_size=batch_size, train=False, augment = False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)


    tf.logging.set_verbosity(tf.logging.INFO)
    
    summary_file = join(ckpt_dir, model+'_'+start_time+ '.txt' )
    with open(summary_file, 'w') as logfile:
        logfile.write("Training run started at: {0}\n".format(strftime('%c')))
        logfile.write("Model trained: {0}\n".format(model))
        logfile.write("Hyperparameters:\n")
        logfile.write("  Optimizer: {0}\n".format(opt))
        logfile.write("  Learning rate: {0}\n".format(lr))
        logfile.write("  Training file: {0}\n".format(train_file))
        logfile.write("  Validation file: {0}\n".format(valid_file))
        logfile.write("  Data augmentation: {0}\n".format('On' if augment==1 else 'Off'))          
        logfile.write("  Batch size: {0}\n".format(batch_size))
        logfile.write("  Max steps: {0}\n".format(max_steps))
 

    # Start training
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
   

if __name__ == '__main__':
      parser = argparse.ArgumentParser("Training, evaluation worklfow")
        
      parser.add_argument(
          '--model',
          type=str,
          default = 'base',
          help='Model to train')
    
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
    
      parser.add_argument(
          '--optimizer',
          type=str,
          default = 'Adam',
          help='Optimizer to use')
        
      parser.add_argument(
          '--batch-size',
          type=int,
          default=64,
          help='Batch size')
    
      parser.add_argument(
          '--max-steps',
          type=int,
          default=5000,
          help='Batch size')
        
      parser.add_argument(
          '--augment',
          type=int,
          default=0,
          help='Data augmentation: 1 - on, 0 - off') 
    
      parser.add_argument(
          '--lr',
          type=float,
          default=0.001,
          help='Learning rate') 
      
      args = parser.parse_args()
    
      main(args.model,
        args.training, 
        args.validation,
        args.augment, 
        args.ckpt, 
        args.optimizer, 
        args.batch_size, 
        args.max_steps,         
        args.lr)

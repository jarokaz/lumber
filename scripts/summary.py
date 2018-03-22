import tensorflow as tf

import numpy as np
import argparse
from time import strftime, time 
from os.path import join
import os

#import trainer.model as model

from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.vgg19 import VGG19
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers


IMAGE_SHAPE = (112, 112, 3,)
NUM_CLASSES = 7
INPUT_NAME = 'image'



def basenet(image_shape, input_name, hidden_units):

    inputs = Input(shape=image_shape, name=input_name)
    x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu')(inputs)
    x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(hidden_units, activation='relu')(x)
    x = Dropout(0.5)(x)
    y = Dense(NUM_CLASSES, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=y)

    return model
 
def vgg16base1(image_shape, input_name, hidden_units):
    
    x = Input(shape=image_shape, name=input_name)
    base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=x)
    
    for layer in base_model.layers:
        layer.trainable =  False
    
    conv_base = base_model.output
  
    a = Flatten()(conv_base)
    a = Dense(hidden_units, activation='relu')(a)
    a = Dropout(0.5)(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=x, outputs=y)
    
    return model
  
def vgg16base2(image_shape, input_name, hidden_units):
    x = Input(shape=image_shape, name=input_name)
    base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=x)
    
    for layer in base_model.layers:
        layer.trainable =  False
    
    conv_base = base_model.get_layer('block4_conv3').output

    a = MaxPooling2D(pool_size=(4,4))(conv_base) 
    a = Flatten()(a)
    a = Dense(hidden_units, activation='relu')(a)
    a = Dropout(0.5)(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=x, outputs=y)
    
    return model
  
def display_model_summary(model, hidden_units):
     
    if model == 'vgg16base1':
        model_fn =  vgg16base1(IMAGE_SHAPE, INPUT_NAME, hidden_units) 
    if model == 'vgg16base2':
        model_fn =  vgg16base2(IMAGE_SHAPE, INPUT_NAME, hidden_units) 
    elif model == 'basenet':
        model_fn = basenet(IMAGE_SHAPE, INPUT_NAME, hidden_units)

    model_fn.summary()



# Main entry
if __name__ == '__main__':
   parser = argparse.ArgumentParser("Training, evaluation worklfow")

    
   parser.add_argument(
        '--model',
        type=str,
        required=True,
        choices = ['basenet', 'vgg16base1', 'vgg16base2'],  
        help='Model to train')

   parser.add_argument(
        '--hidden_units',
        type=int,
        required=True,
        help='Hidden units')

   args = parser.parse_args()

   display_model_summary(args.model, args.hidden_units)

   

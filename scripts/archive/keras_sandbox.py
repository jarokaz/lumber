import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.callbacks import TensorBoard

from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import models
from tensorflow.python.keras import layers


import numpy as np
from time import time

from time import strftime, time 
from os.path import join
import os



def load_tfrecords(file):
    record_iterator = tf.python_io.tf_record_iterator(path=file)
    
    i=0
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
        label = label_feature
        
        images.append(image)
        labels.append(label)
       
    images = np.asarray(images)
    labels = np.asarray(labels)
    
    return  images, labels
    

def load_data_from_tfrecords(training_file, validation_file):
    
    x_train, y_train = load_tfrecords(training_file)
    x_test, y_test = load_tfrecords(validation_file)
    
    return x_train, y_train, x_test, y_test



def VGG16v1():
  
    input = Input(shape=IMAGE_SHAPE, name=INPUT_NAME)
    conv_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=input)
    
    for layer in conv_base.layers:
        layer.trainable = False

    a = Flatten()(conv_base.output)
    a = Dense(256, activation='relu')(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=input, outputs=y)
    
    conv_base.trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=Adadelta(),
                  metrics=['acc'])
    
    return 'VGG16v1FineTune', model


def VGG16v2():
    
    input = Input(shape=IMAGE_SHAPE, name=INPUT_NAME)
    conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_tensor = input)

    model = models.Sequential()
    model.add(conv_base)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    conv_base.trainable = False
    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  metrics=['acc'])
    
    model.summary()
    
    return 'VGG16v2', model

    
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7
TRAIN_TF = '../data/tfrecords/training.tfrecords'
VALID_TF = '../data/tfrecords/validation.tfrecords'
INPUT_NAME = 'my_input'

    
def main():
    


    x_train, y_train, x_test, y_test = load_data_from_tfrecords(TRAIN_TF, VALID_TF)
   
    y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    x_train = x_train/255

    y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)
    x_test = x_test/255 
    

    model_name, model_fun = VGG16v1()
    
      
    ckpt_dir = '../logs/'
    start_time = strftime('%d-%m-%H%M')
    ckpt_folder = join(ckpt_dir, model_name +'_'+start_time)
  
    tensorboard = TensorBoard(log_dir=ckpt_folder)

    model_fun.fit(x_train, y_train, 
                  validation_data = (x_test, y_test),
                  shuffle = True,
                  batch_size=32, epochs=10, verbose=1, callbacks=[tensorboard])


    #model.save(model_file)

main()



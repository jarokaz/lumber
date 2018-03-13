import tensorflow as tf
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.estimator import model_to_estimator

import numpy as np
import h5py
from time import strftime, time 
from os.path import join




def load_data_from_hdf5(training_file, validation_file):

    train = h5py.File(training_file)
    images = train['images'].value
    labels = train['labels'].value

    y_train = labels
    x_train = images
    
    test = h5py.File(validation_file)
    images = test['images'].value
    labels = test['labels'].value

    y_test = labels
    x_test = images

    return x_train, y_train, x_test, y_test



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


    
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7
TRAIN_TF = '../data/tfrecords/training.tfrecords'
VALID_TF = '../data/tfrecords/validation.tfrecords'
TRAIN_HD = '../data/hdf5/training.h5'
VALID_HD = '../data/hdf5/validation.h5'



def main():

    tf_x_train, tf_y_train, tf_x_test, tf_y_test = load_data_from_tfrecords(TRAIN_TF, VALID_TF)
    hd_x_train, hd_y_train, hd_x_test, hd_y_test = load_data_from_hdf5(TRAIN_HD, VALID_HD)

    print(tf_x_train.shape)
    print(tf_y_train.shape)

    print(tf_x_test.shape)
    print(tf_y_test.shape)

    print(hd_x_train.shape)
    print(hd_y_train.shape)

    print(hd_x_test.shape)
    print(hd_y_test.shape)

    print("Comparing ...")


    print(np.allclose(tf_x_train, hd_x_train))
    print(np.allclose(tf_y_train, hd_y_train))

main()







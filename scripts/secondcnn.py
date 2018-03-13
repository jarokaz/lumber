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


def main(mode):
  
    if mode == 'hdf5':
        print("Training from hdf5")
        training_file = '../data/hdf5/training.h5'
        validation_file = '../data/hdf5/validation.h5'
        model_path = '../models/hdf5'
        log_dir = '../logs/hdf5{0}'.format(strftime('%H%M%S'))        
        x_train, y_train, x_test, y_test = load_data_from_hdf5(training_file, validation_file)     
    elif mode == 'tfrecords':
        print("Training from TFRecords")
        training_file = '../data/tfrecords/training.tfrecords'
        validation_file = '../data/tfrecords/validation.tfrecords'
        model_path = '../models/tfrecords'
        log_dir = '../logs/tfrecors{0}'.format(strftime('%H%M%S')) 
        x_train, y_train, x_test, y_test = load_data_from_tfrecords(training_file, validation_file)
    else:
        print("Pitty ...")
        return
        
    tensorboard = TensorBoard(log_dir=log_dir)
    
    input_name = "image"

    model = model_fn(IMAGE_SHAPE, INPUT_NAME)

    x_train = x_train/255
    x_test = x_test/255
    
    y_train =  tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
    y_test =  tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

    model.fit(x_train, y_train, batch_size=32, epochs=5, verbose=1, callbacks=[tensorboard])
    
    model_file = join(model_path, 'cnn_{0}'.format(strftime('%H%M%S')))
    model.save(model_file)
    
    results = model.evaluate(x_test, y_test)
    print(results)


main('tfrecords')



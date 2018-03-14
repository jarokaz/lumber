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




def load_hdf5(file):

    data = h5py.File(file)
    images = data['images'].value
    labels = data['labels'].value

    labels = tf.keras.utils.to_categorical(labels, NUM_CLASSES)
    images = images/255

    return images, labels 



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
   

def parse(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image = image/255
    image = tf.reshape(image, IMAGE_SHAPE)
    label = parsed_features['label']
    label = tf.one_hot(label, NUM_CLASSES)

    return image, label




def loadtf_tfrecords(file):
    dataset = tf.data.TFRecordDataset(file)
    dataset = dataset.map(parse)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(1)
    iterator = dataset.make_one_shot_iterator()
    tf_image, tf_label = iterator.get_next()
    
    images = []
    labels = []
    

    with tf.Session() as sess:
        while True:
           try:
              image, label = sess.run([tf_image, tf_label]) 
              images.append(image[0])
              labels.append(label[0])
           except tf.errors.OutOfRangeError:
               print("Processed the file")
               break
      
    images = np.asarray(images)
    labels = np.asarray(labels)

    return images, labels

    
IMAGE_SHAPE = (112, 112, 3)
NUM_CLASSES = 7
TF_FILE = '../data/tfrecords/validation.tfrecords'
HD_FILE = '../data/hdf5/validation.h5'

def main():
  
    images_tf, labels_tf = loadtf_tfrecords(TF_FILE)
    images_hd, labels_hd = load_hdf5(HD_FILE)

    print("TF shapes:")
    print("labels {0}".format(labels_tf.shape))
    print("images {0}".format(images_tf.shape))

    print("HD shapes:")
    print("labels {0}".format(labels_hd.shape))
    print("images {0}".format(images_hd.shape))


    print("Comparing labels")
    print(np.allclose(labels_hd, labels_tf))


    print("Comparing images")
    print(np.allclose(images_hd, images_tf))

main()







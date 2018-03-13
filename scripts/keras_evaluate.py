import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.models import load_model

import numpy as np
import h5py

validation_file = '../data/hdf5/validation.h5'
model_file = '../models/firstcnn_v3.h5'

test = h5py.File(validation_file)
images = test['images'].value
labels = test['labels'].value

y_test = tf.keras.utils.to_categorical(labels, 7)
x_test = images/255


model = load_model(model_file) 
results = model.evaluate(x_test, y_test, verbose=1)
print(results)



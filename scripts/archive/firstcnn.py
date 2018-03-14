import tensorflow as tf
from tensorflow.python.keras.datasets import mnist, cifar10
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.optimizers import Adadelta
from tensorflow.python.keras.callbacks import TensorBoard

import numpy as np
import h5py
from time import time

training_file = '../data/hdf5/test.h5'
validation_file = '../data/hdf5/validation.h5'
model_file = '../models/firstcnn_v4.h5'

train = h5py.File(training_file)
images = train['images'].value
labels = train['labels'].value

y_train = tf.keras.utils.to_categorical(labels, 7)
x_train = images/255

test = h5py.File(validation_file)
images = test['images'].value
labels = test['labels'].value

y_test = tf.keras.utils.to_categorical(labels, 7)
x_test = images/255

inputs = Input(shape=(112, 112, 3))
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

tensorboard = TensorBoard(log_dir='../logs/{0}'.format(time()))

model.fit(x_train, y_train, batch_size=32, epochs=25, verbose=1, callbacks=[tensorboard])


model.save(model_file)


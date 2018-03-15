import tensorflow as tf

from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras import regularizers

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread
import numpy as np
import argparse
from time import strftime, time 
from os.path import join
import os

 
def parse(example_proto):

    features = {"embedding": tf.FixedLenFeature([INPUT_SHAPE[0]], tf.float32),
                "label": tf.FixedLenFeature([], tf.int64)}

    parsed_features = tf.parse_single_example(example_proto, features)
    
    label = parsed_features['label']
    embedding = parsed_features['embedding']
    label = tf.one_hot(label, NUM_CLASSES)
    
    return  embedding, label


  
def input_fn(file, train, batch_size=32, buffer_size=10000):
    
   
    dataset = tf.data.TFRecordDataset(file)
    
    
    dataset = dataset.map(parse)
      
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(None if train else 1)

    iterator = dataset.make_one_shot_iterator()
    embeddings, labels = iterator.get_next()
    
    print(embeddings.shape)
    
    return {'embedding':embeddings}, labels


def FCNN(input_shape, input_name, optimizer):
  
    x = Input(shape=INPUT_SHAPE, name=INPUT_NAME)
    
    a = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01))(x)
    a = Dense(512, activation='relu',kernel_regularizer=regularizers.l2(0.01))(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
     
    
    model = Model(x, y)

    
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
    
    return model



    
INPUT_SHAPE = (4608,)
NUM_CLASSES = 7
INPUT_NAME = 'embedding'


def main(model, train_file, valid_file, ckpt_dir, opt, batch_size, max_steps,  lr):
   
    if not os.path.exists(train_file):
        print("Training file does not exist")
        return
    
    if not os.path.exists(valid_file):
        print("Validation file does not exist")
        return
    
    if not os.path.isdir(ckpt_dir):
        print("Checkpoint directory does not exist !!!")
        return
        
    if opt == 'Adam':
        optimizer = Adam(lr = lr)
    else:
        print("Unsupported optimizer")
        return
   
    start_time = strftime('%d-%m-%H%M')
    ckpt_folder = join(ckpt_dir, model+'_'+start_time)
  
    if model == 'FCNN':
        model_fn =  FCNN(INPUT_SHAPE, INPUT_NAME, optimizer)      
    else:
        print("Unsupported model")
        return
    
    
    
    estimator = model_to_estimator(keras_model = model_fn, model_dir=ckpt_folder)
    
    
    
    train_input_fn = lambda: input_fn(file=train_file, batch_size=batch_size, train=True)
    valid_input_fn = lambda: input_fn(file=valid_file, batch_size=batch_size, train=False)

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
        logfile.write("  Batch size: {0}\n".format(batch_size))
        logfile.write("  Max steps: {0}\n".format(max_steps))
 
 
    # Start training
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
   

if __name__ == '__main__':
      parser = argparse.ArgumentParser("Training, evaluation worklfow")
        
      parser.add_argument(
          '--model',
          type=str,
          default = 'FCNN',
          help='Model to train')
    
      parser.add_argument(
          '--training',
          type=str,
          default = '../data/embeddings/eb_training.tfrecords',
          help='Training file')
        
      parser.add_argument(
          '--validation',
          type=str,
          default = '../data/embeddings/eb_validation.tfrecords',
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
          '--lr',
          type=float,
          default=0.001,
          help='Learning rate') 
      
      args = parser.parse_args()
    
      main(args.model,
        args.training, 
        args.validation,
        args.ckpt, 
        args.optimizer, 
        args.batch_size, 
        args.max_steps,         
        args.lr)

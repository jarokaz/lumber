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


  
def input_fn(file, train, batch_size=64, buffer_size=10000):
    
   
    dataset = tf.data.TFRecordDataset(file)  
    dataset = dataset.map(parse)    
    
    if train:
        dataset = dataset.shuffle(buffer_size)

    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(None if train else 1)

    iterator = dataset.make_one_shot_iterator()
    embeddings, labels = iterator.get_next()
    
    return {'embedding':embeddings}, labels


def FCN1(input_shape, input_name, optimizer, loss, metrics, regularizer):
      
    x = Input(shape=INPUT_SHAPE, name=INPUT_NAME)
    a = Dense(1024, activation='relu', kernel_regularizer=regularizer)(x)
    y = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizer, name='Softmax')(a)
     
    model = Model(x, y, name='FCN1')
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics) 
    return model


def FCN2(input_shape, input_name, optimizer, loss, metrics, regularizer):    
    
    x = Input(shape=INPUT_SHAPE, name=INPUT_NAME) 
    a = Dense(1024, activation='relu', kernel_regularizer=regularizer)(x)
    a = Dense(1024, activation='relu', kernel_regularizer=regularizer)(x)
    y = Dense(NUM_CLASSES, activation='softmax', kernel_regularizer=regularizer, name='Softmax')(a)
     
    model = Model(x, y, name='FCN2')
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)  
    return model



def my_train_and_evaluate(model_fn, train_file, valid_file, ckpt_folder, batch_size, max_steps):
    
    estimator = model_to_estimator(keras_model = model_fn, model_dir=ckpt_folder)
    
    train_input_fn = lambda: input_fn(file=train_file, batch_size=batch_size, train=True)
    valid_input_fn = lambda: input_fn(file=valid_file, batch_size=batch_size, train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)


    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
   
    
    
INPUT_SHAPE = (4608,)
NUM_CLASSES = 7
INPUT_NAME = 'embedding'


def main(model, train_file, valid_file, ckpt_folder, optimizer, batch_size, max_steps,  lr, l2):
   
    if optimizer == 'Adam':
        optimizer = Adam(lr = lr)
        
    if l2 > 0:
        regularizer = regularizers.l2(l2)
    else:
        regularizer = None

    metrics = ['categorical_accuracy']
    loss = 'categorical_crossentropy'
    
    if model == 'FCN1':
        model_fn =  FCN1(INPUT_SHAPE, INPUT_NAME, optimizer, loss, metrics, regularizer)  
    elif model == 'FCN2':
        model_fn =  FCN2(INPUT_SHAPE, INPUT_NAME, optimizer, loss, metrics, regularizer) 
       
    # Start training  
    my_train_and_evaluate(model_fn = model_fn, 
                          train_file = train_file,
                          valid_file = valid_file,
                          ckpt_folder = ckpt_folder,
                          batch_size = batch_size,
                          max_steps = max_steps)  

 
# Main entry
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training, evaluation worklfow")
        
    parser.add_argument(
          '--model',
          type=str,
          default = 'FCN1',
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
    
    parser.add_argument(
          '--l2',
          type=float,
          default=0,
          help='L2 regularization')
        
  
    args = parser.parse_args()

                          
    if not os.path.exists(args.training):
        print("Training file does not exist")
        exit()
    
    if not os.path.exists(args.validation):
        print("Validation file does not exist")
        exit()
    
    if not os.path.isdir(args.ckpt):
        print("Checkpoint directory does not exist !!!")
        exit()
        
    if args.optimizer not in ['Adam']:
        print("Unsupported optimizer")
        exit()
   
    if args.model not in ['FCN1', 'FCN2']:
        print("Unsupported model")
        exit()
                                                  
    start_time = strftime('%d-%m-%H%M')
    ckpt_folder = join(args.ckpt, args.model + '_' + start_time)                
    summary_file = join(args.ckpt, args.model + '_' + start_time + '.txt' )

    # Logg training parameters
    with open(summary_file, 'w') as logfile:
        logfile.write("Training run started at: {0}\n".format(strftime('%c')))
        logfile.write("Model trained: {0}\n".format(args.model))
        logfile.write("Hyperparameters:\n")
        logfile.write("  Optimizer: {0}\n".format(args.optimizer))
        logfile.write("  Learning rate: {0}\n".format(args.lr))
        logfile.write("  L2 regularization: {0}\n".format(args.l2))
        logfile.write("  Training file: {0}\n".format(args.training))
        logfile.write("  Validation file: {0}\n".format(args.validation))         
        logfile.write("  Batch size: {0}\n".format(args.batch_size))
        logfile.write("  Max steps: {0}\n".format(args.max_steps))
   

    main(args.model,
         args.training,
         args.validation,
         ckpt_folder,
         args.optimizer,
         args.batch_size,
         args.max_steps,
         args.lr,
         args.l2         
        )


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


 
def scale_image(image):
    """Scales image pixesl between -1 and 1"""
    image = image / 127.5
    image = image - 1.
    return image


def _parse(example_proto, augment):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    image = scale_image(image)
    image = tf.reshape(image, IMAGE_SHAPE)
    
    if augment:
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_hue(image, max_delta=0.1)
        
    label = parsed_features['label']
    label = tf.one_hot(label, NUM_CLASSES)

    return  image, label

  
def input_fn(file, train, batch_size=32, buffer_size=10000, augment=False):
   
    rep = None if train else 1

    dataset = tf.data.TFRecordDataset(file)
    parse = lambda x: _parse(x, augment)
    dataset = dataset.map(parse)
    
    if train:
        dataset = dataset.shuffle(buffer_size)
        
    dataset = dataset.batch(batch_size)
    dataset = dataset.repeat(rep)

    iterator = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    
    return {"image": features}, labels


def VGG16Full(image_shape, input_name, optimizer, hidden_units=1024,  pretrain=False):
    
    input = Input(shape=IMAGE_SHAPE, name=INPUT_NAME)
    conv_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=input)
    
    for layer in conv_base.layers:
        layer.trainable = False if pretrain else True

    a = Flatten()(conv_base.output)
    a = Dense(hidden_units, activation='relu')(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=input, outputs=y)
    
    conv_base.trainable = False
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['acc'])
 
    return model



def my_train_and_evaluate(model_fn, train_file, valid_file, ckpt_folder, batch_size, max_steps):
    
    estimator = model_to_estimator(keras_model = model_fn, model_dir=ckpt_folder)
    
    train_input_fn = lambda: input_fn(file=train_file, batch_size=batch_size, train=True)
    valid_input_fn = lambda: input_fn(file=valid_file, batch_size=batch_size, train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
   
    
    
INPUT_SHAPE = (4608,)
NUM_CLASSES = 7
INPUT_NAME = 'embedding'


def main(args):
    
    if args.optimizer == 'Adam':
        optimizer = optimizer, Adam(lr = lr)
   
    # Create model function
    if args.model == 'VGG16Full':
        model_fn =  VGG16Full(image_shape, input_name, hidden_units=args.hidden_units,  pretrain=args.pretrain)  
     
    # Start training
    start_time = strftime('%d-%m-%H%M')
    ckpt_folder = join(args.ckpt, args.model + '_' + start_time)

    return
    my_train_and_evaluate(model_fn = model_fn, 
                          train_file = args.training,
                          valid_file = args.validation,
                          ckpt_folder = ckpt_folder,
                          batch_size = args.batch_size,
                          max_steps = args.max_steps
    
# Main entry
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training, evaluation worklfow")
        
    parser.add_argument(
          '--model',
          type=str,
          default = 'VGG16Full',
          help='Model to train')
    
    parser.add_argument(
          '--training',
          type=str,
          default = '../data/tfrecords/training.tfrecords',
          help='Training file')
        
    parser.add_argument(
          '--validation',
          type=str,
          default = '../data/trfrecords/validation.tfrecords',
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
        
    parser.add_argument(
          '--hidden-units',
          type=int,
          default=0,
          help='Hidden units')     
    
    parser.add_argument(
          '--pretrain',
          type=int,
          default=1,
          help='Pretrain')   
      
    args = parser.parse_args()

                          
    if not os.path.exists(args.training):
        print("Training file does not exist")
        return
    
    if not os.path.exists(args.validation):
        print("Validation file does not exist")
        return
    
    if not os.path.isdir(args.ckpt):
        print("Checkpoint directory does not exist !!!")
        return
        
    if args.optimizer == 'Adam':
        optimizer = optimizer, Adam(lr = lr)
    else:
        print("Unsupported optimizer")
        return
   
    if args.model != 'VGG16Full':
        print("Unsupported model")
        return
                                                  
                          
    summary_file = join(args.ckpt, args.model + '_' + start_time + '.txt' )

    # Logg training parameters
    with open(parameters(summary_file), 'w') as logfile:
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
        logfile.write("  Hidden units: {0}\n".format(args.hidden_units))
        logfile.write("  Pretraining: {0}\n".format('Yes' if args.pretrain else 'No'))
    
    
    #tf.logging.set_verbosity(tf.logging.INFO)

    #main(args)

import tensorflow as tf

from tensorflow.python.keras.optimizers import Adadelta, Adam
from tensorflow.python.keras.estimator import model_to_estimator
from tensorflow.python.keras import Model, Input
from tensorflow.python.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.applications.xception import Xception
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
    image = tf.cast(image, tf.float32)
    image = scale_image(image)
    image = tf.reshape(image, IMAGE_SHAPE)
    
    if augment:
      image = tf.image.random_flip_left_right(image)
      image = tf.image.random_hue(image, max_delta=0.1)
        
    label = parsed_features['label']
    label = tf.one_hot(label, NUM_CLASSES)

    return  image, label

  
def input_fn(file, train, batch_size=32, buffer_size=10000):
   
    if train:
        rep = None 
        augment = True
    else:
        rep = 1
        augment = False

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


def basenet(image_shape, input_name):

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

    return model

 
def vgg16_trunk(image_shape, input_name):
      
    x = Input(shape=image_shape, name=input_name)
    base_model = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=x)
    
    for layer in base_model.layers:
        layer.trainable =  False

    conv_base = base_model.output
   
    a = Flatten()(conv_base)
    a = Dense(1024, activation='relu')(a)
    a = Dropout(0.5)(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=x, outputs=y)
    
    return model


def xception_trunk(image_shape, input_name):
      
    x = Input(shape=image_shape, name=input_name)
    base_model = Xception(weights='imagenet',
                   include_top=False,
                   input_tensor=x)
    
    for layer in base_model.layers:
        layer.trainable =  False

    a = Flatten()(conv_base)
    a = Dense(1024, activation='relu')(a)
    a = Dropout(0.5)(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=x, outputs=y)

    return model

   
def display_model_summary(model):
     
    if model == 'vgg16':
        model_fn =  vgg16_trunk(IMAGE_SHAPE, INPUT_NAME) 
    elif model == 'xception':
        model_fn = xception_trunk(IMAGE_SHAPE, INPUT_NAME)
    elif model == 'basenet':
        model_fn = basenet(IMAGE_SHAPE, INPUT_NAME)

    model_fn.summary()



def my_train_and_evaluate(model_fn, train_file, valid_file, ckpt_folder, batch_size, max_steps):
    
    estimator = model_to_estimator(keras_model = model_fn, model_dir=ckpt_folder)
    
    train_input_fn = lambda: input_fn(file=train_file, batch_size=batch_size, train=True)
    valid_input_fn = lambda: input_fn(file=valid_file, batch_size=batch_size, train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, steps=None)

    tf.logging.set_verbosity(tf.logging.INFO)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

    
IMAGE_SHAPE = (112, 112, 3,)
NUM_CLASSES = 7
INPUT_NAME = 'image'

def main(model, training, validation, ckpt_dir, optimizer, batch_size, max_steps, lr):
         
    if optimizer == 'Adam':
        optimizer = Adam(lr = lr)

    metrics = ['categorical_accuracy']
    loss = 'categorical_crossentropy'
    
    if model == 'vgg16':
        model_fn =  vgg16_trunk(IMAGE_SHAPE, INPUT_NAME) 
    elif model == 'xception':
        model_fn = xception_trunk(IMAGE_SHAPE, INPUT_NAME)
    elif model == 'basenet':
        model_fn = basenet(IMAGE_SHAPE, INPUT_NAME)
     
    model_fn.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=metrics)
 
    
    # Start training  
    my_train_and_evaluate(model_fn = model_fn, 
                          train_file = training,
                          valid_file = validation,
                          ckpt_folder = ckpt_folder,
                          batch_size = batch_size,
                          max_steps = max_steps)  

 
# Main entry
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training, evaluation worklfow")
        
    parser.add_argument(
        '--model',
        type=str,
        default = 'basenet',
        choices = ['basenet', 'vgg16', 'xception'],  
        help='Model to train')

    parser.add_argument(
        '--summary',
        action='store_true',
        help='Display model summary')
    
    parser.add_argument(
        '--training',
        type=str,
        default = '../data/tfrecords/training.tfrecords',
        help='Training file')
        
    parser.add_argument(
        '--validation',
        type=str,
        default = '../data/tfrecords/validation.tfrecords',
        help='Validation file')
      
    parser.add_argument(
        '--ckpt_dir',
        type=str,
        default='../checkpoints',
        help='Checkpoint dir')
      
    parser.add_argument(
        '--ckpt',
        type=str,
        help='The existing checkpoint to start with')
        
    parser.add_argument(
        '--optimizer',
        type=str,
        default = 'Adam',
        choices = ['Adam'],
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
        help='Max steps')
        
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate') 
  
    args = parser.parse_args()
                          
    if not os.path.exists(args.training):
        print("Training file {0} does not exist".format(args.training))
        exit()
    
    if not os.path.exists(args.validation):
        print("Validation file {0} does not exist.format(args.validation)")
        exit()
    
    if not os.path.isdir(args.ckpt_dir):
        print("Checkpoint directory {0} does not exist.".format(args.ckpt_folder))
        exit()
     
    if args.ckpt != None and not os.path.isdir(join(args.ckpt_dir, args.ckpt)):
        print("Checkpoint {0} does not exist.".format(args.ckpt))
        exit()

    if args.summary:
       display_model_summary(args.model) 
       exit()

                                                 
    start_time = strftime('%d-%m-%H%M')
    if args.ckpt != None:
        ckpt_folder = join(args.ckpt_dir, args.ckpt)
        summary_file = ckpt_folder + '.txt' 
    else:
        ckpt_folder = join(args.ckpt_dir, args.model + '_' + start_time)                
        summary_file = join(args.ckpt_dir, args.model + '_' + start_time + '.txt' )


    # Logg training parameters
    with open(summary_file, 'a+') as logfile:
        logfile.write("Training run started at: {0}\n".format(strftime('%c')))
        logfile.write("Model trained: {0}\n".format(args.model))
        logfile.write("Hyperparameters:\n")
        logfile.write("  Optimizer: {0}\n".format(args.optimizer))
        logfile.write("  Learning rate: {0}\n".format(args.lr))
        logfile.write("  Training file: {0}\n".format(args.training))
        logfile.write("  Validation file: {0}\n".format(args.validation))         
        logfile.write("  Batch size: {0}\n".format(args.batch_size))
        logfile.write("  Max steps: {0}\n".format(args.max_steps))
       
        if (args.ckpt == None):
            logfile.write("  Starting from scratch. New checkpoint folder {0} created\n".format(ckpt_folder))
        else:
            logfile.write("  Restarting training using the last checkpoint in {0} folder\n".format(ckpt_folder))
            
    main(args.model,
        args.training,
        args.validation,
        ckpt_folder,
        args.optimizer,
        args.batch_size,
        args.max_steps,
        args.lr)

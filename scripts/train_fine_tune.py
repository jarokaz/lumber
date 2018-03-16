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



def VGG16Full(image_shape, input_name, optimizer, loss, metrics, fine_tuning=False):
      
    input = Input(shape=image_shape, name=input_name)
    conv_base = VGG16(weights='imagenet',
                   include_top=False,
                   input_tensor=input)
    
    for layer in conv_base.layers:
        layer.trainable = True if fine_tuning else False

    a = Flatten()(conv_base.output)
    a = Dense(1024, activation='relu')(a)
    y = Dense(NUM_CLASSES, activation='softmax')(a)
    
    model = Model(inputs=input, outputs=y)
    
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
   
    
    
IMAGE_SHAPE = (112, 112, 3,)
NUM_CLASSES = 7
INPUT_NAME = 'image'


def main(model, train_file, valid_file, ckpt_folder, optimizer, batch_size, max_steps,  lr, l2, fine_tuning):
   
    if optimizer == 'Adam':
        optimizer = Adam(lr = lr)
        
    if l2 > 0:
        regularizer = regularizers.l2(l2)
    else:
        regularizer = None

    metrics = ['categorical_accuracy']
    loss = 'categorical_crossentropy'
    
    if model == 'VGG16Full':
        model_fn =  VGG16Full(IMAGE_SHAPE, INPUT_NAME, optimizer, loss, metrics, fine_tuning)  
        
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
          default = '../data/tfrecords/validation.tfrecords',
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
          help='Max steps')
        
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
          '--fine_tuning',
          type=float,
          default=0,
          help='Finie tuning mode')
        
  
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
   
    if args.model not in ['VGG16Full']:
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
        logfile.write("  Fine tuning: {0}\n".format('Yes' if args.fine_tuning == 1 else 'No'))
   

    main(args.model,
         args.training,
         args.validation,
         ckpt_folder,
         args.optimizer,
         args.batch_size,
         args.max_steps,
         args.lr,
         args.l2,
         args.fine_tuning
        )


import tensorflow as tf

import numpy as np
import argparse
from time import strftime, time 
from os.path import join, split
import os

from trainer.model import display_model_summary, model_fn

 
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

  
def input_fn(file, train, batch_size, buffer_size=10000):
   
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

def serving_input_fn():
    input_image = tf.placeholder(shape=INPUT_SHAPE, dtype=tf.uint8)
    image = tf.cast(input_image, tf.float32)
    scaled_image = scale_image(image)
    
    return tf.estimator.export.ServingInputReceiver({'image': scaled_image}, {'image': input_image})

    
IMAGE_SHAPE = (112, 112, 3,)
NUM_CLASSES = 7
INPUT_NAME = 'image'
INPUT_SHAPE = (None, 112, 112, 3)


def train_evaluate(model_name, hidden_units, train_file, valid_file, ckpt_folder, optimizer, batch_size, max_steps, lr, eval_steps):
    
    estimator = model_fn(model_name, hidden_units, ckpt_folder, optimizer, lr)
    
    train_input_fn = lambda: input_fn(file=train_file, batch_size=batch_size, train=True)
    valid_input_fn = lambda: input_fn(file=valid_file, batch_size=batch_size, train=False)

    train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=max_steps)
    
    export_latest = tf.estimator.FinalExporter("bclassifier", serving_input_fn)
    eval_spec = tf.estimator.EvalSpec(input_fn=valid_input_fn, 
                                      steps=eval_steps,
                                      exporters=export_latest)

    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)



# Main entry
if __name__ == '__main__':
    parser = argparse.ArgumentParser("Training, evaluation worklfow")

    ### Model parameters
    
    parser.add_argument(
        '--model',
        type=str,
        default = 'basenet',
        choices = ['basenet', 'vgg16base1', 'vgg16base2'],  
        help='Model to train')

    parser.add_argument(
        '--hidden-units',
        type=int,
        default = 128,
        help='Hidden units')

    ### Trainer parameters
           
    parser.add_argument(
        '--optimizer',
        type=str,
        default = 'Adam',
        choices = ['Adam'],
        help='Optimizer to use')
        
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate') 
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=64,
        help='Batch size')


    ### Training session parameters
    parser.add_argument(
        '--max-steps',
        type=int,
        default=5000,
        help='Max steps')
       
    parser.add_argument(
        '--training-file',
        required=True,
        help='Training file')
        
    parser.add_argument(
        '--validation-file',
        required=True,
        help='Validation file')
      
    parser.add_argument(
        '--job-dir',
        required=True,
        help='Job dir')

    parser.add_argument(
        '--job-name',
        required=True,
        help='Job dir')
 
    parser.add_argument(
        '--summary-dir',
        required=True,
        help='Summary dir')
  
    parser.add_argument(
        '--eval-steps',
        type=int,
        default = 1000,
        help='Number of steps to run evaluation for at each checkpoint')
    
    ### Utility commends
    
    parser.add_argument(
        '--summary',
        action='store_true',
        help='Display model summary')
    
    parser.add_argument(
        '--verbosity',
        default = 'INFO',
        choices = [
          'DEBUG',
          'ERROR',
          'FATAL',
          'INFO',
          'WARN'],
        help='Control logging level')

    args = parser.parse_args()
                          
    if not os.path.exists(args.training_file):
        print("Training file {0} does not exist".format(args.training_file))
        exit()
    
    if not os.path.exists(args.validation_file):
        print("Validation file {0} does not exist.format(args.validation_file)")
        exit()

    if not os.path.isdir(args.job_dir):
        print("Job directory {0} does not exist.".format(args.job_dir))
        exit()

    if args.summary:
       display_model_summary(args.model, args.hidden_units) 
       exit()
    
    summary_file = join(args.summary_dir, args.job_name + '.txt') 

    # Logg training parameters
    with open(summary_file, 'a+') as logfile:
        logfile.write("Training run started at: {0}. Job: {1}\n".format(strftime('%c'), args.job_name))
        logfile.write("Model parameters:\n")
        logfile.write("  Model trained: {0}\n".format(args.model))
        logfile.write("  Hidden units: {0}\n".format(args.hidden_units))
        logfile.write("Trainer parameters:\n")
        logfile.write("  Optimizer: {0}\n".format(args.optimizer))
        logfile.write("  Learning rate: {0}\n".format(args.lr))
        logfile.write("  Batch size: {0}\n".format(args.batch_size))
        logfile.write("Training session parameters:\n")
        logfile.write("  Training file: {0}\n".format(args.training_file))
        logfile.write("  Validation file: {0}\n".format(args.validation_file))         
        logfile.write("  Max steps: {0}\n".format(args.max_steps))
        logfile.write("  Eval steps: {0}\n".format(args.eval_steps))
       
    tf.logging.set_verbosity(args.verbosity)

    train_evaluate(model_name = args.model,
        hidden_units = args.hidden_units,
        train_file = args.training_file,
        valid_file = args.validation_file,
        ckpt_folder = args.job_dir,
        optimizer = args.optimizer,
        batch_size = args.batch_size,
        max_steps = args.max_steps,
        lr = args.lr,
        eval_steps = args.eval_steps)

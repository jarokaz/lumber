import tensorflow as tf
import numpy as np
from time import strftime, time 
from os.path import join

def parse(example_proto):
    features = {"image": tf.FixedLenFeature((), tf.string, default_value=""),
                "label": tf.FixedLenFeature((), tf.int64, default_value=0)}

    parsed_features = tf.parse_single_example(example_proto, features)
    image = tf.decode_raw(parsed_features['image'], tf.uint8)
    
    label = parsed_features['label']
    label = tf.one_hot(label, 7)

    return {'image': image}, label

def parse1(example_proto):
    features = {"embedding": tf.FixedLenFeature([4608], tf.float32),
                "label": tf.FixedLenFeature([], tf.int64)}

    parsed_features = tf.parse_single_example(example_proto, features)
    
    label = parsed_features['label']
    embedding = parsed_features['embedding']


    return  embedding, label



def main():
    FILE = '../data/embeddings/eb_validation.tfrecords'

    dataset = tf.data.TFRecordDataset(FILE)
    dataset = dataset.map(parse1)
    dataset = dataset.repeat(1)
    dataset = dataset.batch(2)
    iterator = dataset.make_one_shot_iterator()
    embedding, label = iterator.get_next()
    
    with tf.Session() as sess:
       result1 = sess.run(label)
       print(result1.shape)
       #print(type(result2))
        #print(result1)
        #print(type(result2))
        #print(result2)

    #with tf.Session() as sess:
    #   while True:
    #      try:
    #        image, label = sess.run([tf_image, tf_label]) 
    #        images.append(image[0])
    #        labels.append(label[0])
    #       except tf.errors.OutOfRangeError:
    #         print("Processed the file")
    #         break

    #images = np.asarray(images)
    #labels = np.asarray(labels)

main()




import numpy as np
import json
from PIL import Image
from os.path import join
import googleapiclient.discovery
import six


def load_images():
    images_dir = '../data/snapshots/testing'
    test_images = ['st1035_sound_knot_57.PNG', 
               'st1035_split_103.PNG', 
               'st1035_dry_knot_6.PNG', 
               'st1035_sound_10.PNG', 
               'st1035_sound_5.PNG',
               'st1035_sound_11.PNG']


    images = []
    labels = []
    for image_name in test_images:
        image = Image.open(join(images_dir, image_name))
        label = image_name[image_name.find('_')+1 : image_name.rfind('_')]
        images.append(np.array(image))
        labels.append(label)
                      
    return images, labels



def predict(project, model, image, version='v1'):
    
    image = image.tolist()
    instances = [{'image': image}]
  
    service = googleapiclient.discovery.build('ml', 'v1')
    name = 'projects/{}/models/{}'.format(project, model)

    if version is not None:
        name += '/versions/{}'.format(version)

    response = service.projects().predict(
        name=name,
        body={'instances': instances}
    ).execute()

    if 'error' in response:
        raise RuntimeError(response['error'])

    return response['predictions']


def classify_images(images):
    project = 'ferrous-kayak-198317'
    labels = ['sound', 'split', 'wane', 'sound knot', 'dry knot', 'core stripe', 'small knot']
    model = 'lumberclassifier'
    
    results =[]
    probs = []
    for image in images:
        result = predict(project, model, image)
        probabilities = result[0]['dense_2']
        argmax = probabilities.index(max(probabilities))
        results.append(labels[argmax])
        probs.append(probabilities[argmax])
        
    return results, probs
    
    

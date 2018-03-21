import numpy as np
import json
from PIL import Image
from os.path import join
import googleapiclient.discovery
import six


def load_images(images_dir, image_names):
   
    images = []
    labels = []
    for image_name in image_names:
        image = Image.open(join(images_dir, image_name))
        label = image_name[image_name.find('_')+1 : image_name.rfind('_')]
        images.append(np.array(image))
        labels.append(label)
                      
    return images, labels



def predict(project, model, images, version='v1'):
    
    instances = []
    for image in images:
     image_list = image.tolist()
     instances.append({'image': image_list})
  
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
    
    response = predict(project, model, images)
    for result in response:
        probabilities = result['dense_2']
        argmax = probabilities.index(max(probabilities))
        results.append(labels[argmax])
        probs.append(probabilities[argmax])
        
    return results, probs
    
    

import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

def create_hdf5(hdf5_filepath, path_to_images):
    files = [file for file in listdir(path_to_images) if file.endswith(".PNG")]
    images_shape = (len(files), 112, 112, 3)
    labels_shape = (len(files),)
    
    hdf5 = h5py.File(hdf5_filepath,  mode='w')
    images = hdf5.create_dataset("images", images_shape, np.uint8)
    labels = hdf5.create_dataset("labels", labels_shape, np.uint8)

    label_list = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']
    
    for i in range(len(files)):
        name = files[i]
        im = Image.open(join(path_to_images, name))
        label = name[name.find('_')+1 : name.rfind('_')]
        if label in label_list :
           labels[i] = label_list.index(label)+1  
           images[i] = np.array(im)
        if i%500 == 0 :
            print("Processing {0} file".format(i))
        
    hdf5.close()



path_to_training_images = '../data/training'
path_to_testing_images = '../data/testing'
training_file = '../data/hdf5/training.h5'
testing_file = '../data/hdf5/testing.h5'

print("Generating the training file")
create_hdf5(training_file, path_to_training_images)

print("Generating the testing file")
create_hdf5(testing_file, path_to_testing_images)




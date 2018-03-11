import h5py
import numpy as np
from os import listdir
from os.path import isfile, join
from PIL import Image

def create_hdf5(hdf5_filepath, path_to_images):
    label_list = ['sound', 'split', 'wane', 'sound_knot', 'dry_knot', 'core_stripe', 'small_knot']
    
    files = [file for file in listdir(path_to_images) if file.endswith(".PNG")]
    files = [file for file in files if file[file.find('_')+1 : file.rfind('_')] in label_list]
    
    images_shape = (len(files), 112, 112, 3)
    labels_shape = (len(files),)
    
    with h5py.File(hdf5_filepath,  mode='w') as hdf5:
    
        images = hdf5.create_dataset("images", images_shape, np.uint8)
        labels = hdf5.create_dataset("labels", labels_shape, np.uint8)

        for i in range(len(files)):
            name = files[i]
            im = Image.open(join(path_to_images, name))
            label = name[name.find('_')+1 : name.rfind('_')]
            if label in label_list :
               labels[i] = label_list.index(label) 
               images[i] = np.array(im)
            else:
                print('Label {0} not in supported labels'.format(label))
            if i%100 == 0 :
                print("Processing {0} file".format(i))



path_to_training_images = '../data/training'
path_to_testing_images = '../data/testing'
training_file = '../data/hdf5/training.h5'
testing_file = '../data/hdf5/testing.h5'

print("Generating the training file")
create_hdf5(training_file, path_to_training_images)

print("Generating the testing file")
create_hdf5(testing_file, path_to_testing_images)




import numpy as np
from PIL import Image
import argparse

def convert_array_to_string(image):
    e = 0
    img_str = '['
    for n in range(image.shape[0]):
        img_str += '['
        for i in range(image.shape[1]):
            img_str += '['
            for j in range(image.shape[2]):
                img_str += '['
                for c in range(image.shape[3]):
                    img_str += str(image[n,i,j,c]) + ','
                img_str = img_str[:-1]
                img_str += '],'
            img_str = img_str[:-1]
            img_str += '],'
        img_str = img_str[:-1]
        img_str += '],'
    img_str = img_str[:-1]
    img_str += ']'
    
    return img_str   

def create_json(img_str, tag):
    json_str = '{\"' + tag + '\"' + ": " +  img_str + "}"
    return json_str
            

def main(file, image_files):
    
    image_file = image_files.pop(0)
    image = Image.open(image_file)
    image = np.array(image)
    images = np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
    for image_file in image_files:
        image = Image.open(image_file)
        image = np.array(image)
        image = np.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
        images = np.append(images, image, axis=0)
        
    image_str = convert_array_to_string(images)
    image_str = create_json(image_str, 'image')
    
    with open(file, 'w') as json_file:
        json_file.write(image_str)
        
   
   

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Create json test cases")

    ### Model parameters
    
    parser.add_argument(
        '--file',
        type=str,
        required=True, 
        help='filename to save output to')

    parser.add_argument(
        '--images',
        type=str,
        required = True,
        nargs='*',
        help='images to convert')
    
    args = parser.parse_args()
    
    main(args.file, args.images)
    



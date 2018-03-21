''' Adding h5py to ML runtime '''
from setuptools import setup, find_packages

setup(name='bclassifier',
        version='0.9',
        packages=find_packages(),
        include_package_data=True,
        description='BClassifier using Keras models',
        author = 'JK',
        auther_email = 'jarokaz541@gmail.com',
        license = 'MIT',
        install_requires=['h5py'],
        zip_safe=False)



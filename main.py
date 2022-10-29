import pathlib
import numpy as np
import os
import tensorflow as tf
import PIL
from matplotlib import pyplot as plt
import tensorflow_datasets as tfds
dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))

roses = list(data_dir.glob('roses/*'))
img = PIL.Image.open(str(roses[1]))
img_np = np.array(img)

col, row  = img.size
#zero_matrix = np.random.normal(0,3,(row,col,3))

zero_matrix = np.zeros((row,col,3))
img_noisy = img_np+zero_matrix
img_noisy_jpg=PIL.Image.fromarray(img_noisy.astype('uint8'), 'RGB')
img_noisy_jpg.show()
#plt.imshow(img_noisy)

def forward ()
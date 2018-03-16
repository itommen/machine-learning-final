
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import tensorflow as tf


#Load weights
inception = InceptionV3(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()
from keras.models import load_model


def load_image_directory(path):
    images = []
    
    images_name = os.listdir(path)
    images_name.sort()
    
    for filename in images_name:
        images.append(img_to_array(load_img('{0}{1}'.format(path, filename))))
    images = np.array(images, dtype=float)
    
    return images

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = np.resize(i, (299, 299, 3))
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

images_to_predict =  load_image_directory('./input/')

normalized_imgs = 1.0/255 * images_to_predict
normalized_imgs =  rgb2lab(normalized_imgs)[:,:,:,0]
normalized_imgs = normalized_imgs.reshape(normalized_imgs.shape+(1,))

SRmodel = load_model(os.path.dirname(os.path.realpath(__file__))+'/SRmodel.h5')
CLRmodel = load_model(os.path.dirname(os.path.realpath(__file__))+'/ColorizeModel.h5')


srImages = SRmodel.predict(np.array(normalized_imgs))
color_embed = create_inception_embedding(srImages)
clrImages = CLRmodel.predict([srImages,color_embed])
for i in range(len(clrImages)):
    cur = np.zeros((96, 96, 3))
    cur[:,:,0] = np.reshape(srImages[i], (96,96))
    cur[:,:,1:] = clrImages[i] * 128
    imsave("results/image_"+str(i)+".png", lab2rgb(cur))

    
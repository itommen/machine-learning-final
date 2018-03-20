import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.engine import Layer
from keras.applications.inception_v3 import preprocess_input
from keras.layers import Conv2D, UpSampling2D, InputLayer, Conv2DTranspose, Input, Reshape, merge, concatenate
from keras.layers import Activation, Dense, Dropout, Flatten, InputLayer,Input
from keras.layers.normalization import BatchNormalization
from keras.callbacks import TensorBoard
from keras.models import Sequential, Model
from keras.layers.core import RepeatVector, Permute
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb, rgb2gray, gray2rgb
from skimage.transform import resize
from skimage.io import imsave
import numpy as np
import os
import random
import tensorflow as tf
from matplotlib import pyplot as plt

#Load weights
inception = InceptionV3(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()

print("finished loading inceptionv3")

###
# Get images

def load_image_directory(path):
    images = []
    
    images_name = os.listdir(path)
    images_name.sort()
    
    for filename in images_name:
        images.append(img_to_array(load_img('{0}{1}'.format(path, filename))))
    images = np.array(images, dtype=float)
    
    return images

test_images =  load_image_directory('./TrainFlowers/')[:50]
colored_images = load_image_directory('./TrainFlowers/')[50:]

Xtrain = 1.0/255 * colored_images
x = 1.0/255 * colored_images
y = 1.0/255* colored_images

imgs = 1.0/255 * test_images
imgs_to_predict =  rgb2lab(imgs)[:,:,:,0]
imgs_to_predict = imgs_to_predict.reshape(imgs_to_predict.shape+(1,))

#Load weights
inception = InceptionV3(weights='imagenet', include_top=True)
inception.graph = tf.get_default_graph()

x = rgb2lab(x)[:,:,:,0]
y = (rgb2lab(y)[:,:,:,1:]) / 128

x = x.reshape(x.shape+(1,))
#y = y.reshape(y.shape+(2,))
print("finished loading")
###
embed_input = Input(shape=(1000,))

#input here is 96x96 image, in our case the image after the super resolution
encoder_input = Input(shape=(96, 96, 1,))
encoder_output = Conv2D(64, (3,3), activation='relu', padding='same', strides=2)(encoder_input)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(128, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same', strides=2)(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(512, (3,3), activation='relu', padding='same')(encoder_output)
encoder_output = Conv2D(256, (3,3), activation='relu', padding='same')(encoder_output)

#input here is InceptionV3 final layer
fusion_output = RepeatVector(12 * 12)(embed_input) 
fusion_output = Reshape(([12, 12, 1000]))(fusion_output)
fusion_output = concatenate([encoder_output, fusion_output], axis=3) 
fusion_output = Conv2D(256, (1, 1), activation='relu', padding='same')(fusion_output) 

#input here are the encoder_output and fusion_output
decoder_output = Conv2D(128, (3,3), activation='relu', padding='same')(fusion_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(64, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)
decoder_output = Conv2D(32, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(16, (3,3), activation='relu', padding='same')(decoder_output)
decoder_output = Conv2D(2, (3, 3), activation='tanh', padding='same')(decoder_output)
decoder_output = UpSampling2D((2, 2))(decoder_output)

model = Model(inputs=[encoder_input, embed_input], outputs=decoder_output)

opt = keras.optimizers.adam(lr=0.00005, decay=1e-6)
model.compile(optimizer=opt, loss='mse')
print(model.summary())
print("created model")
##
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from IPython.display import clear_output

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

# Image transformer
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=20,
        horizontal_flip=True)

#Generate training data
batch_size = 100

def image_a_b_gen(batch_size):
    for batch in datagen.flow(Xtrain, batch_size=batch_size):
        grayscaled_rgb = gray2rgb(rgb2gray(batch))
        embed = create_inception_embedding(grayscaled_rgb)
        lab_batch = rgb2lab(batch)
        X_batch = lab_batch[:,:,:,0]
        X_batch = X_batch.reshape(X_batch.shape+(1,))
        Y_batch = lab_batch[:,:,:,1:] / 128
        yield ([X_batch, create_inception_embedding(grayscaled_rgb)], Y_batch)



class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.legend()
        plt.show();
        plt.pause(0.05)
        
plot_losses = PlotLosses()

filepath = 'colorize-inv3-ep{epoch:03d}-loss{loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')       
# Train model      
print("starting")

model.fit_generator(image_a_b_gen(batch_size), epochs=100, steps_per_epoch=80,
                   callbacks=[plot_losses,checkpoint])


print("finished")
###

from keras.models import load_model
from keras.callbacks import ModelCheckpoint
from matplotlib import pyplot as plt
from IPython.display import clear_output

def create_inception_embedding(grayscaled_rgb):
    grayscaled_rgb_resized = []
    for i in grayscaled_rgb:
        i = resize(i, (299, 299, 3), mode='constant')
        grayscaled_rgb_resized.append(i)
    grayscaled_rgb_resized = np.array(grayscaled_rgb_resized)
    grayscaled_rgb_resized = preprocess_input(grayscaled_rgb_resized)
    with inception.graph.as_default():
        embed = inception.predict(grayscaled_rgb_resized)
    return embed

model = load_model('./colorize-inv3-ep083-loss0.017.h5')
gray_me = gray2rgb(rgb2gray(1.0/255*test_images))
color_me_embed = create_inception_embedding(gray_me)
color_me = rgb2lab(1.0/255*test_images)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))


# Test model
output = model.predict([color_me, color_me_embed])
output = output * 128

# Output colorizations
for i in range(len(output)):
    cur = np.zeros((96, 96, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("results/clr3_img_"+str(i)+".png", lab2rgb(cur))
	
###
# Test images
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print(model.evaluate(Xtest, Ytest, batch_size=batch_size))
###
from keras.models import load_model
import matplotlib.pyplot as plt
model = load_model('./colorize-inv3-ep083-loss0.017.h5')
color_me = []

grey_images = load_image_directory('./BWFlowers/')[:50]
color_me = np.array(grey_images, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorizations
for i in range(5):
    cur = np.zeros((96, 96, 3))
    cur[:,:,0] = np.reshape(color_me[i])
    cur[:,:,1:] = output[i]
    print(output[i].shape)
    plt.imshow(lab2rgb(cur))
    plt.pause(0.05)
    
###
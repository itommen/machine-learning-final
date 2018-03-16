from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.models import Sequential, load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint
from scipy import ndimage, misc
import os
import re
import numpy as np
from numpy.random import shuffle
import argparse
from matplotlib import pyplot as plt
from IPython.display import clear_output

num_classes = 1
type_data_set_size = 1500
batch_size = 4
epochs = 2000
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

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
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();
        
plot_losses = PlotLosses()

flower_images = []
for root, dirnames, filenames in os.walk("./NewFlowers"):
    for filename in filenames:
        if re.search("\.(jpg|jpeg|png|bmp|tiff|gif)$", filename):
            filepath = os.path.join(root, filename)
            image = ndimage.imread(filepath, mode="RGB")
            flower_images.append(misc.imresize(image, (32, 32)))
            
flower_zipped_images = list(zip([1] * type_data_set_size, flower_images[:type_data_set_size]))
cifar10_zipped_images = list(zip([0] * type_data_set_size, x_train[:type_data_set_size]))            
data = flower_zipped_images + cifar10_zipped_images
shuffle(data)

x = [item[1] for item in data]
y = [item[0] for item in data]

train_size = int(type_data_set_size * 2 * 0.9)
test_size = int(type_data_set_size * 2 * 0.1)
x_train = np.array(x[:train_size])
y_train = np.array(y[:train_size])
x_test = np.array(x[train_size:train_size + test_size])
y_test = np.array(y[train_size:train_size + test_size])

print ('loading model')
loadedModel = load_model('cifar-bestmodel-ep339-loss0.504.h5')
print ('finished loading')

print(loadedModel.summary())
loadedModel.layers.pop()
loadedModel.layers.pop()
loadedModel.layers.pop()
loadedModel.layers.pop()
loadedModel.layers.pop()
print('--------------')
print(loadedModel.summary())

loadedModel.layers[-1].outbound_nodes = []
loadedModel.outputs = [loadedModel.layers[-1].output]

for layer in loadedModel.layers:
    layer.trainable = False


loadedModel.add(Dense(512))
loadedModel.add(Activation('relu'))
loadedModel.add(Dropout(0.5))
loadedModel.add(Dense(num_classes))
loadedModel.add(Activation('sigmoid'))
print('--------------')
print(loadedModel.summary())

# initiate RMSprop optimizer
opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

# Let's train the model using RMSprop
loadedModel.compile(loss='binary_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])

# define checkpoint callback
filepath = 'fine_tune-ep{epoch:03d}-loss{loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

loadedModel.fit(x_train, y_train,  # this is our training examples & labels
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.1,  # this parameter control the % of train data used for validation
                shuffle=True,
                callbacks=[plot_losses, checkpoint])  # this prints our loss at the end of every epoch

# Score trained model.
scores = loadedModel.evaluate(x_test, y_test, verbose=1)
print('==> Test loss:', scores[0])
print('==> Test accuracy:', scores[1])


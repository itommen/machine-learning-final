from keras.callbacks import ModelCheckpoint
from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import os
from keras.models import load_model
from keras.utils import multi_gpu_model
import argparse
from matplotlib import pyplot as plt
from IPython.display import clear_output
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

class CreateGraphs(keras.callbacks.Callback):
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

def getData():
	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	shrink_data = True
	if shrink_data:
		selected_classes = [2, 3, 5, 6, 7]
		print('train\n', x_train.shape, y_train.shape)
		
		x_train, y_train = filterBySelectedClasses(selected_classes, x_train, y_train)
		print(x_train.shape, y_train.shape)
		
		print('test\n', x_test.shape, y_test.shape)
		x_test, y_test = filterBySelectedClasses(selected_classes, x_test, y_test)
		print(x_test.shape, y_test.shape)
		num_classes = len(selected_classes)
	else:
		print('train\n', x_train.shape, y_train.shape)
		print('test\n', x_test.shape, y_test.shape)
		num_classes = 5
		
	return (x_train, y_train), (x_test, y_test), num_classes
		
def createModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                   input_shape=x_train.shape[1:]))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    return model
        
def filterBySelectedClasses(selected_classes, x_list, y_list):
    x = [ex for ex, ey in zip(x_list, y_list) if ey in selected_classes]
    y = [selected_classes.index(ey) for ex, ey in zip(x_list, y_list) if ey in selected_classes]
    
    return np.stack(x), np.stack(y).reshape(-1,1)

def getCheckPoint():
    filepath = 'cifar-bestmodel-ep{epoch:03d}-loss{loss:.3f}.h5'
    return ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')		
	
def getOptimizer():
	learning_rate = 0.00015
	decay = 1e-6
	return keras.optimizers.rmsprop(lr=learning_rate, decay=decay)
	
def convertToCategorizedArrray(y_train, y_test, num_classes):
	return keras.utils.to_categorical(y_train, num_classes), keras.utils.to_categorical(y_test, num_classes)
	
def testModel(x_test, y_test):
	loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
	print('# Test loss:', loss)
	print('# Test accuracy:', accuracy)
	
graphCallbacks = CreateGraphs()

(x_train, y_train), (x_test, y_test), num_classes = getData()

y_train, y_test = convertToCategorizedArrray(y_train, y_test, num_classes)

model = createModel()

opt = getOptimizer()

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])


x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# When finding a better model (better accuracy) than the previous saved one, update the saved model.
checkpoint = getCheckPoint();

batch_size = 150
epochs = 2

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.1, #the percentage of the validation from the training
          shuffle=True,
          callbacks=[graphCallbacks,checkpoint])

testModel(x_test, y_test)

import os
from pathlib import Path
import zipfile

import tensorflow as tf
tf_version = int(tf.__version__.split(".")[0])

from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Dropout


def loadModel(path):
	
	num_classes = 7

	model = Sequential()

	#1st convolution layer
	model.add(Conv2D(64, (5, 5), activation='relu', input_shape=(48,48,1)))
	model.add(MaxPooling2D(pool_size=(5,5), strides=(2, 2)))

	#2nd convolution layer
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	#3rd convolution layer
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(Conv2D(128, (3, 3), activation='relu'))
	model.add(AveragePooling2D(pool_size=(3,3), strides=(2, 2)))

	model.add(Flatten())

	#fully connected neural networks
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1024, activation='relu'))
	model.add(Dropout(0.2))

	model.add(Dense(num_classes, activation='softmax'))

	#---------	-------------------
	print(path)
	model.load_weights(path)

	return model

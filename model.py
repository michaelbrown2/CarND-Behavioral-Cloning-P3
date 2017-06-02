import csv
import numpy as np
import cv2
import random
import sklearn
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tensorflow as tf

from tensorflow.contrib.layers import flatten
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from numpy import newaxis
from scipy import ndimage
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.regularizers import l2, activity_l2

#SET Image Folder
data_folder = './dataset3/'

#GRAYSCALE function (currently unused...)
def grayscale(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = img.reshape(32,32,1)
    return(img)



#INPUT file names for processing
lines = []
with open(data_folder + 'driving_log.csv') as csvfile:
	read = csv.reader(csvfile)
	for line in read:
		lines.append(line)
		
#print(train_samples)
def gen(lines, batch_size=8):
	total_samples = len(lines)
	while 1:
		shuffle(lines)
		for offset in range(0, total_samples, batch_size):
			batch_samples = lines[offset:offset+batch_size]
			images = []
			measurements = []
			for batch_sample in batch_samples:
				#APPEND center images
				center_path = batch_sample[0]
				center_filename = center_path.split('/')[-1]
				center_current = data_folder + 'IMG/' + center_filename
				image = cv2.imread(center_current)
				small_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
				gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
				gray_image = gray_image.reshape(80,160,1)
				images.append(gray_image)
				center_measurement = float(batch_sample[3])
				measurements.append(center_measurement)
				small_image = cv2.flip(small_image, 0)
				gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
				gray_image = gray_image.reshape(80,160,1)
				images.append(gray_image)
				center_measurement = -center_measurement
				measurements.append(center_measurement)
				'''
				#APPEND left images
				left_path = batch_sample[1]
				left_filename = left_path.split('/')[-1]
				left_current = data_folder + 'IMG/' + left_filename
				image = cv2.imread(left_current)
				small_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
				gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
				gray_image = gray_image.reshape(80,160,1)
				images.append(gray_image)				
				left_measurement = center_measurement + 0.10
				measurements.append(left_measurement)
				image = cv2.flip(image, 0)
				small_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
				gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
				gray_image = gray_image.reshape(80,160,1)
				images.append(gray_image)
				left_measurement = -left_measurement
				measurements.append(left_measurement)
				
				#APPEND right images
				right_path = batch_sample[2]
				right_filename = right_path.split('/')[-1]
				right_current = data_folder + 'IMG/' + right_filename
				image = cv2.imread(right_current)
				small_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
				gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
				gray_image = gray_image.reshape(80,160,1)
				images.append(gray_image)
				right_measurement = center_measurement - 0.10
				measurements.append(right_measurement)
				image = cv2.flip(image, 0)
				small_image = cv2.resize(image, (0,0), fx=0.5, fy=0.5)
				gray_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
				gray_image = gray_image.reshape(80,160,1)
				images.append(gray_image)
				right_measurement = -right_measurement
				measurements.append(right_measurement)
				'''

			X_train = np.array(images)
			#print(X_train)
			#print(y_train)
			y_train = np.array(measurements)
			yield sklearn.utils.shuffle(X_train, y_train)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)
train_gen = gen(train_samples, batch_size=8)
valid_gen = gen(validation_samples, batch_size=8)
#print(train_gen)
'''
images = []
measurements = []
for line in lines:
	source_path = line[0]
	filename = source_path.split('/')[-1]
	current = './trainingdata/IMG/' + filename
	image = mpimg.imread(current)
	image = normalize(image)
	images.append(image)
	measurement = float(line[3])
	measurements.append(measurement)
	image = cv2.flip(image, 0)
	images.append(image)
	measurement = -measurement
	measurements.append(measurement)
X_train = np.array(images)
y_train = np.array(measurements)
'''
#print(X_train.shape)
#print(y_train.shape)


#MAIN Pipeline
model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(80,160,1)))
model.add(Cropping2D(cropping=((26,10),(0,0))))
model.add(Convolution2D(16,3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(32,3,3,activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Convolution2D(64,3,3,activation="relu"))
#model.add(Convolution2D(63,2,2,activation="relu"))
#model.add(Convolution2D(63,1,1,activation="relu"))
model.add(Flatten())
#model.add(Convolution2D(63,3,3,activation="relu"))
model.add(Dense(100))
model.add(Dropout(.4))
#model.add(Convolution2D(63,3,3,activation="relu"))
model.add(Dense(50))
model.add(Dropout(.2))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_gen, samples_per_epoch = len(train_samples), validation_data=valid_gen, nb_val_samples=len(validation_samples), nb_epoch=5)

model.save('model.h5')
print("Model Saved")

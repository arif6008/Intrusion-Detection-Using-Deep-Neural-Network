# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 18:31:30 2019

@author: arif
"""
import keras
import scipy.io as sc
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.models import Sequential
from keras.layers import Convolution2D, Dense, Dropout, Flatten, MaxPooling2D

import numpy as np


np.random.seed(7)

## Data Creation
traindata= sc.loadmat('train_feature.mat')
X1 = traindata['train_feature']
feature_train = np.array(X1[:,0:39])
print(feature_train.shape)


testdata= sc.loadmat('test_feature.mat')
X2 = testdata['test_feature']
feature_test = np.array(X2[:,0:39])


classdata= sc.loadmat('train_class.mat')
X1 = classdata['train_class']
class_train = np.array(X1[:,0:1])



classdata= sc.loadmat('test_class.mat')
X2 = classdata['test_class']
class_test = np.array(X2[:,0:1])
print(feature_test.shape)
X = feature_train.reshape((123516, 1, 39, 1))
X_test=feature_test.reshape((19957, 1, 39, 1))

Y=keras.utils.to_categorical(class_train)
Y_test=keras.utils.to_categorical(class_test)
cnn = Sequential()
cnn.add(Convolution2D(64, 3, 1,
    border_mode="same",
    activation="relu",
    input_shape=(1, 39, 1)))
cnn.add(Convolution2D(64, 3, 1, border_mode="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,1),dim_ordering="th"))

cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(128, 3, 1, border_mode="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,1),dim_ordering="th"))

cnn.add(Convolution2D(256, 3, 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(256, 3, 1, border_mode="same", activation="relu"))
cnn.add(Convolution2D(256, 3, 1, border_mode="same", activation="relu"))
cnn.add(MaxPooling2D(pool_size=(2,1),dim_ordering="th"))

cnn.add(Flatten())
cnn.add(Dense(100, activation="relu"))
#cnn.add(Dropout(0.5))
cnn.add(Dense(20, activation="relu"))
#cnn.add(Dropout(0.5))
cnn.add(Dense(5, activation="softmax"))

# define optimizer and objective, compile cnn

cnn.compile(loss="categorical_crossentropy", optimizer="adam",metrics=['accuracy'])
# train

cnn.fit(X, Y, nb_epoch=5,batch_size=50,verbose=1)

score = cnn.evaluate(X_test, Y_test, batch_size=30)
(loss, accuracy)=score

print('\n Testing Accuracy')
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss, accuracy * 100))

score1 = cnn.evaluate(X, Y, batch_size=300)
(loss1, accuracy1)=score1
print('\n Training Accuracy')
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss1, accuracy1 * 100))

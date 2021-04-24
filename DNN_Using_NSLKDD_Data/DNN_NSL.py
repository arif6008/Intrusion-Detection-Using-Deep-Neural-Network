# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 11:21:30 2019

@author: arif
"""
import scipy.io as sc
import keras
import numpy as np

from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD

#import numpy
# fix random seed for reproducibility
np.random.seed(7)

## Data Creation
traindata= sc.loadmat('train_feature.mat')
X1 = traindata['train_feature']
feature_train = np.matrix(X1[:,0:39])

testdata= sc.loadmat('test_feature.mat')
X2 = testdata['test_feature']
feature_test = np.matrix(X2[:,0:39])

classdata= sc.loadmat('train_class.mat')
X1 = classdata['train_class']
class_train = np.array(X1[:,0:1])

classdata= sc.loadmat('test_class.mat')
X2 = classdata['test_class']
class_test = np.array(X2[:,0:1])

#X = feature_train.reshape((4817124, 1, 38, 1))
Y=keras.utils.to_categorical(class_train)
#y_train = keras.utils.to_categorical(np.transpose(class_train))
y_train = keras.utils.to_categorical(class_train)


#y_train = keras.utils.to_categorical(np.transpose(class_train))

model = Sequential()
# Dense(64) is a fully-connected layer with 64 hidden units.
# in the first layer, you must specify the expected input data shape:
# here, 20-dimensional vectors.
model.add(Dense(120, activation='relu', input_dim=feature_train.shape[1]))
model.add(Dropout(0.5))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(60, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(y_train.shape[1], activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(feature_train, y_train,
          epochs=50,
          batch_size=300,verbose=1, validation_split=0.1)
#y_test=keras.utils.to_categorical(np.transpose(class_test))
y_test=keras.utils.to_categorical(class_test)
score = model.evaluate(feature_test, y_test, batch_size=200)
(loss, accuracy)=score

print('\n Testing Accuracy')
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss,
	accuracy * 100))

score1 = model.evaluate(feature_train, Y, batch_size=200)
(loss1, accuracy1)=score1
print('\n Training Accuracy')
print("[INFO] loss={:.4f}, accuracy: {:.4f}%".format(loss1, accuracy1 * 100))



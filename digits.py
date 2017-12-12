import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import  MaxPooling2D, AveragePooling2D, Conv2D
from keras.utils import np_utils
import numpy as np
import math
from math import sqrt

from os import listdir
from os.path import isfile, join

import pylab
from scipy import misc

from sklearn.model_selection import train_test_split

np.random.seed(123)



def get_digits_images():
  path_train="train/"
  path_test="test/"
  train= pd.read_csv("train.csv",sep=",")
  test= pd.read_csv("test.csv",sep=",")
  X_train=[]
  X_test=[]
  for row in train.itertuples():
    #print(row[1])
    im=misc.imread(path_train+row[1],flatten =True) # niveaux de gris
    X_train.append(im) 
  for row in test.itertuples():
    im=misc.imread(path_test+row[1],flatten =True)# niveaux de gris
    X_test.append(im)
  return np.asarray(X_train), train.iloc[:,1:2],np.asarray(X_test)  #renvoie xtrain, ytrain, xtest
  
#charger les images
X_train,y_train,X_test=get_digits_images()

# partitionner jeux train/test
xtrain, xtest, ytrain, ytest = train_test_split(  X_train, y_train, test_size=0.3, random_state=123)




######## perceptrons
vector_size=xtrain.shape[1] * xtrain.shape[2]
xtrain = xtrain.reshape(xtrain.shape[0], vector_size)  # transformer l'image 28*28 en vecteur 1D
xtest = xtest.reshape(xtest.shape[0], vector_size)# transformer l'image 28*28 en vecteur 1D

# normaliser
xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')
xtrain=xtrain/255.0
xtest=xtest/255.0

# one hot encoding
ytrain=np_utils.to_categorical(ytrain)
ytest=np_utils.to_categorical(ytest)

#importer pylab
pylab.show() ### display sous windows


def perceptrons(vector_size):
  model = Sequential()
  model.add(Dense(vector_size, activation='relu', input_dim=(vector_size)))
  model.add(Dense(10, activation='softmax')) # 10 sorties/classes
  model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
  return model
  

model_perc = perceptrons(vector_size)  
model_perc.fit(xtrain,ytrain,batch_size=200, nb_epoch=10, verbose=1)
# loss: 0.0108 - acc: 0.9985

score = model_perc.evaluate(xtest, ytest, verbose=1)
#[0.084462631291830209, 0.9754421768707483]
#2.45578231293 % error

def perceptrons_deeper(vector_size):
  model = Sequential()
  model.add(Dense(vector_size, activation='relu', input_dim=(vector_size)))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='softmax')) # 10 sorties/classes
  model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
  return model
  
  
model_perc = perceptrons_deeper(vector_size)  
model_perc.fit(xtrain,ytrain,batch_size=200, nb_epoch=10, verbose=1)
# loss: 0.0085 - acc: 0.9980


score = model_perc.evaluate(xtest, ytest, verbose=1)
#[0.091340311163035384, 0.97646258503401362]
#2.3537414966% error

  


  
  

############### 2d convolution
#https://elitedatascience.com/keras-tutorial-deep-learning-in-python

# partitionner jeux train/test
xtrain, xtest, ytrain, ytest = train_test_split(  X_train, y_train, test_size=0.3, random_state=123)

####reshape pour ajouter un canal de profondeur aux images (requis par keras)
xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], xtrain.shape[1],1) ### attention ici ordre des parametres pour tensorflow
xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], xtest.shape[1],1)

# normaliser
xtrain=xtrain.astype('float32')
xtest=xtest.astype('float32')
xtrain=xtrain/255.0
xtest=xtest/255.0

# one hot encoding
ytrain=np_utils.to_categorical(ytrain)
ytest=np_utils.to_categorical(ytest)

def conv_neuronet():
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(28,28,1))) # attention input_shape different tensorflow ou theano
  model.add(Conv2D(32, (3, 3), activation='relu'))
  model.add(MaxPooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25))  
  model.add(Flatten()) #passage NN perceptron
  model.add(Dense(128, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(10, activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
  return model


model_conv=conv_neuronet()
model_conv.fit(xtrain,ytrain,batch_size=200, nb_epoch=10, verbose=1)
#loss: 0.0360 - acc: 0.9885
score = model_conv.evaluate(xtest, ytest, verbose=1)
#[0.043251634816287507, 0.98734693877551016]
#1.26530612245 % error


#Efficiency Optimization of Trainable Feature Extractors for a Consumer Platform
def conv_neuronet2():
  model = Sequential()
  model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28,28,1))) # attention input_shape different tensorflow ou theano
  #print(model.layers[-1].output_shape)
  model.add(AveragePooling2D(pool_size=(2,2)))
  #print(model.layers[-1].output_shape)
  model.add(Conv2D(16, (3, 3), activation='relu'))
  #print(model.layers[-1].output_shape)
  model.add(AveragePooling2D(pool_size=(2,2)))
  model.add(Dropout(0.25)) # random 0 to prevent overfitting 
  model.add(Flatten()) #passage NN perceptron
  model.add(Dense(128, activation='relu'))
  model.add(Dense(50, activation='relu'))
  model.add(Dense(10, activation='softmax'))
  model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
  return model
  
model_conv2=conv_neuronet2()
model_conv2.fit(xtrain,ytrain,batch_size=200, nb_epoch=10, verbose=1)
#loss: 0.0505 - acc: 0.9841
score = model_conv2.evaluate(xtest, ytest, verbose=1)
#[0.044315310374383819, 0.98727891156462588]
#1.27210884354 % error


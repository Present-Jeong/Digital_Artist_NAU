# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

# LOAD LIBRARIES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, BatchNormalization,InputLayer,Conv2DTranspose
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import LearningRateScheduler

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.datasets import mnist
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras import regularizers


# LOAD THE DATA
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")
train.head()

Y_train = train["label"]
X_train = train.drop(labels = ["label"],axis = 1)
X_train = X_train / 255.0
X_test = test / 255.0
X_train = X_train.values.reshape(-1,28,28,1)
X_test = X_test.values.reshape(-1,28,28,1)
X_train.shape,X_test.shape

def train_val_split(x_train, y_train):
    rnd = np.random.RandomState(seed=42)
    perm = rnd.permutation(len(x_train))
    train_idx = perm[:int(0.8 * len(x_train))]
    val_idx = perm[int(0.8 * len(x_train)):]
    return x_train[train_idx], y_train[train_idx], x_train[val_idx], y_train[val_idx]

n = 20000  # for 2 random indices
index = np.random.choice(X_train.shape[0], n, replace=False) 
X_train=X_train[index]
x_train, y_train, x_val, y_val = train_val_split(X_train, X_train)

max_value = float(x_train.max())
x_train = x_train.astype('float32') / max_value
x_val = x_val.astype('float32') / max_value
print(x_train.shape, x_val.shape)

input=Input(shape=(x_train.shape[1:]))
encoded=Conv2D(16, (3, 3), activation='relu', padding='same')(input)
encoded=MaxPooling2D((2, 2), padding='same')(encoded)
encoded=Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
encoded=MaxPooling2D((2, 2), padding='same')(encoded)
encoded=Conv2D(8, (3, 3), strides=(2,2), activation='relu', padding='same')(encoded)
encoded=Flatten()(encoded)

decoded=Reshape((4, 4, 8))(encoded)
decoded=Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
decoded=UpSampling2D((2, 2))(decoded)
decoded=Conv2D(8, (3, 3), activation='relu', padding='same')(decoded)
decoded=UpSampling2D((2, 2))(decoded)
decoded=Conv2D(16, (3, 3), activation='relu')(decoded)
decoded=UpSampling2D((2, 2))(decoded)
decoded=Conv2D(1, (3, 3), activation='sigmoid', padding='same')(decoded)
        
autoencoder=Model(input,decoded)


autoencoder.summary()

print(x_train.shape)

encoder = Model(inputs=autoencoder.input, outputs=autoencoder.layers[6].output)
encoder.summary()

encoded_input = Input(shape=(128,))

deco = autoencoder.layers[-8](encoded_input)
deco = autoencoder.layers[-7](deco)
deco = autoencoder.layers[-6](deco)
deco = autoencoder.layers[-5](deco)
deco = autoencoder.layers[-4](deco)
deco = autoencoder.layers[-3](deco)
deco = autoencoder.layers[-2](deco)
deco = autoencoder.layers[-1](deco)
# create the decoder model
decoder = Model(encoded_input, deco)
decoder.summary()

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

print(len(autoencoder.layers),len(encoder.layers),len(decoder.layers))


from keras.preprocessing import image

gen = image.ImageDataGenerator()
batches = gen.flow(x_train, x_train, batch_size=64)
val_batches=gen.flow(x_val, x_val, batch_size=64)

history=autoencoder.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=4, 
                    validation_data=val_batches, validation_steps=val_batches.n)

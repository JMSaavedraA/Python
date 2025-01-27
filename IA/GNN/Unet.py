import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"]="1";  
import keras
from keras.layers import Input, Dense, Activation, Conv2D
from keras.layers import MaxPooling2D, Dropout, UpSampling2D
from keras.layers import BatchNormalization, Reshape
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.models import Sequential
from keras.utils import plot_model

import numpy as np
import matplotlib.pyplot as plt

def resumen(model=None):
    '''
    Descipción del modelo en foam compacta (la prefiero a `summary` de keras)
    '''
    header = '{:4} {:16} {:24} {:24} {:10}'.format('#', 'Layer Name','Layer Input Shape','Layer Output Shape','Parameters'
    )
    print('='*(len(header)))
    print(header)
    print('='*(len(header)))
    count=0
    count_trainable=0
    for i, layer in enumerate(model.layers):
        count_trainable += layer.count_params() if layer.trainable else 0
        input_shape = '{}'.format(layer.input_shape)
        output_shape = '{}'.format(layer.output_shape)
        str = '{:<4d} {:16} {:24} {:24} {:10}'.format(i,layer.name, input_shape, output_shape, layer.count_params())
        print(str)
        count += layer.count_params()
    print('_'*(len(header)))
    print('Total Parameters : ', count)
    print('Total Trainable Parameters : ', count_trainable)
    print('Total No-Trainable Parameters : ', count-count_trainable)


from keras.datasets import mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

from keras.datasets import mnist

# lectura de los datos
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Dimensiones del conjunto de entrenamiento: ', train_images.shape)
print('Dimensiones del conjunto de evaluación: ',    test_images.shape)

num_data, nrows, ncols = train_images.shape

X_train = np.copy(train_images).astype('float64')/255.
Y_train = np.copy(train_images).astype('float64')/255.
X_test  = np.copy(test_images).astype('float64')/255.
Y_test  = np.copy(test_images).astype('float64')/255.

sigma = 0.4
X_train += np.random.normal(loc=0, scale=sigma, size=train_images.shape)
X_test  += np.random.normal(loc=0, scale=sigma, size=test_images.shape)
Y_train = Y_train>0.5
Y_test  = Y_test>0.5

num_test_images, num_rows, num_cols = X_test.shape
X_test.shape
(10000, 28, 28)
imgs=10
plt.figure(figsize=(14,4))
for i in range(imgs):
    plt.subplot(3,imgs,i+1)
    idx = list(train_labels).index(i)
    plt.imshow(train_images[idx], 'gray')
    plt.title(train_labels[idx])
    plt.axis('off')
    
    plt.subplot(3,imgs,i+1+imgs)
    idx = list(train_labels).index(i)
    plt.imshow(X_train[idx], 'gray')
    #plt.title(train_labels[idx])
    plt.axis('off')

    plt.subplot(3,imgs,i+1+2*imgs)
    idx = list(train_labels).index(i)
    plt.imshow(Y_train[idx], 'gray')
    #plt.title(train_labels[idx])
    plt.axis('off')

X_train = np.expand_dims(X_train, axis=3)
Y_train = np.expand_dims(Y_train, axis=3)
X_test  = np.expand_dims(X_test,  axis=3)
Y_test  = np.expand_dims(Y_test,  axis=3)

print('Dimensiones de entradas (X) para entrenamiento  (imagenes x rows x cols) =', X_train.shape)
print('Dimensiones de saida (Y) para entrenamiento     (imagenes x rows x cols) =', Y_train.shape)
print('Dimensiones de entradas (X) para evaluación     (imagenes x rows x cols) =', X_test.shape)
print('Dimensiones de saida (Y) para evaluación        (imagenes x rows x cols) =', Y_test.shape)
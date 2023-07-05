import cv2 as cv
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import matplotlib.pyplot as plt
import numpy as np

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()


x_train = x_train/255.0
x_test = x_test/255.0
x_train = np.expand_dims(x_train, axis = -1)
x_test = np.expand_dims(x_test, axis = -1)


inputs = KL.Input(shape = (28,28, 1)) 
conv_layer = KL.Conv2D(32,(3,3), padding = "same", activation = tf.nn.relu)(inputs)


max_pooling = KL.MaxPool2D((2,2), (2,2))(conv_layer)
conv_layer2 = KL.Conv2D(64,(3,3), padding = "same", activation = tf.nn.relu)(max_pooling)

max_pooling2 = KL.MaxPool2D((2,2), (2,2))(conv_layer2)
conv_layer3 = KL.Conv2D(128,(3,3), padding = "same", activation = tf.nn.relu)(max_pooling2)


flat = KL.Flatten()(max_pooling2)


outputs = KL.Dense(10,activation = tf.nn.softmax)(flat)


model = KM.Model(inputs, outputs)

model.summary()
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])
model.fit(x_train,y_train, epochs = 10)
test_loss, test_acc = model.evaluate(x_test,y_test)


model.save('MNIST.h5')

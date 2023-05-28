import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM

import numpy as np

#load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range in order for image to fit in a smaller frame and allow the NN to have consistent inputs
#shape of x_train = (60000, 28, 28)
#            60000 images, 28x28 for size of image
x_train = x_train/255.0
x_test = x_test/255.0

#add another dimension for "channel" to help NN determine intensity of greyscale vs white space
x_train = np.expand_dims(x_train, axis = -1)
x_test = np.expand_dims(x_test, axis = -1)

print(x_test.shape)

#build model
inputs = KL.Input(shape = (28,28,1)) #shape is from x_test and x_train
#32 input filters each exiting with a weight and feature map, no padding so smaller output(risk of loosing information at edges),
# 3x3 filter gets applies to 28x28 image to get feature map and results in 3x3 feature map(slding 3x3 onto 28x28 to get clearer image)
conv_layer = KL.Conv2D(32,(3,3), padding = "valid", activation = tf.nn.relu)(inputs)
#divide input into regions and returns max value 
max_pooling = KL.MaxPool2D((2,2), (2,2))(conv_layer)
#
flat = KL.Flatten()(max_pooling)
#
outputs = KL.Dense(10,activation = tf.nn.softmax)(flat)
#
model = KM.Model(inputs, outputs)
model.summary()


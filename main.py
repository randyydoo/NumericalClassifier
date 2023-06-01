import cv2 as cv
import tensorflow as tf
import tensorflow.keras.layers as KL
import tensorflow.keras.models as KM
import matplotlib.pyplot as plt
import numpy as np
'''
#load data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_test, y_test, x_train, y_train)

# Rescale the images from [0,255] to the [0.0,1.0] range in order for image to fit in a smaller frame and 0-1 range will represent darkeness 
#shape of x_train = (60000, 28, 28)
#            60000 images, 28x28 for size of image

x_train = x_train/255.0
x_test = x_test/255.0
#add another dimension for "channel" to help NN determine intensity of greyscale vs white space
x_train = np.expand_dims(x_train, axis = -1)
x_test = np.expand_dims(x_test, axis = -1)


#build model
inputs = KL.Input(shape = (28,28, 1)) #shape is from x_test and x_train
#32 3x3 filters gets applied to 28x28 image with stride 1. 3x3 moves moves over 1 to right and gets largest value 
#we get 32 26x26 matrix from [(input_size - filter_size + 2 * padding) / stride] + 1 = [(28 - 3 + 2 * 0) / 1] + 1
conv_layer = KL.Conv2D(32,(3,3), padding = "same", activation = tf.nn.relu)(inputs)

#divide 3x3 matrix and slide 1x1 matrix over and return a max values onto 2x2 matrix  
max_pooling = KL.MaxPool2D((2,2), (2,2))(conv_layer)
conv_layer2 = KL.Conv2D(64,(3,3), padding = "same", activation = tf.nn.relu)(max_pooling)

max_pooling2 = KL.MaxPool2D((2,2), (2,2))(conv_layer2)
conv_layer3 = KL.Conv2D(128,(3,3), padding = "same", activation = tf.nn.relu)(max_pooling2)

#turn matrix into array and each position has value refereing to intensity of pixel
flat = KL.Flatten()(max_pooling2)

#connect layers after the 32 inital nodes to get output
outputs = KL.Dense(10,activation = tf.nn.softmax)(flat)

#create model
model = KM.Model(inputs, outputs)

# model.summary()
# adam optimzer is most common
model.compile(optimizer = "adam", loss = "sparse_categorical_crossentropy", metrics = ['accuracy'])
model.fit(x_train,y_train, epochs = 3)
test_loss, test_acc = model.evaluate(x_test,y_test)
model.save('MNIST.model')
print(f"Test Loss: {test_loss} ---- Test Accuracy {test_acc}")
'''
#read images and output predictions
for num in range(1,6):
    img = cv.imread(f'{num}.png')[:,:,0]
    img = np.array([img])
    plt.imshow(img[0])
    plt.show()

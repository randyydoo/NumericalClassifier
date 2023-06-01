import cv2 as cv
import numpy as np
from tensorflow.keras.models import load_model

model = load_model('MNIST.model')
for num in range(1,6):
    image = cv.imread(f'{num}.png')[:,:,0]
    image = np.invert(np.array([image]))
    prediction = model.predict(image)
    print(f'The number is most likely: {np.argmax(prediction)}')

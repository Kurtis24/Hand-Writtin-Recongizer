import os
import cv2 as cv
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model


# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(10, activation='softmax'))


# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# model.fit(x_train, y_train, epochs=3)

# model.save('Handwrittin_Digits.keras')

model = load_model('Handwrittin_Digits.keras')


image_number = 1
while os.path.isfile(f"Photo/Untitled_Artwork {image_number}.png"):
    try:
        img = cv.imread(f"Photo/Untitled_Artwork {image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("Error")
    finally: 
        image_number += 1 
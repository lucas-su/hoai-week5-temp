from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from keras import models, layers, losses
import cv2

mode = 'patch' # mnist or patch

if mode == 'mnist':
    X_mnist, y_mnist = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False, parser='auto')
    
    y_mnist = np.array(y_mnist, dtype=int)
    dim_row = 28
    dim_col = 28

    # unflatten mnist images
    X_mnist = np.array([np.reshape(xf,(dim_row, dim_col)) for xf in X_mnist])
    iters = 2
else:
    dim_row = 27
    dim_col = 19
    iters = 500 

with open(f"{os.path.dirname(os.path.realpath(__file__))}/all_drawings.json", 'r') as file:
    data = json.load(file)
X_tsp = np.array([d[0] for d in data])
y_tsp = np.array([int(d[1]) for d in data])

def plot_digits(X, y):
    plt.figure(figsize=(20,6))
    for i in range(10):
        if np.where(y==f"{i}")[0].size > 0:
            index = np.where(y==f"{i}")[0][0]
            digit_sub = plt.subplot(2, 5, i + 1)
            digit_sub.imshow(np.reshape(X[index], (dim_row,dim_col)), cmap="gray")
            digit_sub.set_xlabel(f"Digit {y[index]}")
    plt.show()

if mode == 'mnist':
    X_train, X_test, y_train, y_test = train_test_split(X_mnist, y_mnist, test_size=0.25)
else:
    X_train, X_test, y_train, y_test = train_test_split(X_tsp, y_tsp, test_size=0.25)


model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(dim_row, dim_col,1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

hist = model.fit(X_train,y_train, epochs=iters, validation_split=0.2)

plt.plot(hist.history['accuracy'], label='accuracy')
plt.plot(hist.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

test_loss, test_acc = model.evaluate(X_test,  y_test, verbose=2)
print(f"Test loss: {test_loss} Test accuracy: {test_acc}")



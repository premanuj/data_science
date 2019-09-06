from keras.datasets import mnist
from keras import models, layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt
"""
This model is effected from OVERFITTING problem
It is because as we can training accuracing is greater than test accuracy

Note: Training accuracy can be see in last epoch of training
"""

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print("===========Train Image ===============")
print("Train image shape: ", train_images.shape)
print("Length Train label: ", len(train_labels))

print("===========Test Image ===============")
print("Test image shape: ", test_images.shape)
print("Length Test label: ", len(test_labels))

#Network Architecture
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

#Compilation step
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#Preparing the image data
original_train_images = train_images
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype('float32') / 255

#Preparing the labels
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

#Fit model 
network.fit(train_images, train_labels, epochs=5, batch_size=128)

test_loss, test_acc=network.evaluate(test_images, test_labels)

print("Test loss: ", test_loss)
print("Test accurate: ", test_acc)

# digit = original_train_images[4]

# print("SHAPE: ", digit.shape)

# plt.imshow(digit, cmap="gray")
# plt.show()
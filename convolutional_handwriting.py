import keras 
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from keras.optimizers import Adadelta
from keras.losses import categorical_crossentropy

pixel_width = 28
pixel_height = 28 
no_of_classes = 10
epochs=10
batch_size = 32 
optimizer = Adadelta(lr=1.5)
# load data
(features_train, labels_train) , (features_test, labels_test) = mnist.load_data()

features_train = features_train.reshape(features_train.shape[0], pixel_width, pixel_height, 1)
features_test = features_test.reshape(features_test.shape[0], pixel_width, pixel_height, 1)

input_shape = (pixel_width, pixel_height, 1)

# FLOAT 32 VALUES
features_train = features_train.astype('float32')
features_test = features_test.astype('float32')
# print(features_train[0])
# CONVERTING TO %
features_train /= 255 
features_test /= 255 

# print(labels_train[5])

# flattened to binary matrix
labels_train = keras.utils.to_categorical(labels_train, no_of_classes)
labels_test = keras.utils.to_categorical(labels_test, no_of_classes)
# print(labels_train[5])

models = Sequential()
models.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
# print("post conn2d")
# print(models.output_shape)
models.add(MaxPooling2D(pool_size=(2,2), ))
# print("post MaxPooling")
# print(models.output_shape)
models.add(Dropout(rate=0.25))
# print("post dropout")
# print(models.output_shape)
# print("post flatten")
models.add(Flatten())
# print(models.output_shape)
models.add(Dense(no_of_classes, activation='softmax'))
# print("Post Dense", models.output_shape)
models.compile(loss= categorical_crossentropy, optimizer= optimizer, metrics=['accuracy'])

models.fit(features_train, labels_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(features_test, labels_test))
score = models.evaluate(features_test, labels_test, verbose=0)
print(score[0])

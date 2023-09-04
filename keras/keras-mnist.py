import numpy 
numpy.random.seed(1337)   # for experiment reproducibility 

import tensorflow
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense 
from tensorflow.keras.optimizers.legacy import SGD 

num_classes = 10 # we are going to train for digits 0 to 9. Therefore, we need 10 classes

batch_size = 128 # 128 data points will be sent to the network at a time for batch processing
epochs = 25 # the number of iterations (the number of times the data is fed to the machine for additional, incremental learning). You may want to use 20+ in production.

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
# (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)
# x_train has 60000 images at a 28x28 resolution
# y_train has the 60000 corresponding labels (targets)

# we want to flatten the 28 x 28 dimension to a single 784 (28 * 28) horizontal row (i.e. flat vector)
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

# we also want the values of the flat vector in deciman (float) not integer
x_train = x_train.astype(float)
x_test = x_test.astype(float)

print(x_train[0]) # notice all values are currently between 0 - 255

# We want to normalize the data so that all values are between 0-1 to make learning easier
x_train /= 255 # divides all the individual data values by 255 
x_test /= 255 # same

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print(x_train.shape)
print(y_train.shape)

print(x_train[0]) # notice all the data values are now between 0 to 1

# after normalizing the data (x), we now need to transform the labels (y) in class matrixes

# convert class vectors to binary class matrices (i.e. one-hot vectors)
# e.g. a label matching class 2 is: (0, 0, 1, 0, 0, 0, 0, 0, 0, 0) 
# since we have ten classes we need 9 zeros and one 1 per row to signify true/false for each possible class
print('y before transformation', y_train[0])
y_train = tensorflow.keras.utils.to_categorical(y_train, num_classes)
y_test = tensorflow.keras.utils.to_categorical(y_test, num_classes)
print('y after transformation', y_train[0]) # this is called a one-hot vector

# create a Sequential model with three hidden layers with the final layer returning the softmax value (the maximum value)
# input -> Dense Layer 1 -> Dense Layer 2 -> Dense Layer 3 -> Ouput (softmax value)
model = Sequential()
# note: we must use the comma (784,) so the input data is treated as a tuple
model.add( Dense(512, activation='sigmoid', input_shape=(784,)  ) ) # first layer takes input in the shape of a flat vector containing 784 values and its comprised of 512 nodes
model.add( Dense(512, activation='sigmoid'  ) ) # the next layer is also 512 nodes
model.add( Dense(num_classes, activation='softmax')) # the final layer outputs the soft max value based on the number of classes (num_classes)

# view the model
print(model.summary())

# Compile the model 
model.compile(loss='categorical_crossentropy', optimizer=SGD(), 
              metrics=['accuracy']) # use SGD to opimize the model and use accuracy that measurement to optimize against for each iteration

# let's perform the learning 
history = model.fit( x_train, y_train, 
           batch_size=batch_size,
           epochs=epochs, 
           verbose=1, 
           validation_data=(x_test, y_test))

# Let's evaluate the model 
score = model.evaluate(x_test, y_test)

print('The score', score[1])

from sklearn import datasets

digits = datasets.load_digits()

# viewing all data
print ('All data', digits.data, '\n\n')

# view one sample data element to see what an individual data point looks like
print ('Data point', digits.data[0], '\n\n')

# get the label/target of the data point (i.e. what the data point represents)
print ('Data point target', digits.target[0], '\n\n')

# get the shape of the targets (e.g. the number of targets in the data set)
print ('Data targets shape', digits.target.shape, '\n\n')

# get the shape of the data point
print ('Data point shape', digits.data[0].shape, '\n\n')

# overall data shape
print ('Data shape', digits.data.shape, '\n\n')

# last ten targets
print ('Last ten targets', digits.data[-10:], '\n\n')

# train a classifier
from sklearn import svm # import the svm classifier
clf = svm.SVC(gamma=0.001, C=100.) # create a classifier
clf.fit(digits.data[:-1], digits.target[:-1]) # fit = learn; fit the model to the data points and corresponding data targets

predicted_value = clf.predict(digits.data[-1:])

print('Predicted value vs actual value',  "correct" if predicted_value == digits.target[-1:] else 'incorrect', predicted_value, digits.target[-1:],)

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.figure(figsize=(0.4,0.4))
plt.imshow(digits.data[-1:], interpolation='nearest', cmap=plt.cm.binary)
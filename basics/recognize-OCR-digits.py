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


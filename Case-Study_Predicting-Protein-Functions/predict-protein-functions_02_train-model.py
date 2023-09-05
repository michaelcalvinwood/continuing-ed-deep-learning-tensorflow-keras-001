import numpy as np
import json
import os 

from tensorflow.keras.preprocessing import sequence

sequences_file = os.path.join('.', 'data', 'protein-seqs-2023-09-04-193448.txt')
functions_file = os.path.join('.', 'data', 'protein-functions-2023-09-04-193448.txt')

with open(functions_file) as fn_file:
    has_function = json.load(fn_file)

print(has_function)

max_sequence_size = 500   # any sequence longer than this, we ignore (just for now) 
max_num = 0

# create the data (X) and the labels (y)
X = []           # sequences in the same order corresponding to elements of p 
y = []           # output class: 1 if protein has the function, 0 if not 

# for seeing how many examples we've found for each class 
pos_examples = 0
neg_examples = 0   

with open(sequences_file) as f:
    for line in f:
        ln = line.split(',')
        protein_id = ln[0].strip()
        seq = ln[1].strip()

        # we're doing this to reduce input size
        if len(seq) >= max_sequence_size:
            continue
        
        print(line)
        
        X.append(seq)
        
        if protein_id in has_function: 
            y.append(1) 
            pos_examples += 1 
        else: 
            y.append(0) 
            neg_examples += 1 

print("Positive Examples: %d" % pos_examples)
print("Negative Examples: %d" % neg_examples)  # Total is different because we ignored longer sequences 

# must convert the characters in the protein sequence into numbers
def sequence_to_indices(sequence):
    """Convert amino acid letters to indices. 
       _ means no amino acid (used for padding to accommodate for variable length)"""
    
    try:
        acid_letters = ['_', 'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M',
                'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']

        indices = [acid_letters.index(c) for c in list(sequence)]
        return indices
    except Exception:
        print(sequence)
        raise Exception

# print('sequence_to_indices test', sequence_to_indices('AD')) # test the function

X_all = [] 
for i in range(len(X)): 
    x = sequence_to_indices(X[i])
    X_all.append(x) 

# print('X_all', X_all)

X_all = sequence.pad_sequences(X_all, maxlen=max_sequence_size)
X_all = np.array(X_all) # convert list to np.array
y_all = np.array(y) # same

# check that the data makes sense
print('first label', y[0])
print('first data point', X_all[0])
print('length of first data point', len(X_all[0]))

print('X_all.shape', X_all.shape)  # extremely important that you view this! 
print('y_all.shape', y_all.shape)  # make sure you are comfortable with shapes! 

# split the data
n = X_all.shape[0]  # number of data points 

randomize = np.arange(n) # randomize to shuffle first. Pro Tip: important to use randomized shuffling in Deep Learning in lieu of K-Fold Cross Validation
np.random.shuffle(randomize)

print ('randomize', randomize)

X_all = X_all[randomize]
y_all = y_all[randomize]

test_split = round(n * 2 / 3)
X_train = X_all[:test_split]   # start to (just before) test_split 
y_train = y_all[:test_split]   
X_test  = X_all[test_split:]   # test_split to end 
y_test  = y_all[test_split:]

# Print shapes again 
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)

# X_train.shape (38298, 500) -- a 2 dimensional array
# each of the 500 values is between 0 and 22
# we will be converting each of these integer values to its one-hot representation: an array of 22 0's and a single 1 (e.g. 5 becomes [0, 0, 0, 0, 0, 1, 0, 0...] )
# after the one-hot representation this results in a 3 dimensional array with the following shape (38298, 500, 23)

# Create and Apply the Model
from tensorflow.keras.layers import Embedding, Input, Dropout, Flatten, Dense, Activation
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import SGD

num_amino_acids = 23 
embedding_dims = 10 
nb_epoch = 2
batch_size = 2

model = Sequential() 

model.add(Embedding(num_amino_acids, embedding_dims, input_length=max_sequence_size  ))
model.add(Flatten())
model.add(Dense(25, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=SGD(),
metrics=['accuracy'])

print('model summary', model.summary())

hist = model.fit(X_train, y_train,
                  batch_size = batch_size,
                  epochs = nb_epoch, 
                  validation_data = (X_test, y_test),
                  verbose=1)   

print('history', hist.history)
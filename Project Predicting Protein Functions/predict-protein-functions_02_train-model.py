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



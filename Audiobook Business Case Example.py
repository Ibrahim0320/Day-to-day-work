# Audiobook business case example

# Given cutomer purchasing and engagement data, build a machine learning algorithm to
# see whether a customer will buy again

import numpy as np
from sklearn import preprocessing
import tensorflow as tf

'''
raw_data= np.loadtxt('Relevant CSV Files/Audiobooks_data.csv', delimiter= ',')

unscaled_inputs_all= raw_data[:, 1:-1]
targets_all= raw_data[:, -1]


# Balance the dataset, Goal: as many 1s as 0s in targets 
num_one_targets= int(np.sum(targets_all))
zero_targets_counter= 0
indices_to_remove= []

for i in range(targets_all.shape[0]):
    if targets_all[0] == 0:
        zero_targets_counter += 1
        if zero_targets_counter > num_one_targets:
            indices_to_remove.append(i)

unscaled_targets_equal_priors= np.delete(unscaled_inputs_all, indices_to_remove, axis=0)
target_equal_priors= np.delete(targets_all, indices_to_remove, axis= 0)

# standardise the inputs 
scaled_inputs= preprocessing.scale(unscaled_targets_equal_priors)

# shuffle the data

shuffled_indices= np.arange(scaled_inputs.shape[0])
np.random.shuffle(shuffled_indices)

shuffled_inputs= scaled_inputs[shuffled_indices]
shuffled_targets= target_equal_priors[shuffled_indices]

# Split the dataset into train, validation, and test

samples_count= shuffled_inputs.shape[0]

train_samples_count= int(0.8*samples_count)
validation_samples_count= int(0.1*samples_count)
test_samples_count= samples_count - train_samples_count - validation_samples_count

# Train inputs and targets
train_inputs= shuffled_inputs[:train_samples_count]
train_targets= shuffled_targets[:train_samples_count]

# Validation inputs and targets
validation_inputs= shuffled_inputs[train_samples_count: train_samples_count+validation_samples_count]
validation_targets= shuffled_targets[train_samples_count: train_samples_count+validation_samples_count]

# Test inputs and targets
test_inputs= shuffled_inputs[train_samples_count+validation_samples_count:]
test_targets= shuffled_targets[train_samples_count+validation_samples_count:]

print(np.sum(train_targets), train_samples_count, np.sum(train_targets) / train_samples_count)
print(np.sum(validation_targets), validation_samples_count, np.sum(validation_targets) / validation_samples_count)
print(np.sum(test_targets), test_samples_count, np.sum(test_targets) / test_samples_count)

#np.savez('Audiobook_data_train', inputs= train_inputs, targets= train_targets)
#np.savez('Audiobook_data_validation', inputs= validation_inputs, targets= validation_targets)
#np.savez('Audiobook_data_test', inputs= test_inputs, targets= test_targets)

'''

npz= np.load('Audiobook_data_train.npz')

train_inputs= npz['inputs'].astype(float)
train_targets= npz['targets'].astype(int)

npz= np.load('Audiobook_data_validation.npz')
validation_inputs, validation_targets= npz['inputs'].astype(float), npz['targets'].astype(int)

npz= np.load('Audiobook_data_test.npz')
test_inputs, test_targets= npz['inputs'].astype(float), npz['targets'].astype(int)

# Modelling

# We can in fact use the MNIST model, but with some alterations

input_size= 10
output_size= 2
hidden_layer_size= 50

model= tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, activation= 'relu'),
    tf.keras.layers.Dense(hidden_layer_size, activation= 'relu'),
    tf.keras.layers.Dense(output_size, activation= 'softmax')
])

model.compile(optimizer= 'adam', loss='sparse_categorical_crossentropy', metrics= ['accuracy'])

batch_size= 100
max_epochs= 100

early_stopping= tf.keras.callbacks.EarlyStopping(patience=2)


model.fit(train_inputs,
          train_targets,
          batch_size= batch_size, 
          epochs= max_epochs, 
          callbacks= [early_stopping],
          validation_data= (validation_inputs, validation_targets), 
            verbose=2)


# Testing the model

test_loss, test_accuracy= model.evaluate(test_inputs, test_targets)

print('Test loss: {0:.2f}, Test accuracy: {1:.2f}'.format(test_loss, test_accuracy*100.))
import os
import shutil

# Check that input data exists, clean output data and ensure folder exists
input_directory = 'input_data'
output_directory = 'output_data'

if not os.path.exists(input_directory):
    print('No data detected in input_data folder')
    exit()

if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
    os.makedirs(output_directory)
else:
    os.makedirs(output_directory)

# Run Script for ps1 (Perceptron Implementation)
import split_training_data
import create_feature_vector

# Part 1: Split Training data into training and validation set
split_training_data.run()

# Part 2: Transform each email in the training set into a feature vector
feature_vector_list = create_feature_vector.run()

print('Script complete')

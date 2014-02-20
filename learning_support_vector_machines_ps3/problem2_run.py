import pylab
import math
import numpy as np
from operator import itemgetter

# Run Script for ps3 (Learning Support Vector Machines)
import split_training_data
import create_feature_vectors
import pegasos_svm

def run():
    # Part 2
    print('\n=========================================================================================')
    print('Problem 2 Preparing data:')
    # Transform each email in the training set into a feature vector
    (feature_vector_list_training,
     integer_list_training) = create_feature_vectors.run_mnist('./input_data/mnist_train.txt')
    


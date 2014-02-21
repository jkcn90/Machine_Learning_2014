import pylab
import math
import numpy as np
from operator import itemgetter

# Run Script for ps3 (Learning Support Vector Machines)
import create_feature_vectors
import multi_class_prediction

def run():
    # Part 2
    print('\n=====================================================================================')
    print('Problem 2 Preparing data:')
    # Transform each email in the training set into a feature vector
    (feature_vector_list_training,
     integer_list_training) = create_feature_vectors.run_mnist('./input_data/mnist_train.txt')
    
    # Part 2 a/b: Train a multi-class svm model using cross-validation and 
    print('\n=====================================================================================')
    print('Problem 2 Use cross-validation to find the best classifier:')
    lambda_ = pow(2, -5)
    classifier = multi_class_prediction.multi_class_prediction_train(
                    feature_vector_list_training, integer_list_training, lambda_)

    cross_validation_error = multi_class_prediction.multi_class_prediction_cross_validation_test(
                                classifier, feature_vector_list_training, integer_list_training, 5)
    print(cross_validation_error)

import pylab
import math
import numpy as np
from operator import itemgetter
from sklearn import svm, cross_validation 

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

    (feature_vector_list_testing,
     integer_list_testing) = create_feature_vectors.run_mnist('./input_data/mnist_test.txt')
    
    # Part 2 a/b: Train a multi-class svm model using cross-validation and 
    print('\n=====================================================================================')
    print('Problem 2 Use cross-validation to find the best classifier:')
    cross_validation_error_list = []

    lambda_set = [pow(2, exponent) for exponent in range(-5, 2)]
    
    for lambda_ in lambda_set:
        print('Processing lamba: ' + str(lambda_))
        cross_validation_error = multi_class_prediction.multi_class_prediction_cross_validation(
                                    feature_vector_list_training, integer_list_training, lambda_, 5)
        print('Cross validation error: ' + str(cross_validation_error))
        cross_validation_error_list.append(cross_validation_error)

    cross_validation_error_list = [100*cross_validation_error 
                                   for cross_validation_error 
                                   in cross_validation_error_list]

    # Plot data ------------------------------------------------------------------------------------
    pylab.plot(lambda_set, cross_validation_error_list)
    
    pylab.xlabel('lambda')
    pylab.ylabel('Cross validation error (% out of 100')
    pylab.title('Cross validation error as a function of lambda')
    pylab.legend(['Cross Validation Error'], loc=2)
    pylab.grid(True)
    pylab.savefig("Cross_validation_error_lambda.png")
    pylab.show()
    pylab.close()
    pylab.clf()
    # End plot data --------------------------------------------------------------------------------

    print('\n=====================================================================================')
    print('Problem 2 Find error on test set using classifier with least error:')
    (minimum_index, minimum_cross_validation_error) = min(enumerate(cross_validation_error_list),
                                                          key = itemgetter(1))
    minimum_lambda = lambda_set[minimum_index]
    print('The best lambda value is: ' + str(minimum_lambda) + ' (with error: ' +
            str(minimum_cross_validation_error) + ')')

    classifier = multi_class_prediction.multi_class_prediction_train(
                    feature_vector_list_training, integer_list_training, minimum_lambda)

    test_error = multi_class_prediction.multi_class_prediction_test(
                    classifier, feature_vector_list_testing, integer_list_testing)
    
    print('The test error is: ' + str(test_error))

def run_libsvm():
    # Part 2/libsvm
    print('\n=====================================================================================')
    print('Problem 2/libsvm Preparing data:')
    (feature_vector_list_training,
     integer_list_training) = create_feature_vectors.run_mnist('./input_data/mnist_train.txt')

    (feature_vector_list_testing,
     integer_list_testing) = create_feature_vectors.run_mnist('./input_data/mnist_test.txt')

    # Use libsvm to train on the data and get the testing error using default parameters
    classifier = svm.SVC()
    classifier.fit(feature_vector_list_training, integer_list_training)  
    predicted_integer_list_testing = classifier.predict(feature_vector_list_testing)

    prediction_error_rate = 100*np.mean(np.array(predicted_integer_list_testing) !=
                                    np.array(integer_list_testing))
    print('The default libsvm prediction error is: ' + str(prediction_error_rate) + '%')

    # Find the cross validation error on the training set
    classifier = svm.SVC()
    scores = cross_validation.cross_val_score(classifier, feature_vector_list_training,
                                              integer_list_training, cv=10)
    print('The 10-fold cross validation error on the test set is: ' + 
          str(100*(1-scores.mean())) + '%')

    # Cross Validation varying gamma and C
    c_list = [1, 1e2, 1e4]
    gamma_list = [1e-2, 1e-3, 1e-4]

    cross_validation_error_list = []
    gamma_c_list = []

    for c in c_list:
        for gamma in gamma_list:
            classifier = svm.SVC(C=c, gamma=gamma)
            scores = cross_validation.cross_val_score(classifier, feature_vector_list_training,
                                                      integer_list_training, cv=10)

            cross_validation_error = 100*(1-scores.mean())
            print('10-fold cross validation error for gamma=' + str(gamma) +
                  ' c=' + str(c) + ': ' + str(cross_validation_error) + '%')

            cross_validation_error_list.append(cross_validation_error)
            gamma_c_list.append((gamma, c))

    (minimum_index, minimum_cross_validation_error) = min(enumerate(cross_validation_error_list),
                                                          key = itemgetter(1))

    (minimum_gamma, minimum_c) = gamma_c_list[minimum_index]

    print('Minimum 10-fold cross validation error is: ' + str(minimum_cross_validation_error) + '%'
          + ', gamma=' + str(minimum_gamma) + ', c=' + str(c))

    # Train on the data with the minimum gamma and c specified. Find the testing error.
    classifier = svm.SVC(C=minimum_c, gamma=minimum_gamma)
    classifier.fit(feature_vector_list_training, integer_list_training)  
    predicted_integer_list_testing = classifier.predict(feature_vector_list_testing)

    prediction_error_rate = 100*np.mean(np.array(predicted_integer_list_testing) !=
                                    np.array(integer_list_testing))
    print('With parameters gamma=' + str(minimum_gamma) + ', c=' +str(c) + 
          ' libsvm prediction error is: ' + str(prediction_error_rate) + '%')

import pylab
import math
import numpy as np
from operator import itemgetter

# Run Script for ps3 (Learning Support Vector Machines)
import split_training_data
import create_feature_vectors
import pegasos_svm

def run():
    # Part 1
    print('\n=====================================================================================')
    print('Problem 1 Preparing data:')
    # Split Training data into training and validation set
    split_training_data.run()
    
    # Transform each email in the training set into a feature vector
    (feature_vector_list_training,
     is_spam_list_training,
     vocabulary_list) = create_feature_vectors.run_spam('./output_data/training_set')
    
    print('\n=====================================================================================')
    print('Problem 1 Creating "svm objective as a function of iterations" graph:')
    # Part 1a: Plot the svm objective as a function of iterations
    (weight_vector, svm_objective_list) = pegasos_svm.pegasos_svm_train(feature_vector_list_training,
                                                                        is_spam_list_training,
                                                                        pow(2, -5))
    
    # Plot data ------------------------------------------------------------------------------------
    m = len(feature_vector_list_training)
    iterations = [(i+1)*m for i in range(0, len(svm_objective_list))]
    
    pylab.plot(iterations, svm_objective_list)
    
    pylab.xlabel('Number of Iterations')
    pylab.ylabel('SVM Objective')
    pylab.title('SVM Objective as a function of iterations')
    pylab.grid(True)
    pylab.savefig("SVM_Objective_Graph.png")
    #pylab.show()
    pylab.close()
    pylab.clf()
    # End plot data --------------------------------------------------------------------------------
    
    print('\n=====================================================================================')
    print('Problem 1 Creating "Training Error as a function of lambda" graph:')
    # Part 1b/c:
    # Setup validation error data
    (feature_vector_list_validation,
     is_spam_list_validation,
     _) = create_feature_vectors.run_spam('./output_data/validation_set', vocabulary_list)
    
    # Setup data for loop
    lambda_set = [pow(2,power) for power in range(-9, 2)]
    
    weight_list = []
    average_training_error_list = []
    average_hinge_loss_error_list = []
    average_validation_error_list = []
    
    # Get data for varying values of lambda
    for lambda_ in lambda_set:
        print('Processing data for lambda: ' + str(lambda_))
        (this_weight_vector, _) = pegasos_svm.pegasos_svm_train(feature_vector_list_training,
                                                                is_spam_list_training, lambda_)
        weight_list.append(this_weight_vector)
    
        # Average training error
        average_training_error = pegasos_svm.pegasos_svm_test(this_weight_vector,
                                                              feature_vector_list_training,
                                                              is_spam_list_training)
        average_training_error_list.append(average_training_error)
    
        # Average hinge error
        average_hinge_loss_error = pegasos_svm.calculate_average_hinge_loss_error(
                                        this_weight_vector, feature_vector_list_training,
                                        is_spam_list_training)
        average_hinge_loss_error_list.append(average_hinge_loss_error)
    
        # Average validation error
        average_validation_error = pegasos_svm.pegasos_svm_test(this_weight_vector,
                                                              feature_vector_list_validation,
                                                              is_spam_list_validation)
        average_validation_error_list.append(average_validation_error)
    
    # Normalize to percentage out of 100
    average_training_error_list = [100*average_training_error 
                                   for average_training_error 
                                   in average_training_error_list]
    
    average_hinge_loss_error_list = [100*average_hinge_loss_error 
                                   for average_hinge_loss_error 
                                   in average_hinge_loss_error_list]
    
    average_validation_error_list = [100*average_validation_error 
                                     for average_validation_error 
                                     in average_validation_error_list]
    
    # Plot data ------------------------------------------------------------------------------------
    log_lambda = [math.log(lambda_, 2) for lambda_ in lambda_set]
    
    # Plot logic
    pylab.plot(log_lambda, average_training_error_list)
    pylab.plot(log_lambda, average_hinge_loss_error_list)
    pylab.plot(log_lambda, average_validation_error_list)
    
    pylab.xlabel('log base 2 of lambda')
    pylab.ylabel('Average error (% out of 100')
    pylab.title('Average Errors as a function of log base 2 of lambda')
    pylab.legend(['Training Error', 'Hinge Loss Error', 'Validation Error'], loc=2)
    pylab.grid(True)
    pylab.savefig("Average_Error_Over_Log2_Lambda.png")
    #pylab.show()
    pylab.close()
    pylab.clf()
    # End plot data --------------------------------------------------------------------------------
    
    print('\n=====================================================================================')
    print('Problem 1 Evaluating values:')
    # Print results for part 1:
    (minimum_index, minimum_average_validation_error) = min(enumerate(average_validation_error_list),
                                                            key = itemgetter(1))
    print('The minimum of the validation error was: ' + str(minimum_average_validation_error) + '%')
    
    # Setup test set data
    (feature_vector_list_training,
     is_spam_list_training,
     vocabulary_list) = create_feature_vectors.run_spam('./input_data/spam_train.txt')

    (feature_vector_list_test,
     is_spam_list_test,
     _) = create_feature_vectors.run_spam('./input_data/spam_test.txt', vocabulary_list)
    
    # Calculate Test Error
    minimum_lambda = lambda_set[minimum_index]
    (minimum_validation_error_classifier, _) = pegasos_svm.pegasos_svm_train(
                                                    feature_vector_list_training,
                                                    is_spam_list_training, minimum_lambda)

    test_set_error = pegasos_svm.pegasos_svm_test(minimum_validation_error_classifier,
                                                  feature_vector_list_test,
                                                  is_spam_list_test)
    
    print('Error on the test set: ' + str(test_set_error*100) + '%')
    
    # Calculate the number of support vectors
    is_support_vector = [1 if is_spam_list_training[i]*np.dot(minimum_validation_error_classifier,
                                                              feature_vector_list_training[i]) <= 1
                         else 0 for i in range(0, len(feature_vector_list_training))]
    number_of_support_vectors = sum(is_support_vector)

    print('Number of support vectors: ' + str(number_of_support_vectors))

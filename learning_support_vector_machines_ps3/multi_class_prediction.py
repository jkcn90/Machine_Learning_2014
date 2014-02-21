import numpy as np
import math
from operator import itemgetter
from pegasos_svm import pegasos_svm_train

def multi_class_prediction_cross_validation(feature_vector_list, integer_list, lambda_, k):
    validation_error_list = []

    # Split out the data into sections containing k data points.
    number_of_k_sections = int(math.ceil(len(feature_vector_list)/float(k)))
    
    for offset in range(0, number_of_k_sections):
        # Feature vector and integer list setup for training and validation
        feature_vector_list_index_end = len(feature_vector_list)
        offset_end = min(k*offset+k, feature_vector_list_index_end)

        index_training = range(0, offset) + range(offset_end, feature_vector_list_index_end)
        index_validation = range(k*offset, offset_end)

        feature_vector_list_training = feature_vector_list[index_training]
        feature_vector_list_validation = feature_vector_list[index_validation]

        integer_list_training = integer_list[index_training]
        integer_list_validation = integer_list[index_validation]

        # Get classifier and evaluate data
        classifier = multi_class_prediction_train(feature_vector_list_training, 
                                                  integer_list_training, lambda_)

        validation_error = multi_class_prediction_test(classifier, feature_vector_list_validation,
                                                       integer_list_validation)
        validation_error_list.append(validation_error)

    averaged_validation_error = sum(validation_error_list)/float(len(validation_error_list))
    return averaged_validation_error

def multi_class_prediction_test(classifier, feature_vector_list, integer_list):
    number_of_misclassifications = 0
    feature_vector_list_length = len(feature_vector_list)

    for index in range(0, feature_vector_list_length):
        feature_vector = feature_vector_list[index]
        actual_value = integer_list[index]

        predicted_value = get_predicted_value(classifier, feature_vector)
        if predicted_value != actual_value:
            number_of_misclassifications += 1

    error = number_of_misclassifications / float(feature_vector_list_length)
    return error

def multi_class_prediction_train(feature_vector_list, integer_list, lambda_, number_of_passes=20,
                                 display_intermediate_steps=False):
    unique_integers = set(integer_list)
    classifier_dictionary = dict()

    # Map each problem to a spam problem and get a classifier for each integer in the integer list
    for integer in unique_integers:
        this_classifier = map_to_spam_problem(feature_vector_list, integer_list, integer,
                                              lambda_, number_of_passes, display_intermediate_steps)
        classifier_dictionary[integer] = this_classifier

    return classifier_dictionary

def map_to_spam_problem(feature_vector_list, integer_list, integer, lambda_, number_of_passes,
                        display_intermediate_steps):
    is_spam_list = [1 if this_integer == integer else -1 for this_integer in integer_list]
    
    (classifier, _) = pegasos_svm_train(feature_vector_list, is_spam_list, lambda_,
                                        number_of_passes, display_intermediate_steps)
    return classifier

def get_predicted_value(classifier, feature_vector):
    dot_product_value_list = [(integer, np.dot(weight_vector, feature_vector))
                              for (integer, weight_vector) in classifier.items()]
    predicted_value = max(dot_product_value_list, key=itemgetter(1))[0]
    return predicted_value

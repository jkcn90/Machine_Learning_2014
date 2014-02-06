import numpy as np

def perceptron_test(weight_vector, feature_vector_list, is_spam_list):
    number_of_misclassifications = 0
    feature_vector_list_length = len(feature_vector_list)

    for index in range(0, feature_vector_list_length):
        feature_vector = feature_vector_list[index]
        actual_value = is_spam_list[index]

        predicted_value = get_predicted_value(weight_vector, feature_vector)
        if predicted_value != actual_value:
            number_of_misclassifications += 1

    error = str(number_of_misclassifications / float(feature_vector_list_length))
    return error

def perceptron_train_averaged(feature_vector_list, is_spam_list,
                              maximum_number_of_data_to_read=float('Inf'),
                              maximum_number_of_iterations=float('Inf'),
                              display_intermediate_steps=False):
    (weight_vectors_list,
     number_of_misclassifications_list,
     number_of_runs
     ) = perceptron_train_raw(feature_vector_list, is_spam_list,
                              maximum_number_of_data_to_read,
                              maximum_number_of_iterations,
                              display_intermediate_steps)

    averaged_weight_vector = sum(weight_vectors_list) / len(weight_vectors_list)
    averaged_weight_vectora = sum(weight_vectors_list)
    total_number_of_misclassifications = sum(number_of_misclassifications_list)

    return (averaged_weight_vector, total_number_of_misclassifications, number_of_runs)

def perceptron_train(feature_vector_list, is_spam_list,
                     maximum_number_of_data_to_read=float('Inf'),
                     maximum_number_of_iterations=float('Inf'),
                     display_intermediate_steps=False):
    (weight_vectors_list,
     number_of_misclassifications_list,
     number_of_runs
     ) = perceptron_train_raw(feature_vector_list, is_spam_list,
                              maximum_number_of_data_to_read,
                              maximum_number_of_iterations,
                              display_intermediate_steps)

    weight_vector = weight_vectors_list[-1]
    total_number_of_misclassifications = sum(number_of_misclassifications_list)
    return (weight_vector, total_number_of_misclassifications, number_of_runs)

def perceptron_train_raw(feature_vector_list, is_spam_list,
                         maximum_number_of_data_to_read=float('Inf'),
                         maximum_number_of_iterations=float('Inf'),
                         display_intermediate_steps=False):
    # Initialize a zero weight vector
    weight_vector = np.zeros(shape=feature_vector_list[0].size, dtype=int)
    weight_vectors_list = [weight_vector]

    # Initialize count variables
    number_of_misclassifications = float('Inf')
    number_of_misclassifications_list = []
    number_of_runs = 0

    while number_of_misclassifications > 0:
        number_of_runs += 1
        if display_intermediate_steps:
            print('Processing Run: ' + str(number_of_runs))

        (weight_vector,
         number_of_misclassifications
         ) = perceptron_train_iter(weight_vector, feature_vector_list,
                                   is_spam_list, maximum_number_of_data_to_read)

        number_of_misclassifications_list.append(number_of_misclassifications)
        weight_vectors_list.append(weight_vector)
        if display_intermediate_steps:
            print('Number of misclassifications: ' + str(number_of_misclassifications))

        if number_of_runs > maximum_number_of_iterations:
            break
    return (weight_vectors_list, number_of_misclassifications_list, number_of_runs)

def perceptron_train_iter(weight_vector, feature_vector_list,
                          is_spam_list, maximum_number_of_data_to_read=float('Inf')):
    number_of_misclassifications = 0
    
    for index in range(0, min(len(feature_vector_list), maximum_number_of_data_to_read)):
        feature_vector = feature_vector_list[index]
        actual_value = is_spam_list[index]

        predicted_value = get_predicted_value(weight_vector, feature_vector)
        if predicted_value != actual_value:
            number_of_misclassifications += 1
            weight_vector = update_weight_vector(weight_vector, feature_vector, actual_value)
    return (weight_vector, number_of_misclassifications)

def update_weight_vector(weight_vector, feature_vector, actual_value):
    modified_feature_vector = [actual_value*coordinate for coordinate in feature_vector]
    updated_weight_vector = weight_vector + modified_feature_vector
    return updated_weight_vector

def get_predicted_value(weight_vector, feature_vector):
    dot_product_value = np.dot(weight_vector, feature_vector)
    predicted_value = 1 if dot_product_value >= 0 else -1
    return predicted_value

if __name__ == '__main__':
    import create_feature_vectors

    # Train on the training set to get weight vector
    print("Running with last trained weight vector")
    (feature_vector_list_training,
     is_spam_list_training,
     vocabulary_list) = create_feature_vectors.run('./output_data/training_set')
    (weight_vector, _, _) = perceptron_train(
                                feature_vector_list_training, is_spam_list_training,
                                display_intermediate_steps=True)

    # Verify error on the training set
    error = perceptron_test(weight_vector, feature_vector_list_training, is_spam_list_training)
    print('Error on training set: ' + error)

    # Verify error on the validation set
    (feature_vector_list_validation,
     is_spam_list_validation,
     _
     ) = create_feature_vectors.run('./output_data/validation_set', vocabulary_list)
    error = perceptron_test(weight_vector, feature_vector_list_validation, is_spam_list_validation)
    print('error on validation set: ' + error)

    # Train on the training set to get averaged weight vector
    print('=======================================================================================')
    print("Running with averaged trained weight vector")
    (averaged_weight_vector, _, _) = perceptron_train_averaged(
                                        feature_vector_list_training, is_spam_list_training,
                                        display_intermediate_steps=True)

    # Verify error on the training set
    error = perceptron_test(averaged_weight_vector, feature_vector_list_training,
                            is_spam_list_training)
    print('Error on training set: ' + error)

    # Verify error on the validation set
    error = perceptron_test(averaged_weight_vector, feature_vector_list_validation,
                            is_spam_list_validation)
    print('error on validation set: ' + error)

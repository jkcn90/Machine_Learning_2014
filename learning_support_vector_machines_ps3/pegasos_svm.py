import numpy as np

def pegasos_svm_test(weight_vector, feature_vector_list, is_spam_list):
    number_of_misclassifications = 0
    feature_vector_list_length = len(feature_vector_list)

    for index in range(0, feature_vector_list_length):
        feature_vector = feature_vector_list[index]
        actual_value = is_spam_list[index]

        predicted_value = get_predicted_value(weight_vector, feature_vector)
        if predicted_value != actual_value:
            number_of_misclassifications += 1

    error = number_of_misclassifications / float(feature_vector_list_length)
    return error

def pegasos_svm_train(feature_vector_list, is_spam_list, lambda_, number_of_passes=20,
                      display_intermediate_steps=False):
    (weight_vectors_list, svm_objective_list) = pegasos_svm_train_raw(feature_vector_list,
                                                                      is_spam_list, lambda_,
                                                                      number_of_passes,
                                                                      display_intermediate_steps)
    return(weight_vectors_list[-1], svm_objective_list)

def pegasos_svm_train_raw(feature_vector_list, is_spam_list, lambda_, number_of_passes,
                          display_intermediate_steps=False):
    # Initialize a zero weight vector
    weight_vector = np.zeros(shape=feature_vector_list[0].size, dtype=int)
    weight_vectors_list = [weight_vector]

    # Initialize count variables
    svm_objective_list = []
    t = 0

    for pass_number in range(1, number_of_passes+1):
        if display_intermediate_steps:
            print('Processing pass number: ' + str(pass_number) + '/' + str(number_of_passes))

        (weight_vector, t) = perceptron_train_iter(weight_vector, feature_vector_list,
                                              is_spam_list, lambda_, t)

        svm_objective = evaluate_svm_objective(weight_vector, feature_vector_list, 
                                               is_spam_list, lambda_)
        svm_objective_list.append(svm_objective)
        weight_vectors_list.append(weight_vector)

        if display_intermediate_steps:
            print('SVM objective: ' + str(svm_objective))
            print('Weight Vector: ' + str(weight_vector))

    return (weight_vectors_list, svm_objective_list)

def perceptron_train_iter(weight_vector, feature_vector_list, is_spam_list, lambda_, t):
    for index in range(0, len(feature_vector_list)):
        t += 1
        eta = 1/float(t*lambda_)
        feature_vector = feature_vector_list[index]
        actual_value = is_spam_list[index]

        # Update weight vector
        test_value = actual_value*np.dot(weight_vector, feature_vector) 
        if test_value < 1:
            weight_vector = (1-eta*lambda_)*weight_vector + eta*actual_value*feature_vector
        else:
            weight_vector = (1-eta*lambda_)*weight_vector 
    return (weight_vector, t)

def evaluate_svm_objective(weight_vector, feature_vector_list, is_spam_list, lambda_):
    average_hinge_loss_error = calculate_average_hinge_loss_error(weight_vector, 
                                                                  feature_vector_list, is_spam_list)

    svm_objective = (lambda_ / 2.0)*np.linalg.norm(weight_vector) + average_hinge_loss_error
    return svm_objective

def calculate_average_hinge_loss_error(weight_vector, feature_vector_list, is_spam_list):
    m = len(feature_vector_list)
    hinge_loss_error_list = [max(0, 1-is_spam_list[i]*np.dot(weight_vector, feature_vector_list[i]))
                             for i in range(0, m)]
    average_hinge_loss_error = 1/float(m) * sum(hinge_loss_error_list)
    return average_hinge_loss_error

def get_predicted_value(weight_vector, feature_vector):
    dot_product_value = np.dot(weight_vector, feature_vector)
    predicted_value = 1 if dot_product_value >= 0 else -1
    return predicted_value

if __name__ == '__main__':
    import create_feature_vectors

    (feature_vector_list,
     is_spam_list,
     vocabulary_list) = create_feature_vectors.run('./output_data/training_set')

    (weight_vector, svm_objective_list) = pegasos_svm_train(feature_vector_list,
                                                            is_spam_list, pow(2, -5),
                                                            display_intermediate_steps=True)
    error = pegasos_svm_test(weight_vector, feature_vector_list, is_spam_list)
    print(error)

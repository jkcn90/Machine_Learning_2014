import os
import shutil
import pylab

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
import create_feature_vectors
import perceptron

# Part 1: Split Training data into training and validation set
split_training_data.run()

# Part 2: Transform each email in the training set into a feature vector
(feature_vector_list_training,
 is_spam_list_training,
 vocabulary_list) = create_feature_vectors.run('./output_data/training_set')

# Part 3/4: Train the data on the training set and return the last weight vector. Test the percent
# error when this weight is run on the validation set
print('\n=========================================================================================')
print('Problem 4:')
(weight_vector,
 total_number_of_misclassifications,
 number_of_runs) = perceptron.perceptron_train(feature_vector_list_training, is_spam_list_training)

(feature_vector_list_validation,
 is_spam_list_validation,
 _) = create_feature_vectors.run('./output_data/validation_set', vocabulary_list)

training_set_error = perceptron.perceptron_test(
                            weight_vector, feature_vector_list_training, is_spam_list_training)
validation_set_error = perceptron.perceptron_test(
                            weight_vector, feature_vector_list_validation, is_spam_list_validation)

print('Total number of misclassifications: ' + str(total_number_of_misclassifications))
print('Training set error: ' + str(training_set_error))
print('Validation set error: ' + str(validation_set_error))

# Part 5: Find words in the vocabulary with the most positive and negative weights
print('\n=========================================================================================')
print('Problem 5:')
sorted_weight_index_least_to_greatest = sorted(range(len(weight_vector)),
                                               key=lambda k: weight_vector[k])
top_most_positive_weights = [vocabulary_list[index]
                             for index in sorted_weight_index_least_to_greatest[-15:]]
top_most_positive_weights = list(reversed(top_most_positive_weights))

top_most_negative_weights = [vocabulary_list[index]
                             for index in sorted_weight_index_least_to_greatest[:15]]

print("Top 15 words with most positive weights: " + str(top_most_positive_weights))
print("Top 15 words with most negative weights: " + str(top_most_negative_weights))

# Part 6: Run the averaged perceptron algorithm where the weight vector is the average of all the
# weight vectors in the run
(weight_vector_averaged, _, _) = perceptron.perceptron_train_averaged(
                                    feature_vector_list_training, is_spam_list_training)

# Part 7: Train the perceptron and averaged perceptron algorithm on N = 100, 200, 400, 800, 2000,
# 4000 rows of the training data. Evaluate the corresponding validation error on all the validation
# data
print('\n=========================================================================================')
print('Problem 7:')
N = [100, 200, 400, 800, 2000, 4000]
validation_set_error_n_list = []
validation_set_error_averaged_n_list = []
number_of_runs_list = []
number_of_runs_averaged_list = []

for n in N:
    # Perceptron Algorithm data
    (weight_vector_n,
     total_number_of_misclassifications_n,
     number_of_runs_n) = perceptron.perceptron_train(
                            feature_vector_list_training, is_spam_list_training,
                            maximum_number_of_data_to_read=n)
    number_of_runs_list.append(number_of_runs_n)

    # Averaged Perceptron Algorithm data
    (weight_vector_averaged_n,
     total_number_of_misclassifications_averaged_n,
     number_of_runs_averaged_n) = perceptron.perceptron_train_averaged(
                                        feature_vector_list_training, is_spam_list_training,
                                        maximum_number_of_data_to_read=n)
    number_of_runs_averaged_list.append(number_of_runs_averaged_n)

    # Perceptron Algorithm Validation error
    validation_set_error_n = perceptron.perceptron_test(
                                weight_vector_n, feature_vector_list_validation,
                                is_spam_list_validation)
    validation_set_error_n_list.append(validation_set_error_n)

    # Averaged Perceptron Algorithm Validation error
    validation_set_error_averaged_n = perceptron.perceptron_test(
                                            weight_vector_averaged_n,
                                            feature_vector_list_validation,
                                            is_spam_list_validation)
    validation_set_error_averaged_n_list.append(validation_set_error_averaged_n)

print('N: ' + str(N))
print('Validation Error for Perceptron Algorithm: ' + str(validation_set_error_n_list))
print('Validation Error for Averaged Perceptron Algorithm: ' +
        str(validation_set_error_averaged_n_list))

# Plot data
y = [float(validation_error)*100 for validation_error in validation_set_error_n_list]
y_averaged = [float(validation_error)*100 
              for validation_error in validation_set_error_averaged_n_list]

pylab.plot(N, y)
pylab.plot(N, y_averaged)

pylab.xlabel('Number of Rows (n)')
pylab.ylabel('Validation Error (% scale of 100)')
pylab.title('Validation Error as a function of Number of Rows of Data')
pylab.grid(True)
pylab.legend(['Perceptron Algorithm', 'Averaged Perceptron Algorithm'])
pylab.savefig("Validation_Error_Graph.png")
#pylab.show()
pylab.close()
pylab.clf()

# Part 8: Perceptron Iterations as a function of N
print('Problem 8:')
print('\n=========================================================================================')
print('N: ' + str(N))
print('Number of Iterations for Perceptron Algorithm: ' + str(number_of_runs_list))
# Plot data
y = number_of_runs_list

pylab.plot(N, y)

pylab.xlabel('Number of Rows (n)')
pylab.ylabel('Number of Iterations (n'')')
pylab.title('Number of Iterations as a function of Number of Rows of Data')
pylab.ylim((min(y), max(y)+1))
pylab.grid(True)
pylab.savefig("Number_of_Iterations_Graph.png")
#pylab.show()
pylab.close()
pylab.clf()


# Part 9/10: Find a good configuration that gives a low validation error on the validation set
print('Problem 9/10:')
print('\n=========================================================================================')

# Part 10: Train on the initial training set and give the error on the testing set
print('Problem 10:')
print('\n=========================================================================================')

# Train on the entire training set
(spam_feature_vector_list_training,
 spam_is_spam_list_training,
 spam_vocabulary_list) = create_feature_vectors.run('./input_data/spam_train.txt')

# Run the Perceptron Algorithm of choice
(spam_weight_vector, _, _) = perceptron.perceptron_train(
                                spam_feature_vector_list_training, spam_is_spam_list_training)

# Get Validation error on the testing set
(spam_feature_vector_list_validation,
 spam_is_spam_list_validation,
 _) = create_feature_vectors.run('./input_data/spam_test.txt', spam_vocabulary_list)

spam_validation_set_error = perceptron.perceptron_test(spam_weight_vector,
                                                       spam_feature_vector_list_validation,
                                                       spam_is_spam_list_validation)

print('Validation set error for testing set: ' + str(spam_validation_set_error))

print('\n=========================================================================================')
print('Script complete')

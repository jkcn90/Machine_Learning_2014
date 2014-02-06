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
import create_feature_vectors
import perceptron

# Part 1: Split Training data into training and validation set
split_training_data.run()

# Part 2: Transform each email in the training set into a feature vector
(feature_vector_list_training, is_spam_list_training, vocabulary_list) = create_feature_vectors.run('./output_data/training_set')

# Part 3/4: Train the data on the training set and return the last weight vector. Test the percent
# error when this weight is run on the validation set
(weight_vector, total_number_of_misclassifications, number_of_runs) = perceptron.run_perceptron_train(feature_vector_list_training, is_spam_list_training)

(feature_vector_list_validation, is_spam_list_validation, _) = create_feature_vectors.run('./output_data/validation_set', vocabulary_list)
validation_set_error = perceptron.run_perceptron_test(weight_vector, feature_vector_list_validation, is_spam_list_validation)

# Part 5: Find words in the vocabulary with the most positive and negative weights
sorted_weight_index_least_to_greatest = sorted(range(len(weight_vector)), key=lambda k: weight_vector[k])
top_most_positive_weights = [vocabulary_list[index]
                             for index in sorted_weight_index_least_to_greatest[-15:]]
top_most_negative_weights = [vocabulary_list[index]
                             for index in sorted_weight_index_least_to_greatest[:16]]

print("Top 15 words with most positive weights: " + str(top_most_positive_weights))
print("With weights: " + str(weight_vector[sorted_weight_index_least_to_greatest[-15:]]))

print("Top 15 words with most negative weights: " + str(top_most_negative_weights))
print("With weights: " + str(weight_vector[sorted_weight_index_least_to_greatest[:16]]))

# Part 6: 

print('Script complete')

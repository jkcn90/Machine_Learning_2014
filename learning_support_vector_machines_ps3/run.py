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

# Run Script for ps3 (Learning Support Vector Machines)
import split_training_data
import create_feature_vectors
import pegasos_svm

# Part 1
# Split Training data into training and validation set
split_training_data.run()

# Transform each email in the training set into a feature vector
(feature_vector_list,
 is_spam_list,
 vocabulary_list) = create_feature_vectors.run('./output_data/training_set')

# Part 1a: Plot the svm objective as a function of iterations
(weight_vector, svm_objective_list) = pegasos_svm.pegasos_svm_train(feature_vector_list,
                                                                is_spam_list, pow(2, -5))

# Plot data
m = len(feature_vector_list)
iterations = [(i+1)*m for i in range(0, len(svm_objective_list))]

pylab.plot(iterations, svm_objective_list)

pylab.xlabel('Number of Iterations')
pylab.ylabel('SVM Objective')
pylab.title('SVM Objective as a function of iterations')
pylab.grid(True)
pylab.savefig("SVM_Objective_Graph.png")
pylab.show()
pylab.close()
pylab.clf()


print('\n=========================================================================================')
print('Script complete')

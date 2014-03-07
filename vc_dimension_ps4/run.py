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

# Run Script for ps4 (VC-dimension and Decision Trees)
import preprocess_data
import get_training_accuracy
from sklearn.tree import DecisionTreeClassifier as dtc

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

# Part 1/2: Split Training data into training and validation set, fill in missing values, and map
# categorical features to boolean features
(training_data, training_labels, validation_data, validation_labels) = preprocess_data.run(0.7)

max_depth_list = range(1, 31)
training_accuracy_list = []
validation_accuracy_list = []
for this_max_depth in max_depth_list:
    print('Processing max depth: ' + str(this_max_depth) + '/' + str(len(max_depth_list)))
    clf = dtc(criterion='entropy', max_depth=this_max_depth)
    (training_accuracy, validation_accuracy) = get_training_accuracy.run(clf, training_data,
                                                                         training_labels,
                                                                         validation_data,
                                                                         validation_labels)
    training_accuracy_list.append(training_accuracy)
    validation_accuracy_list.append(validation_accuracy)
    print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)

# Plot data ------------------------------------------------------------------------------------
pylab.plot(max_depth_list, training_accuracy_list)
pylab.plot(max_depth_list, validation_accuracy_list)

pylab.xlabel('Max Depth')
pylab.ylabel('Accuracy (% out of 100')
pylab.title('Training and Validation Accuracy as function of Max Depth')
pylab.legend(['Training Accuracy', 'Validation Accuracy'], loc=2)
pylab.grid(True)
pylab.savefig("Accuracy_vs_Max_Depth.png")
#pylab.show()
pylab.close()
pylab.clf()
# End plot data --------------------------------------------------------------------------------


min_samples_leaf_list = range(1, 51)
clf = dtc(criterion='entropy', max_depth=None, min_samples_leaf=1)

print('\n=========================================================================================')
print('Script complete')

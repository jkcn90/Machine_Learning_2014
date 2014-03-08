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
import classifier_run
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.tree import export_graphviz
import get_training_accuracy
import StringIO
import pydot 

# Part 1/2: Split Training data into training and validation set, fill in missing values, and map
# categorical features to boolean features
(training_data, training_labels,
 validation_data, validation_labels) = preprocess_data.run_for_training_data(0.7)

# Align data
missing_headers = training_data.columns.diff(validation_data.columns)
if len(missing_headers) > 0:
    validation_data[missing_headers] = training_data[missing_headers]
    validation_data[missing_headers] = validation_data[missing_headers].applymap(lambda x: False)

missing_headers = validation_data.columns.diff(training_data.columns)
if len(missing_headers) > 0:
    training_data[missing_headers] = validation_data[missing_headers]
    training_data[missing_headers] = training_data[missing_headers].applymap(lambda x: False)

# Process Decision Tree
best_max_depth = classifier_run.run_max_depth(training_data, training_labels,
                                              validation_data, validation_labels)
best_min_samples_leaf = classifier_run.run_min_samples_leaf(training_data, training_labels,
                                                            validation_data, validation_labels)
print('Optimal max depth was: ' + str(best_max_depth))
print('Optimal min samples leaf: ' + str(best_min_samples_leaf))

clf = dtc(criterion='entropy', max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf)
clf.fit(training_data, training_labels)
dot_data = StringIO.StringIO()
export_graphviz(clf, out_file=dot_data, max_depth=2) 
graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
graph.write_pdf("Decision_Tree.pdf") 

(best_n_estimator, best_n_estimator_accuracy) = classifier_run.run_random_forest(
                                                   training_data, training_labels,
                                                   validation_data, validation_labels)
(best_n_estimator_modified, 
 best_n_estimator_modified_accuracy) = classifier_run.run_random_forest(
                                            training_data, training_labels, validation_data,
                                            validation_labels, best_max_depth=30,
                                            best_min_samples_leaf=1)

print('Optimal N Estimator with default settings was: ' + str(best_n_estimator) +
      ' with accuracy: ' + str(best_n_estimator_accuracy))
print('Optimal N Estimator with modified settings was: ' + str(best_n_estimator_modified) + 
      ' with accuracy: ' + str(best_n_estimator_modified_accuracy))

# Get Test error with best configuration of Decision Tree and Random Forest
(training_data, training_labels, _, _) = preprocess_data.run_for_training_data(1)
(test_data, test_labels) = preprocess_data.run_for_test_data()

# Align data
missing_headers = training_data.columns.diff(test_data.columns)
test_data[missing_headers] = training_data[missing_headers].applymap(lambda x: False)

# Decision Tree
clf = dtc(criterion='entropy', max_depth=best_max_depth, min_samples_leaf=best_min_samples_leaf)
(_, test_accuracy_dt) = get_training_accuracy.run(clf, training_data, training_labels,
                                                  test_data, test_labels)

# Random Forest
clf = rfc(n_estimators=best_n_estimator_modified, max_depth=best_max_depth,
          min_samples_leaf=best_min_samples_leaf)
(_, test_accuracy_rf) = get_training_accuracy.run(clf, training_data, training_labels,
                                                  test_data, test_labels)

print('Test accuracy for Decision Tree: ' + str(test_accuracy_dt))
print('Test accuracy for Random Forest: ' + str(test_accuracy_rf))

print('\n=========================================================================================')
print('Script complete')

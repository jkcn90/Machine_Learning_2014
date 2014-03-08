import pylab
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.ensemble import RandomForestClassifier as rfc
import get_training_accuracy
from operator import itemgetter

CURSOR_UP_ONE = '\x1b[1A'
ERASE_LINE = '\x1b[2K'

def run_max_depth(training_data, training_labels, validation_data, validation_labels):
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
    training_accuracy_list = [training_accuracy*100 for training_accuracy
                              in training_accuracy_list]
    validation_accuracy_list = [validation_accuracy*100 for validation_accuracy 
                                in validation_accuracy_list]

    pylab.plot(max_depth_list, training_accuracy_list)
    pylab.plot(max_depth_list, validation_accuracy_list)
    
    pylab.xlabel('Max Depth')
    pylab.ylabel('Accuracy (% out of 100)')
    pylab.title('Training and Validation Accuracy as function of Max Depth')
    pylab.legend(['Training Accuracy', 'Validation Accuracy'], loc=2)
    pylab.grid(True)
    pylab.savefig("Accuracy_vs_Max_Depth.png")
    #pylab.show()
    pylab.close()
    pylab.clf()
    # End plot data --------------------------------------------------------------------------------

    (best_index, best_accuracy) = max(enumerate(validation_accuracy_list), key = itemgetter(1))
    best_max_depth = max_depth_list[best_index]
    return best_max_depth
    
def run_min_samples_leaf(training_data, training_labels, validation_data, validation_labels):
    min_samples_leaf_list = range(1, 51)
    
    training_accuracy_list = []
    validation_accuracy_list = []
    for this_min_samples_leaf in min_samples_leaf_list:
        print('Processing min samples leaf: ' + str(this_min_samples_leaf) + '/' +
                str(len(min_samples_leaf_list)))
        clf = dtc(criterion='entropy', min_samples_leaf=this_min_samples_leaf)
        (training_accuracy, validation_accuracy) = get_training_accuracy.run(clf, training_data,
                                                                             training_labels,
                                                                             validation_data,
                                                                             validation_labels)
        training_accuracy_list.append(training_accuracy)
        validation_accuracy_list.append(validation_accuracy)
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
    
    # Plot data ------------------------------------------------------------------------------------
    training_accuracy_list = [training_accuracy*100 for training_accuracy
                              in training_accuracy_list]
    validation_accuracy_list = [validation_accuracy*100 for validation_accuracy 
                                in validation_accuracy_list]

    pylab.plot(min_samples_leaf_list, training_accuracy_list)
    pylab.plot(min_samples_leaf_list, validation_accuracy_list)
    
    pylab.xlabel('Min Samples Leaf')
    pylab.ylabel('Accuracy (% out of 100)')
    pylab.title('Training and Validation Accuracy as function of Min Samples Leaf')
    pylab.legend(['Training Accuracy', 'Validation Accuracy'], loc=2)
    pylab.grid(True)
    pylab.savefig("Accuracy_vs_Min_Samples_Leaf.png")
    #pylab.show()
    pylab.close()
    pylab.clf()
    # End plot data --------------------------------------------------------------------------------
    
    (best_index, best_accuracy) = max(enumerate(validation_accuracy_list), key = itemgetter(1))
    best_min_samples_leaf = min_samples_leaf_list[best_index]
    return best_min_samples_leaf

def run_random_forest(training_data, training_labels, validation_data, validation_labels,
                      best_max_depth=[], best_min_samples_leaf=[]):
    n_estimators_list = range(1, 51)
    training_accuracy_list = []
    validation_accuracy_list = []
    for this_n_estimator in n_estimators_list:
        print('Processing n estimator: ' + str(this_n_estimator) + '/' + str(len(n_estimators_list)))
        if best_max_depth == []:
            clf = rfc(n_estimators=this_n_estimator)
        else:
            clf = rfc(n_estimators=this_n_estimator, max_depth=best_max_depth,
                      min_samples_leaf=best_min_samples_leaf)
        (training_accuracy, validation_accuracy) = get_training_accuracy.run(clf, training_data,
                                                                             training_labels,
                                                                             validation_data,
                                                                             validation_labels)
        training_accuracy_list.append(training_accuracy)
        validation_accuracy_list.append(validation_accuracy)
        print(CURSOR_UP_ONE + ERASE_LINE + CURSOR_UP_ONE)
    
    # Plot data ------------------------------------------------------------------------------------
    training_accuracy_list = [training_accuracy*100 for training_accuracy
                              in training_accuracy_list]
    validation_accuracy_list = [validation_accuracy*100 for validation_accuracy 
                                in validation_accuracy_list]

    pylab.plot(n_estimators_list, training_accuracy_list)
    pylab.plot(n_estimators_list, validation_accuracy_list)
    
    pylab.xlabel('N Estimators')
    pylab.ylabel('Accuracy (% out of 100)')
    pylab.legend(['Training Accuracy', 'Validation Accuracy'], loc=2)
    pylab.grid(True)
    if best_max_depth == []:
        pylab.title('Training and Validation Accuracy as function of N Estimators')
        pylab.savefig("Accuracy_vs_N_Estimators.png")
    else:
        pylab.title('Training and Validation Accuracy as function of N Estimators With' +
                    ' Best Max Depth and Best Min Sample Leaf')
        pylab.savefig("Accuracy_vs_N_Estimators_modified.png")
    #pylab.show()
    pylab.close()
    pylab.clf()
    # End plot data --------------------------------------------------------------------------------

    (best_index, best_accuracy) = max(enumerate(validation_accuracy_list), key = itemgetter(1))
    best_n_estimator = n_estimators_list[best_index]
    return (best_n_estimator, best_accuracy)

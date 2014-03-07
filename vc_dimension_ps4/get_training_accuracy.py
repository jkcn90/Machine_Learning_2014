import numpy as np

def get_accuracy(predicted, actual):
    actual = np.array(actual)
    number_of_values = len(predicted)

    number_correct = sum([1 for ix in range(0, number_of_values) 
                          if predicted[ix]==actual[ix]])
    accuracy = number_correct / float(number_of_values)
    return accuracy

def run(clf, training_data, training_labels, validation_data, validation_labels):
    clf.fit(training_data, training_labels)

    predicted_training_labels = clf.predict(training_data)
    predicted_validation_labels = clf.predict(validation_data)

    training_accuracy = get_accuracy(predicted_training_labels, training_labels)
    validation_accuracy = get_accuracy(predicted_validation_labels, validation_labels)
    return (training_accuracy, validation_accuracy)


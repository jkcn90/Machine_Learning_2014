import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction import DictVectorizer

def import_data(data_location, header_location):
    # Import data, strip strings, map ? to nan
    headers = pd.read_table(header_location, sep=':', header=None)[0]

    data = pd.read_csv(data_location, names=headers, header=-1, index_col=False)
    data = data.applymap(lambda x: x if isinstance(x, int) else x.strip())
    data.replace('?', np.nan, inplace=True)

    #Import Lables
    labels = pd.read_csv(data_location, header=None, index_col=False)
    labels = labels.applymap(lambda x: x if isinstance(x, int) else x.strip())
    label_index = labels.columns[-1]
    labels = labels[label_index]
    return (data, labels)

def fill_in_nan_data(data):
    # Replace 0 values for age with median
    data['age'].replace(0, data['age'].median(), inplace=True)

    # Replace 0 working-hours worked with mean
    data['hours-per-week'].replace(0, data['hours-per-week'].mean(), inplace=True)

    # Replace missing string data with the most common string
    modes = data.mode().to_dict(outtype='records')[0]
    data.fillna(modes, inplace=True)
    return data

def turn_categorical_into_boolean(data):
    # Transform categorical data into boolean features
    for column in data.columns:
        if data[column].dtypes != 'int64':
           dummy_columns = pd.get_dummies(data[column], prefix = column, dummy_na=True)
           data = data.drop(column, axis=1)
           data = data.join(dummy_columns)
    return data

def split_data(data, labels, percent_training_data):
    number_of_rows_to_sample = int(len(data.index)*percent_training_data)
    rows = random.sample(data.index, number_of_rows_to_sample)
    
    training_data = data.ix[rows]
    validation_data = data.drop(rows)

    training_labels = labels.ix[rows]
    validation_labels = labels.drop(rows)
    return (training_data, validation_data, training_labels, validation_labels)

def run_for_training_data(training_validation_split):
    (data, labels) = import_data('./input_data/adult_train.txt', './input_data/features.txt')
    (training_data, validation_data,
     training_labels, validation_labels) = split_data(data, labels, training_validation_split)

    training_data = fill_in_nan_data(training_data)

    training_data = turn_categorical_into_boolean(training_data)
    validation_data = turn_categorical_into_boolean(validation_data)
    return (training_data, training_labels, validation_data, validation_labels)

def run_for_test_data():
    (data, labels) = import_data('./input_data/adult_test.txt', './input_data/features.txt')
    data = turn_categorical_into_boolean(data)
    return (data, labels)

if __name__ == '__main__':
    (training_data, training_labels, 
     validation_data, validation_labels) = run_for_training_data(0.7)
    print(training_data)
    print(validation_data)

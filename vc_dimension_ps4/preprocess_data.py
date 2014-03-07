import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction import DictVectorizer

def import_data(data_location, header_location):
    # Import data, strip strings, map ? to nan
    headers = pd.read_table(header_location, sep=':', header=None)[0]

    data = pd.read_csv(data_location, names=headers, header=0, index_col=False)
    data = data.applymap(lambda x: x if isinstance(x, int) else x.strip())
    data.replace('?', np.nan, inplace=True)

    #Import Lables
    labels = pd.read_csv(data_location, header=None, index_col=False)
    label_index = labels.columns[-1]
    labels = labels[label_index]
    return (data, labels)

def preprocess_data(data):
    # Replace missing data
    data.fillna(data.mode(), inplace=True)

    # Transform categorical data into boolean features
    for column in data.columns:
        if data[column].dtypes != 'int64':
           dummy_columns = pd.get_dummies(data[column], prefix = column)
           data = data.drop(column, axis=1)
           data = data.join(dummy_columns)
    return data

def split_data(data, percent_training_data):
    number_of_rows_to_sample = int(len(data.index)*percent_training_data)
    rows = random.sample(data.index, number_of_rows_to_sample)
    
    training_data = data.ix[rows]
    validation_data = data.drop(rows)

    return (training_data, validation_data)

def run(percent_training_data):
    (data, labels) = import_data('./input_data/adult_train.txt', './input_data/features.txt')
    data = preprocess_data(data)
    (training_data, validation_data) = split_data(data, percent_training_data)
    (training_labels, validation_labels) = split_data(labels, percent_training_data)
    return (training_data, training_labels, validation_data, validation_labels)

if __name__ == '__main__':
    (training_data, training_labels, validation_data, validation_labels) = run(0.7)
    print(training_data)
    print(validation_data)

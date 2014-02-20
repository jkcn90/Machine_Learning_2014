import itertools
import numpy as np
from collections import Counter

def file_to_list(file_name):
    with open(file_name, 'r') as f:
        output = f.read()
    email_list = output.split('\n')
    email_list = [coordinate
                  for coordinate in email_list
                  if coordinate != '']
    return email_list

def clean_email_list(raw_email_list):
    # Each line will start with a boolean value indicating if it is spam or not. Seperate this
    # value. Our output will be (message, True/False).
    email_list_and_boolean = [(set(coordinate[1:].split()), 1 if coordinate[0] == '1' else -1)
                              for coordinate in raw_email_list]
    (email_list, is_spam) = zip(*email_list_and_boolean)
    return (email_list, is_spam)

def create_vocabulary_list(email_list, threshold):
    vocabulary_list_and_count = Counter(itertools.chain.from_iterable(email_list))
    vocabulary_list = [word for (word, count)
                       in vocabulary_list_and_count.items()
                       if count >= threshold]
    return vocabulary_list

def create_feature_vector_list(email_list, vocabulary_list):
    feature_vector_list = [create_feature_vector(email, vocabulary_list)
                           for email in email_list]
    return feature_vector_list

def create_feature_vector(email, vocabulary_list):
    feature_vector = [1 if word in email else 0 for word in vocabulary_list]
    return feature_vector

def run(input_file, vocabulary_list = []):
    raw_email_list = file_to_list(input_file)
    (email_list, is_spam_list) = clean_email_list(raw_email_list)

    if len(vocabulary_list) == 0:
        vocabulary_list = create_vocabulary_list(email_list, 30)

    feature_vector_list = create_feature_vector_list(email_list, vocabulary_list)
    feature_vector_list = np.array(feature_vector_list)
    is_spam_list = np.array(is_spam_list)
    return (feature_vector_list, is_spam_list, vocabulary_list)

if __name__ == '__main__':
    (feature_vector_list, is_spam_list) = run('./output_data/training_set')
    print(feature_vector_list)
    print(is_spam_list)

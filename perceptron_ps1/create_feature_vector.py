import itertools
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
    # Each line will start with a boolean value indicating if it is spam or not. Remove this boolean
    # indicator.
    email_list = [set(coordinate[1:].split()) if coordinate[0] in ('0', '1')
              else set(coordinate.split())
              for coordinate in raw_email_list]
    return email_list

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

if __name__ == '__main__':
    raw_email_list = file_to_list('./output_data/training_set')
    email_list = clean_email_list(raw_email_list)
    vocabulary_list = create_vocabulary_list(email_list, 30)
    feature_vector_list = create_feature_vector_list(email_list, vocabulary_list)
    print(feature_vector_list)

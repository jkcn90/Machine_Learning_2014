from itertools import islice

def create_training_set(input_file, output_folder):
    slice_text_file(input_file, output_folder + 'training_set', 0, 4000)

def create_validation_set(input_file, output_folder):
    slice_text_file(input_file, output_folder + 'validation_set', 4000, 5000)

def slice_text_file(input_file, output_file, start_line, end_line):
    with open(input_file, 'r') as f_input:
        with open(output_file, 'a') as f_output:
            output_data = list(islice(f_input, start_line, end_line))
            for line in output_data:
                f_output.write(line)

def run():
    create_training_set('./input_data/spam_train.txt', './output_data/')
    create_validation_set('./input_data/spam_train.txt', './output_data/')

if __name__ == '__main__':
    run()

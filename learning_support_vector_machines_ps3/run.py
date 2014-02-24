import os
import shutil

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
import problem1_run
import problem2_run

#problem1_run.run()
#problem1_run.run()
problem2_run.run()

print('\n=========================================================================================')
print('Script complete')

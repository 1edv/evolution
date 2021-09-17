Dear Reader,

This file contains a copy of the main project directory shared on CodeOcean and on the Google Cloud VM :

"
All the code and data for reproducing our results in a single location. All code is available as live jupyter notebooks and all data is linked to the code making this directory fully functional. We will describe the structure of this repository here and explain the contents of each folder in detail.

Directories :


### results_summary
- This notebook reproduces the prediction performance scatterplots and contains the prediction tables for each model including the benchmarking models.
- The notebook here reproduces every plot in the directory.
- A copy of the code in this notebook is run when the 'Reproducible Run' button is pressed on CodeOcean. This is because CodeOcean is slow to run - all the other notebooks are fully functional and can be run on the Google Cloud VM shared with the project ( or on any other machine that hosts all the code here). 



### code 
- Subdirectories : 
#### referee_response 
- This folder contains all the code referenced in the response to Referee 3 in the second round in fully functional form. It is organized into the following subfolders with self-explanatory titles. Each subfolder contains numbered notebooks that can be read and run in order to reproduced the results  :
##### benchmarking_models
##### biochemical_model
##### gpu_only_model
##### reproduce_test_data_performance
##### tpu_model

There is another subdirectory called CodeOcean_run here, it is simply there because of CodeOcean's directory structure requirements and the contents can be better accessed in the results_summary directory mentioned above.


### data 
- 47.18GB 
- Contains the data associated with the project
Subdirectories of data :
#### Glu 
The training data for the complex media. The *.h5 are a subset of the training_data_Glu.txt files.
#### SC_Ura
The training data for the defined media.
#### native_sequenes_only
Simply a list of native chunks of DNA. The data appears in more usable forms elsewhere
#### test_data
Some of the prominent test data used in the project. This is not an exhaustive list and the notebooks explain exactly how to extract information from the files. Please be careful if deviating from the instructions listed in the notebooks.



### github_repo
- A copy of the github repository at https://github.com/1edv/evolution. 
"
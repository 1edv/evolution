The markdown that follows describes the directory structure of our code and linked data shared on [CodeOcean](https://codeocean.com/capsule/8020974/tree). This directory on GitHub contains all of the code for (training and using) the various models used in our paper. Each subdirectory contains a README and informatively titled Jupyter notebooks that describe the code in more detail in the Markdown cells.

Please note that the phrases 'convolutional model' and the 'GPU model' refer to the same model and are used interchangeably. Similarly, 'transformer model' and 'TPU model' refer to the same model and the two are also used interchangeably. 


---
Dear Reader,

This directory contains all the code and data for reproducing our results in a single location. All code is available as live jupyter notebooks and all data is linked to the code making this directory fully functional. We will describe the structure of this repository here and explain the contents of each folder in detail.

<h1> Directories</h1>

<ul>
  <li> <h3>results_summary</h3>
    <ul>
      <li> The notebook in this directory reproduces a summary of the prediction performance scatterplots and contains the prediction tables for each model including the benchmarking models.
      <li> The notebook in the directory reproduces every plot in the directory.
      <li> A copy of the code in this notebook is all that is run when the 'Reproducible Run' button is pressed on CodeOcean. This is because CodeOcean is slow to run - all the other notebooks are fully functional and can be run on the Google Cloud VM shared with the project (or on any other machine that hosts all the code here). 
    </ul>


  <li> <h3>code</h3> 
        Subdirectories : 
        <ul>
      <li> <h4>referee_response</h4>
            This folder contains all the code referenced in the response to Referee 3 in the second round in fully functional form. It is organized into the following subfolders with self-explanatory titles. Each subfolder contains numbered notebooks that can be read and run in order to reproduced the results  :
      <ul>
      <li> <h4>benchmarking_models</h4>
      <li> <h4>biochemical_model</h4>
       <li> <h4>gpu_only_model</h4>The convolutional model is referred to as the "gpu_only_model" in this repo. We recommended that future readers use the transformer model instead, for their applications going forward. The convolutional model will be deprecated and is unlikely to be supported by future versions of TensorFlow.
      <li> <h4>reproduce_test_data_performance</h4>
      <li> <h4>tpu_model</h4>The transformer model is referred to as the "tpu_model" in this repo. This is our recommended model for future users.
    </ul>
    </ul>
    <br>
There is another subdirectory called CodeOcean_run here, it is simply there because of CodeOcean's directory structure requirements and CodeOcean_run's contents can be better accessed in the results_summary directory mentioned above. Pressing the 'Reproducible Run' button on CodeOcean runs the code here and reproduces Fig. R1 and Fig. R2 in the Reviewer response.


  <li> <h3>data </h3>
        47.18GB. Contains the data associated with the project. Subdirectories of data :
    <ul>
      <li> <h4>Glu </h4>
              The training data for the complex media. The *.h5 are a subset of the training_data_Glu.txt files.
      <li> <h4>SC_Ura</h4>
              The training data for the defined media.
      <li> <h4>native_sequenes_only</h4>
              Simply a list of native chunks of DNA. The data appears in more usable forms elsewhere
      <li> <h4>test_data</h4>
              Some of the prominent test data used in the project. This is not an exhaustive list and the notebooks explain exactly how to extract information from the files. Please be careful if deviating from the instructions listed in the notebooks. The full list of test data is linked in the answers to each specific question in the point-by-point response.
    </ul>


 </ul>

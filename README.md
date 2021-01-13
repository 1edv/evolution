Dear User,

This repository contains the code for our manuscript : 

### A complete fitness landscape model reveals the evolutionary history and future evolvability of eukaryotic cis-regulatory DNA sequences

All data and code used in the manuscript can be downloaded from : https://console.cloud.google.com/storage/browser/evolution_bucket

The conda environment required for running this notebook can be installed and activated by running the following on the command line from within this folder: 
<code>conda env create -f evolution_env.yml </code>  \
<code>conda activate me</code>


The directory is organized into the following folders :
<li> model : contains the notebooks for preprocessing the data, training the model and using the model. This model can run on TPUs and GPUs. A subfolder called 'gpu_only_model' contains the same for the version of the model that runs on GPUs only (and not TPUs) described in the manuscript.
    
<li> trajectories : contains the notebooks for computing the sequence trajectories under different evolutionary regimes
    
<li> ecc_mr_fr : contains the notebooks for computing the Expression Conservation Coefficient (ECC), Mutational Robustness and Fitness Responsivity.
    
<li> evolvability : contains the notebooks for computing the evolvability vectors and their two-dimensional representations. Also has examples for generating the landscape visualizations shown in the manuscript that the user may use for their own new sequences.
    
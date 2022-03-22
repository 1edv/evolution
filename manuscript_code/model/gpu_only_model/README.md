## Important note for future readers: 
### Please use the [transformer model](https://github.com/1edv/evolution/tree/master/manuscript_code/model/tpu_model) for your future applications instead of the convolutional model in this directory 
The “convolutional model” described in this study was simply the ‘snapshot’ of the model architecture (during our iterations in development) that we happened to use when designing some of our early validation experiments in Fig. 2. The convolutional model in this directory is quite unconventional and quirky: the input data has an unused singleton dimension, the forward and reverse strand sequences need to be input separately, the shapes of convolutional kernels are atypical and the code still uses the old tf.Session() api). (We still needed to describe our (inefficient and unconventional) “convolutional model” in our manuscript to accurately communicate the original model used to design some of these validation experiments noted above. We could not change the “convolutional model” architecture (to make it more efficient, for instance) after the experiments had concluded because we needed to report the original convolutional model architecture exactly as it was originally (inefficiently) designed, for posterity, in our manuscript. Even though the “convolutional model” predicts expression well (Fig. 1a, Extended Data Fig. 1, etc.), if one were to implement a convolutional model for this task again, we would probably recommend that they design it differently.) <br>In the process of developing an optimal model architecture for this task by iterating and experimenting, we made many changes and additions to the architecture, which lead to the eventual “transformer model” (referred to as the "tpu model" in this [repo](https://github.com/1edv/evolution/tree/master/manuscript_code/model/tpu_model)). The “transformer model” was used for all of the analyses in the study (including the re-analysis of the Figure 2 sections where the “convolutional model” was originally used to design experiments; now part of Extended Data Figure 3 of the final published manuscript. The findings from these sections were also validated by experiments, as noted in the manuscript). Thus, even though the “convolutional model” and “transformer model” have comparable prediction performance as shown in our manuscript, we recommend that future readers use the “transformer model” for their work. The transformer model will also be more readily compatible with future versions of tf since it does not require the use of the tf.Session() api.



This directory contains the code used for training and evaluating the convolutional(GPU) model along with the saved models. It also contains the code for reproducing the prediction performance scatterplots using the saved convolutional(GPU) model. The notebooks describe the corresponding code in detail.

<ul>

  <li><h4>Using the saved convolutional(GPU) model to make predictions and reproduce results:</h4>
    <code>1_use_gpu_model_to_generate_results.ipynb</code>. The <code>1_use_gpu_model_to_generate_results_defined_media.ipynb</code> does the same for the defined media model.

  <li><h4>Training a convolutional(GPU) model:</h4>
      <code>2_train_gpu_only_model.ipynb</code>
  
</ul>

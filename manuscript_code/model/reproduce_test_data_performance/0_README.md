This directory contains the various test data used to generate the scatterplots for evaluating the model performances throughout the manuscript.

### This notebook takes the reader through all the test data for model prediction performance validation (shown as scatterplots) used throughout the manuscript and creates clean and usable files for each panel

For each saved file, the fields correspond to :
<pre>
`sequence` : the sequence
`Measured Expression` : the measured expression (or expression change in the case of ED Fig. 2f-i) in the corresponding media 
`Predicted Expression` : the predicted expression (or expression change in the case of ED Fig. 2f-i) in the corresponding media using the TPU model.
</pre>
We have shown that using other models with equivalent predictive power leads to equivalent results and biological conclusions
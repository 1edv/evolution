#### Important Note for Readers : 
- Our test datasets in the manuscript (for example the ones used in Fig. 1b,c, Extended Data Fig. 2, Supplementary Fig. 4, etc. ) are not simply held-out subsets of the training datasets. They are separate test datasets generated as part of completely independent experiments with lower-complexity (~1000 fold lower sequence diversity) libraries than the large-scale training data generation experiments resulting in expression measurements with a low measurement-error. 
- Since the training data and the test data are collected in different experiments, the units of expression are on different unrelated scales (the units are arbitrary units local to experiments and not absolute comparable units across experiments) because of the nature of GPRA/Sort-seq experiments.
- Note that these sequences already have the constant flanks attached. Please use the following function if you would like to extract the 80bp variable region of the test data :

<code>
def population_remove_flank(population) : 
    return_population = []
    for i in range(len(population)): 
        return_population= return_population + [(population[i][17:-13])]
    return return_population
</code>
<b>The test data can be found in this directory along with our model's predictions for the sequence</b> 

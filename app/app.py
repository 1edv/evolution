import sys 
sys.path.insert(0, './')
import app_aux
from app_aux import *  
tf.reset_default_graph() 
tf.keras.backend.clear_session()
gc.collect() 

st.set_page_config(
    page_title=" Evolution, Evolvability and Expression",
    layout="wide",
    initial_sidebar_state="expanded",)
 
st.write('Path Prefix is ' + path_prefix)

@st.cache
def download_link(object_to_download, download_filename, download_link_text):
    """
    Generates a link to download the given object_to_download.

    object_to_download (str, pd.DataFrame):  The object to be downloaded.
    download_filename (str): filename and extension of file. e.g. mydata.csv, some_txt_output.txt
    download_link_text (str): Text to display for download link.

    Examples:
    download_link(YOUR_DF, 'YOUR_DF.csv', 'Click here to download data!')
    download_link(YOUR_STRING, 'YOUR_STRING.txt', 'Click here to download your text!')

    """
    if isinstance(object_to_download,pd.DataFrame):
        object_to_download = object_to_download.to_csv(index=False)

    # some strings <-> bytes conversions necessary here
    b64 = base64.b64encode(object_to_download.encode()).decode()

    return f'<a href="data:file/txt;base64,{b64}" download="{download_filename}">{download_link_text}</a>'

@st.cache
def parse_seqs(sequences) :
    sequences = population_add_flank(sequences) ### NOTE : This is different from all other functions ! (User input doesn't have flanks)
    for i in tqdm(range(0,len(sequences))) : 
        if (len(sequences[i]) > 110) :
            sequences[i] = sequences[i][-110:]
        if (len(sequences[i]) < 110) : 
            while (len(sequences[i]) < 110) :
                sequences[i] = 'N'+sequences[i]



    A_onehot = np.array([1,0,0,0] ,  dtype=np.bool)
    C_onehot = np.array([0,1,0,0] ,  dtype=np.bool)
    G_onehot = np.array([0,0,1,0] ,  dtype=np.bool) 
    T_onehot = np.array([0,0,0,1] ,  dtype=np.bool)
    N_onehot = np.array([0,0,0,0] ,  dtype=np.bool)

    mapper = {'A':A_onehot,'C':C_onehot,'G':G_onehot,'T':T_onehot,'N':N_onehot}
    worddim = len(mapper['A'])
    seqdata = np.asarray(sequences)
    seqdata_transformed = seq2feature(seqdata)


    return np.squeeze(seqdata_transformed) , sequences



st.title('The evolution, evolvability and engineering of gene regulatory DNA')
"""
[![Paper DOI:https://doi.org/10.1101/2021.02.17.430503](https://zenodo.org/badge/DOI/10.1101/2021.02.17.430503.svg)](https://doi.org/10.1101/2021.02.17.430503)&nbsp[![Star](https://img.shields.io/github/stars/1edv/evolution.svg?logo=github&style=social)](https://github.com/1edv/evolution)
&nbsp[![Follow](https://img.shields.io/twitter/follow/edv_tweets?style=social)](https://www.twitter.com/edv_tweets)
"""
valid_input = 0 

with st.beta_container() : 
    st.header('What would you like to compute?')
    mode = st.selectbox(
        '',
        ['Mutational Robustness','Evolvability vector' , "Expression"] ,
    )

with st.beta_container() : 
    st.header('Please upload your sequences :')  
    reqs = st.beta_expander('Example input file and sequence file format requirements 👉', expanded=False)
    with reqs : 
        st.write('* Every line in the file must contain just a single DNA sequence with no additional special characters, spaces, tabs or linebreaks.')
        st.write('* If there are more than 80 bases in a given sequence, the last 80 bases will be used for the downstream computations. If there are less than 80 bases in a given sequence, it will be padded with N bases in the beginning until each input sequence is 80bp in length.')
        st.write('* You can download a sample input file below.')
        
        ### START : This block of code should be ignored by users to avoid confusion 
        df = pd.read_csv(path_prefix+'sample_seqs_vs_gluexp.txt', sep = '\t' , header = None)
        sample_list = population_remove_flank(list(df.iloc[:,0].values)) # Removing the flanks is necessary here (see Methods section for details.)
        tmp_download_link = download_link("\n".join(sample_list), 'example_input_file.txt', 'Click here to download an example sequence input file')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        ### END : This block of code should be ignored by users to avoid confusion 

    if 0 :
        st.subheader('Upload the sequence file here 👇')
        uploaded_file = st.file_uploader("")

        st.write('**OR**')

        st.subheader('Paste one sequence per line here 👇')
        text_area = st.text_area(label='' , value = 'GAGGCATCGTTTTATCAGATGATAGTTTAATTAGTACGTGCAGCACCTTAAAGGATATAAGGGCCGGTAGAACATAACGC\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGACACTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGGTTGTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA')

        submit = st.button('Submit Sequences')



    cols = st.beta_columns([0.95, 0.1 , 0.95])

    with cols[0] : 
        with st.beta_container() : 
            st.subheader('Upload the sequence file here 👇')
            uploaded_file = st.file_uploader("")

    with cols[1] : 
        """### \
&nbsp;
\
&nbsp;
\
&nbsp;
\
&nbsp;
\
&nbsp;**OR**""" 

    with cols[-1] : 
        with st.beta_container() : 
            st.subheader('Paste one sequence per line here 👇')
            text_area = st.text_area(label='' , value = 'GAGGCATCGTTTTATCAGATGATAGTTTAATTAGTACGTGCAGCACCTTAAAGGATATAAGGGCCGGTAGAACATAACGC\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGACACTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGGTTGTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA')
        

    button_cols = st.beta_columns([0.95, 0.1, 0.95])
    with button_cols[-1] :
        with st.beta_container() : 
            submit = st.button('Submit Sequences')





### Read and Validate Input
if text_area and uploaded_file :
    st.warning('Warning : A file was uploaded and sequences were pasted into the textbox. This application will now proceed with the file and ignore the textbox. Please pick one mode of input if that is not the intended behaviour.')


if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file , header = None)
    valid_input = 1

elif text_area : 
    X = text_area
    X_list= X.split('\n')
    input_df = pd.DataFrame(X_list)
    valid_input = 1




#st.sidebar.header('')

condition = st.sidebar.selectbox(
    "Pick the model's environment",
    ("Defined Media", "Complex Media")
)
 


st.sidebar.header('') 


st.sidebar.image(path_prefix+'MIT_logo.png')
st.sidebar.write('')

st.sidebar.image(path_prefix+'Broad_logo.png')
st.sidebar.write('')

st.sidebar.image(path_prefix+'HHMI_logo.jpeg')

 

if valid_input : 
    with st.spinner('Loading deep transformer neural network model ...'):
        if condition == "Defined Media" :
            model_conditions='SC_Ura' #SC_Ura 
        else :
            model_conditions='Glu'
        NUM_GPU = len(get_available_gpus())
        if(NUM_GPU>0) :
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
        fitness_function_graph = tf.Graph()
        with fitness_function_graph.as_default():
            ## Next line should be a box where you can pick models (on the left side) 
            model, scaler,batch_size = load_model(model_conditions)

    
    if mode=="Expression" :
        sequences = list(input_df.iloc[:,0].values)
        single_sequence_input = 0 
        if(len(sequences) == 1) :
            single_sequence_input = 1 
            sequences = sequences+sequences
        X,_ = parse_seqs(sequences)

        with st.spinner('Predicting expression from sequence using model...'):
            Y_pred = evaluate_model(X, model, scaler, batch_size , fitness_function_graph)
        st.success('Expression prediction complete !')

        expression_output_df = pd.DataFrame([ sequences , Y_pred ]  ).transpose()
        expression_output_df.columns = ['Sequence' , 'Expression']
        expression_output_df['Expression (percentile)']  = expression_output_df['Expression'].rank(pct=1)

        if single_sequence_input==1 :
            expression_output_df = pd.DataFrame(expression_output_df.loc[0,:]).T
        
        with st.beta_container() : 
            st.header('Results')
            expression_output_df
            tmp_download_link = download_link(expression_output_df, 'expression_output_df.csv', 'Click here to download the expression results as a CSV')
            st.markdown(tmp_download_link, unsafe_allow_html=True)
            
        if 0 : ### this section is should be run for Complex Media only. And it is just for testing the app, it needs expression measurements for computing pearson's R ! 

            fig=plt.figure( figsize=(3.5,3.5/1.6) , dpi= 500, facecolor='w', edgecolor='k')
            fig.tight_layout(pad = 1)

            x,y = normalized_kde(Y_pred)
            plt.plot(x,y, c='k' ,label='Expression Distribution')
            plt.fill_between(x, 0, y ,color='gray' ,alpha=0.5)
            st.pyplot(fig)
            with open(path_prefix+'sample_seqs_vs_gluexp.txt') as f:
                reader = csv.reader(f, delimiter="\t")
                d = list(reader)

                expressions = [di[1] for di in d]
                expdata = np.asarray(expressions)
                Y = expdata.astype('float') 
                x,y = normalized_kde(Y)
                plt.plot(x,y, c='k' ,label='Expression Distribution')
                plt.fill_between(x, 0, y ,color='gray' ,alpha=0.5)
                st.pyplot(fig) 

                st.write(scipy.stats.pearsonr(Y,Y_pred)[0])



    if mode=="Evolvability vector"  or mode=="Mutational Robustness":
        epsilon = 0.1616
        def population_mutator( population_current , args) :
            population_current = population_remove_flank(population_current)
            population_next = []  
            for i in range(len(population_current)) :         
                for j in range(args['sequence_length']) : 
                #First create three copies of the same individual, one for each possible mutation at the basepair.
                    population_next.append(list(population_current[i]))
                    population_next.append(list(population_current[i]))
                    population_next.append(list(population_current[i]))
                    
                
                    if (population_current[i][j] == 'A') :
                        population_next[3*(args['sequence_length']*i + j) ][j] = 'C'
                        population_next[3*(args['sequence_length']*i + j) + 1][j] = 'G'
                        population_next[3*(args['sequence_length']*i + j) + 2][j] = 'T'
                        
                    elif (population_current[i][j] == 'C') :
                        population_next[3*(args['sequence_length']*i + j)][j] = 'A'
                        population_next[3*(args['sequence_length']*i + j) + 1][j] = 'G'
                        population_next[3*(args['sequence_length']*i + j) + 2][j] = 'T'
                    
                    elif (population_current[i][j] == 'G') :
                        population_next[3*(args['sequence_length']*i + j)][j] = 'C'
                        population_next[3*(args['sequence_length']*i + j) + 1][j] = 'A'
                        population_next[3*(args['sequence_length']*i + j) + 2][j] = 'T'
                        
                    elif (population_current[i][j] == 'T') :
                        population_next[3*(args['sequence_length']*i + j)][j] = 'C'
                        population_next[3*(args['sequence_length']*i + j) + 1][j] = 'G'
                        population_next[3*(args['sequence_length']*i + j) + 2][j] = 'A'
                    
                
            population_next= population_add_flank(population_next)        
            return list(population_next) 


        def get_snpdev_dist(population) : 
            population_fitness = np.array(evaluate_model(list(population),model,scaler,batch_size,fitness_function_graph))
            args  = {'sequence_length' : 80 , 'nucleotide_frequency' :[0.25,0.25,0.25,0.25] , 'randomizer' : np.random } 
            population_1bp_all_sequences = population_mutator(list(population) , args)
            population_1bp_all_fitness = np.array(evaluate_model(list(population_1bp_all_sequences),model,scaler,batch_size,fitness_function_graph))


            snpdev_dist = []
            for i in (range(len(population))) :   
                original_fitness = population_fitness[i]
                sequence = population[i]

                exp_dist = population_1bp_all_fitness[3*args['sequence_length']*i:3*args['sequence_length']*(i+1)]
                snpdev_dist = snpdev_dist + [np.sort((exp_dist-original_fitness))]

            sequences = population
            return snpdev_dist

        sequences = list(input_df.iloc[:,0].values)
        single_sequence_input = 0 
        if(len(sequences) == 1) :
                single_sequence_input = 1 
                sequences = sequences+sequences
        X , sequences_flanked = parse_seqs(sequences)
        with st.spinner('Computing expression from sequence using the model...'):
            Y_pred = evaluate_model(X, model, scaler, batch_size , fitness_function_graph)

        evolvability_output_df = pd.DataFrame([ sequences , Y_pred ]  ).transpose()
        evolvability_output_df.columns = ['Sequence' , 'Expression']
        
        if mode=="Evolvability vector" : 
            with st.spinner('Computing evolvability vectors for sequences...'):
                evolvability_vector = get_snpdev_dist(sequences_flanked)
            st.success('Evolvability vectors and expression computed !')
            evolvability_output_df['Evolvability vector'] = evolvability_vector

        if mode=="Mutational Robustness" :
            with st.spinner('Computing mutational robustness for sequences...'):
                evolvability_vector = get_snpdev_dist(sequences_flanked)
            st.success('Mutational robustness, evolvability vectors and expression computed !')
            evolvability_output_df['Evolvability Vector'] = evolvability_vector
            evolvability_output_df['Mutational Robustness'] = [np.sum(np.abs(i)<epsilon)/240 for i in evolvability_vector]
            evolvability_output_df['Mutational Robustness (percentile)']  = evolvability_output_df['Mutational Robustness'].rank(pct=1)
            evolvability_output_df['Expression (percentile)']  = evolvability_output_df['Expression'].rank(pct=1)

        if single_sequence_input==1 :
            evolvability_output_df = pd.DataFrame(evolvability_output_df.loc[0,:]).T
        if 1 : 
            with st.beta_container() : 
                st.header('Results')
                evolvability_output_df
                tmp_download_link = download_link(evolvability_output_df, 'evolvability_output_df.csv', 'Click here to download the results as a CSV')
                st.markdown(tmp_download_link, unsafe_allow_html=True)



with st.beta_container() : 
    st.header('Project Overview')
    image_cols = st.beta_columns([0.05 , 0.05 , 0.05  , 0.7 , 0.05, 0.05 ,0.05 ])
    with image_cols[3] :
        st.image(path_prefix+'overview.png' , caption = '')

import sys, re 
sys.path.insert(0, './')
import aux_app,app_aux
from aux_app import *  
tf.reset_default_graph() 
tf.keras.backend.clear_session()
gc.collect() 
###Functions moved here from aux_app.py
#st.write(os.environ)

def evaluate_model(X,model, scaler, batch_size,session, *graph) :
    #K.set_session(session)
    if(graph) : 
        default_graph = graph[0]

    else : 
        default_graph = tf.get_default_graph()
 
    with default_graph.as_default(): ### attempted to use a closed session is the error i get here.
        with session.as_default() : 
            #K.set_session(session)
            NUM_GPU = len(get_available_gpus())
            if(len(X[0])==80):
                X = population_add_flank(X)
            if( type(X[0])==str or type(X[0])==np.str_) : 
                X = seq2feature(X)
            if NUM_GPU == 0 :    ### Pad for TPU evaluation 
                if(X.shape[0]%batch_size == 0) :
                    Y_pred = model.predict(X , batch_size = batch_size , verbose=1)
                if(X.shape[0]%batch_size != 0) :
                    n_padding = (batch_size*(X.shape[0]//batch_size + 1) - X.shape[0])
                    X_padded = np.concatenate((X,np.repeat(X[0:1,:,:],n_padding,axis=0)))
                    Y_pred_padded = model.predict(X_padded , batch_size = batch_size , verbose=1)
                    Y_pred = Y_pred_padded[:X.shape[0]]
            if NUM_GPU > 0 :    ### Pad for GPU evaluation 
                Y_pred = model.predict(X , batch_size = batch_size , verbose=1)
            Y_pred = [float(x) for x in Y_pred]
            Y_pred = scaler.inverse_transform(Y_pred)
    
    return Y_pred
    
@st.cache(allow_output_mutation=True)
def load_model(model_conditions ) : 
    NUM_GPU = len(get_available_gpus())
    dir_path=os.path.join(path_prefix+'models',model_conditions)
    model_path=os.path.join(dir_path,"fitness_function.h5")

    ### Load the parameters used for training the model
    f = open(os.path.join(dir_path,'model_params.pkl'),"rb")
    model_params = pickle.load(f)
    batch_size = model_params['batch_size']
    f.close()


    
    ### Load the model on multiple CPU/GPU(s) with the largest possible batch size
    scaler= sklearn.externals.joblib.load(os.path.join(dir_path,'scaler.save'))
    model_params['batch_size'] = np.power(2,10 + NUM_GPU)
    batch_size = model_params['batch_size']
    model_params['device_type'] = 'gpu'
    model = fitness_function_model(model_params)
    model.load_weights(model_path)
    if NUM_GPU > 1 :
        model = tf.keras.utils.multi_gpu_model(model,NUM_GPU,cpu_merge=True,cpu_relocation=False)

    if 0 : #Change to 1 if using TPU ## Changing the batch size on using the tf.keras.models.load_model is not permitted,but TPU needs this
        scaler= sklearn.externals.joblib.load(os.path.join(dir_path,'scaler.save'))
        batch_size = model_params['batch_size']
        model_params['device_type'] = 'tpu'
        model = fitness_function_model(model_params)
        model.load_weights(model_path)
        
        if(model_params['device_type']=='tpu'):
            tpu_name = os.environ['TPU_NAME']
            tpu_grpc_url = TPUClusterResolver(tpu=[tpu_name] , zone='us-central1-a').get_master()
            if(tpu_grpc_url) : 
                model = tf.contrib.tpu.keras_to_tpu_model(model,
                        strategy=tf.contrib.tpu.TPUDistributionStrategy(
                            tf.contrib.cluster_resolver.TPUClusterResolver(tpu_grpc_url)))

            if 0 : 
                model = tensorflow.keras.models.load_model(model_path , custom_objects={
                    'MultiHeadAttention' : MultiHeadAttention , 
                    'FeedForward' : FeedForward,
                    'correlation_coefficient' : correlation_coefficient,
                    'LayerNormalization' : LayerNormalization,
                    'rc_Conv1D' : rc_Conv1D})
                    
    model._make_predict_function()
    #model.summary()
    session = K.get_session()
    return model , scaler, batch_size,session



###

###events
from bokeh.models import ColumnDataSource, CustomJS
from bokeh.plotting import figure
from streamlit_bokeh_events import streamlit_bokeh_events
import bokeh

####Session state using the Streamlit session
if 'input' not in st.session_state:
    st.session_state['input'] = None

if 'seq_list' not in st.session_state:
    st.session_state['seq_list'] = []

if 'mutation_list' not in st.session_state:
    st.session_state['mutation_list'] = []

if 'event_result_list' not in st.session_state:
    st.session_state['event_result_list'] = []

if 'counter' not in st.session_state:
    st.session_state['counter'] = 0

if 'cmap_range' not in st.session_state:
    st.session_state['cmap_range'] = 'Absolute'

if 'print_trajectory' not in st.session_state:
    st.session_state['print_trajectory'] = 0

if 'random_sequence' not in st.session_state:
    st.session_state['random_sequence'] = None

#st.session_state = SessionState.get(input = None, seq_list = [] ,mutation_list = [], event_result_list = [] , 
#                                counter = 0, cmap_range='Absolute' , print_trajectory = 0 ,  random_sequence = None)

def reset_state() : 
    st.session_state.seq_list = [] 
    st.session_state.mutation_list = [] 
    st.session_state.event_result_list = []  
    st.session_state.p_list = [] 
    st.session_state.download_list = [] 

    st.session_state.counter = 0

###events

def plot_el_visualization(sequences_flanked):
    output = pd.DataFrame(index = ['A','C','G','T'] , columns = [i+1 for i in range(80)])
    sequences_unflanked = population_remove_flank(sequences_flanked)
    el_list , el= get_ordered_snpdev([sequences_flanked[0]]) #only the first element
    
    s = sequences_unflanked[0]
    loc_list = get_map(s)
    el_map = dict(zip(loc_list, el_list))
    for i in el_map : 
        output.loc[i] = el_map[i]
        
    output = output.fillna(el[0]) 
    output.columns = [ i for i in s]
    
    cmap_list = plt.colormaps()

        
    #with st.container() : 
        
    #select_cmap = st.expander('Expression', expanded=True)
    #with select_cmap : 
    #    cmap = st.selectbox('Please select your preferred colormap', cmap_list , index = 18)

    ###Plot with Bokeh
    ###Bokeh imports
    from math import pi
    from bokeh.io import show
    from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, PrintfTickFormatter
    from bokeh.plotting import figure
    ###
    output.columns = output.columns +[str(i+1) for i in range(len(output.columns))]
    output.columns.name = 'sequence'
    output.index.name = 'base'

    df = pd.DataFrame(output.stack(), columns=['Expression']).reset_index()
    df.columns =  ['base' , 'position' , 'Expression']
    ###
    maxima=df.loc[df.Expression.idxmax()]
    maxima.name='Max'
    minima=df.loc[df.Expression.idxmin()]
    minima.name='Min'
    ###

    # this is the colormap from the original NYTimes plot
    colors = ["#75968f", "#a5bab7", "#c9d9d3", "#e2e2e2", "#dfccce", "#ddb7b1", "#cc7878", "#933b41", "#550b1d"]
    #bokeh.palettes.__palettes__
    palette = sns.color_palette("Spectral_r" , n_colors = 256 ).as_hex()#'Blues256'
    if cmap_range =="Absolute" : 
        mapper = LinearColorMapper(palette= palette, low=3, high=16)
    if cmap_range == 'Relative' : 
        mapper = LinearColorMapper(palette=palette, low=df.Expression.min(), high=df.Expression.max())

    TOOLS = "hover,save,pan,box_zoom,reset,wheel_zoom,tap"

    p = figure(
            x_range=list(output.columns), y_range=list(output.index.values),
            x_axis_location="above", plot_width=1000, plot_height=200, #1000,200
            tools=TOOLS, toolbar_location='below',
            tooltips=[('Mutation', '@position@base'), ('Expression', '@Expression')])

    p.grid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "9px"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = pi / 3
    



    ### Shade max and min
    base_y_dict = {'A':0 , 'C' : 1 , 'G' : 2,  'T' : 3}
    max_x = int(re.split('(\d+)',str(maxima['position']))[1])-1 
    max_y = base_y_dict[maxima['base']]

    min_x = int(re.split('(\d+)',str(minima['position']))[1])-1 
    min_y = base_y_dict[minima['base']]

    from bokeh.models import BoxAnnotation
    line_color = 'black'
    box_max = BoxAnnotation(left=max_x, right=max_x+1, bottom = max_y, top = max_y+1 , line_dash = "solid" , line_width = 2 , line_color = line_color,line_alpha =1 , fill_alpha=0)
    p.add_layout(box_max)

    box_min = BoxAnnotation(left=min_x, right=min_x+1, bottom = min_y, top = min_y+1 , line_dash = "dotted" , line_width = 2 , line_color = line_color, line_alpha = 1, fill_alpha=0)
    p.add_layout(box_min)




    source = ColumnDataSource(df)
    tmp_download_link = download_link(output, 'output.csv', 'Click here to download the results as a CSV')

    renderer= p.rect(x="position", y="base", width=1, height=1,
        source=source,
        fill_color={'field': 'Expression', 'transform': mapper},
        line_color=None,
            )

    #renderer.nonselection_glyph = Null

    #p.add_glyph(    )
    color_bar = ColorBar(color_mapper=mapper, major_label_text_font_size="9px",
                        ticker=BasicTicker(desired_num_ticks=len(colors)),
                        formatter=PrintfTickFormatter(format="%d"),
                        label_standoff=6, border_line_color=None,
                        location=(0,0))
    p.add_layout(color_bar, 'right')
    p.output_backend="svg"

    #st.bokeh_chart(p , use_container_width=0)      # show the plot

    
        

                
    return s,tmp_download_link,maxima,minima,df,source,p



st.set_page_config(
    page_title=" Evolution, Evolvability and Expression",
    layout="wide",
    initial_sidebar_state="auto",)
 
#st.write('Path Prefix is ' + path_prefix)
if "HOSTNAME" in os.environ:
    path_prefix = './app/'
    #st.write('Path Prefix is ' + path_prefix)

if 0:
    path_prefix = './app/'
    st.write('Path Prefix is ' + path_prefix)
    if os.environ['platform'] == 'streamlit_sharing' and 0:
        with st.container() : 
            st.write("""
            [![Paper DOI : https://doi.org/10.1038/s41586-022-04506-6](https://badgen.net/badge/Nature%20DOI/10.1038%2Fs41586-022-04506-6/F96854)](https://doi.org/10.1038/s41586-022-04506-6)&nbsp[![Star](https://img.shields.io/github/stars/1edv/evolution.svg?logo=github&style=social)](https://github.com/1edv/evolution)
            &nbsp[![Follow](https://badgen.net/badge/twitter/Eeshit%20Dhaval%20Vaishnav)](https://twitter.com/i/user/1349259546)
            """)

            st.title('The evolution, evolvability, and engineering of gene regulatory DNA \n')
            st.title('')
            st.header('Click on the image below to use the live app now 👇')
            st.header('')
            if 0 : 
                st.write(""" 
                | [![App URL](https://raw.githubusercontent.com/1edv/evolution/master/app/overview.png)](https://1edv.github.io/evolution/) |
                | ------ |
                """)
            if 1 : 
                st.markdown('''
                    <p align = 'center'>
                    <a href='https://1edv.github.io/evolution/'><img width = "75%" style="border:4px solid black" src="https://raw.githubusercontent.com/1edv/evolution/master/app/overview.png"/></a>  

                    </p>''',
                    unsafe_allow_html=True
                )

        st.stop() 
else :
    pass 


#@st.cache
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

#@st.cache
def parse_seqs(sequences) :
    sequences = population_add_flank(sequences) ### NOTE : This is different from all other functions ! (User input doesn't have flanks)
    for i in (range(0,len(sequences))) : 
        if (len(sequences[i]) > 110) :
            sequences[i] = sequences[i][-110:]
        if (len(sequences[i]) < 110) : 
            while (len(sequences[i]) < 110) :
                sequences[i] = 'N'+sequences[i]



    A_onehot = np.array([1,0,0,0] ,  dtype=bool)
    C_onehot = np.array([0,1,0,0] ,  dtype=bool)
    G_onehot = np.array([0,0,1,0] ,  dtype=bool) 
    T_onehot = np.array([0,0,0,1] ,  dtype=bool)
    N_onehot = np.array([0,0,0,0] ,  dtype=bool)

    mapper = {'A':A_onehot,'C':C_onehot,'G':G_onehot,'T':T_onehot,'N':N_onehot}
    worddim = len(mapper['A'])
    seqdata = np.asarray(sequences)
    seqdata_transformed = seq2feature(seqdata)


    return np.squeeze(seqdata_transformed) , sequences



#############Functions and global constants
epsilon = 0.1616
#@st.cache
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
    with fitness_function_graph.as_default() : 
        with session.as_default(): 
            #population_fitness = np.array(evaluate_model(list(population),model,scaler,batch_size,session,fitness_function_graph))
            population_fitness= scaler.inverse_transform(model.predict(seq2feature(list(population)), verbose = 0)).flatten()
            args  = {'sequence_length' : 80 , 'nucleotide_frequency' :[0.25,0.25,0.25,0.25] , 'randomizer' : np.random } 
            population_1bp_all_sequences = population_mutator(list(population) , args)
            #population_1bp_all_fitness = np.array(evaluate_model(list(population_1bp_all_sequences),model,scaler,batch_size,session,fitness_function_graph))
            population_1bp_all_fitness= np.array(scaler.inverse_transform(model.predict(seq2feature(list(population_1bp_all_sequences)),batch_size = 1024 , verbose = 0))).flatten()


    snpdev_dist = []
    for i in (range(len(population))) :   
        original_fitness = population_fitness[i]
        sequence = population[i]

        exp_dist = population_1bp_all_fitness[3*args['sequence_length']*i:3*args['sequence_length']*(i+1)]
        snpdev_dist = snpdev_dist + [np.sort((exp_dist-original_fitness))]

    sequences = population
    return snpdev_dist

#### Functions for Visualization
def snpdev_str_to_list(snpdev_str) : 
    return [float(i) for i in snpdev_str.replace("\n" , "").replace("[","").replace("]","").split()]

def get_ordered_snpdev(population) : 
    with fitness_function_graph.as_default() : 
        with session.as_default(): 
            #population_fitness = np.array(evaluate_model(list(population),model,scaler,batch_size,session,fitness_function_graph))
            population_fitness= scaler.inverse_transform(model.predict(seq2feature(list(population)), verbose = 0)).flatten()
            args  = {'sequence_length' : 80 , 'nucleotide_frequency' :[0.25,0.25,0.25,0.25] , 'randomizer' : np.random } 
            population_1bp_all_sequences = population_mutator(list(population) , args)
            #population_1bp_all_fitness = np.array(evaluate_model(list(population_1bp_all_sequences),model,scaler,batch_size,session,fitness_function_graph))
            population_1bp_all_fitness= np.array(scaler.inverse_transform(model.predict(seq2feature(list(population_1bp_all_sequences)),batch_size = 1024, verbose = 0))).flatten()

    snpdev_dist = []
    for i in (range(len(population))) :   
        original_fitness = population_fitness[i]
        sequence = population[i]

        exp_dist = population_1bp_all_fitness[3*args['sequence_length']*i:3*args['sequence_length']*(i+1)]

    return exp_dist , population_fitness

def get_map(s):
    loc_list = []
    for j in range(len(s)) : 
        i = j+1
        if (s[j] == 'A') :
            loc_list = loc_list + [('C',i)]
            loc_list = loc_list + [('G',i)]
            loc_list = loc_list + [('T',i)]

        elif (s[j]  == 'C') :
            loc_list = loc_list + [('A',i)]
            loc_list = loc_list + [('G',i)]
            loc_list = loc_list + [('T',i)]

        elif (s[j] == 'G') :
            loc_list = loc_list + [('C',i)]
            loc_list = loc_list + [('A',i)]
            loc_list = loc_list + [('T',i)]

        elif (s[j]  == 'T') :
            loc_list = loc_list + [('C',i)]
            loc_list = loc_list + [('G',i)]
            loc_list = loc_list + [('A',i)]
    return loc_list
####


valid_input = 0 

############

with st.container() : 
    st.write("""
    [![Paper DOI : https://doi.org/10.1038/s41586-022-04506-6](https://badgen.net/badge/Nature%20DOI/10.1038%2Fs41586-022-04506-6/F96854)](https://doi.org/10.1038/s41586-022-04506-6)&nbsp[![Star](https://img.shields.io/github/stars/1edv/evolution.svg?logo=github&style=social)](https://github.com/1edv/evolution)
    &nbsp[![Follow](https://badgen.net/badge/twitter/Eeshit%20Dhaval%20Vaishnav)](https://twitter.com/i/user/1349259546)
    """)

    st.title('The evolution, evolvability, and engineering of gene regulatory DNA \n')
    st.title('')


#st.write('')



with st.container() : 
    st.header('What would you like to compute?')
    mode = st.selectbox(
        '',
        ["Visualize Sequences and Generate Trajectories",'Mutational Robustness','Evolvability Vector' , 
        "Expression","Interpretability : In Silico Mutagenesis (ISM) score"] ,
    )

with st.container() : 
    st.header('Please upload your sequences :')  
    reqs = st.expander('Example input file and sequence file format requirements 👉', expanded=False)
    with reqs : 
        st.write('* Every line in the file must contain just a single DNA sequence with no additional special characters, spaces, tabs or linebreaks.')
        st.write('* If there are more than 80 bases in a given sequence, the last 80 bases will be used for the downstream computations. If there are less than 80 bases in a given sequence, it will be padded with N bases after adding the constant flanking sequence in the beginning (as described in the paper) until each input sequence has the same length.')
        st.write('* You can download a sample input file below.')
        
        ### START : This block of code should be ignored by users to avoid confusion 
        df = pd.read_csv(path_prefix+'Random_testdata_complex_media.txt', sep = '\t' , header = None)
        sample_list = population_remove_flank(list(df.iloc[:,0].values)) # Removing the flanks is necessary here (see Methods section for details.)
        tmp_download_link = download_link("\n".join(sample_list), 'example_input_file.txt', 'Click here to download an example sequence input file')
        st.markdown(tmp_download_link, unsafe_allow_html=True)
        ### END : This block of code should be ignored by users to avoid confusion 
        
        
    reqs_seqgen = st.expander('Generate a random sequence 🧬', expanded=False)
    with reqs_seqgen : 
        #seqgen_button = st.button('Click here to run the a random sequence generator applet in your browser')
        #generate_button = st.button('Generate')
        args  = {'population_size' : int(1), 'sequence_length' : 80 , 'nucleotide_frequency' :[0.25,0.25,0.25,0.25] , 'randomizer' : np.random } 
        population  = population_generator_unflanked(args)
        st.session_state.random_sequence = population[0]
        st.write(st.session_state.random_sequence)

    if 0 :
        st.subheader('Upload the sequence file here👇')
        uploaded_file = st.file_uploader("")

        st.write('**OR**')

        st.subheader('Paste one sequence per line here👇')
        text_area = st.text_area(label='' , value = 'GAGGCATCGTTTTATCAGATGATAGTTTAATTAGTACGTGCAGCACCTTAAAGGATATAAGGGCCGGTAGAACATAACGC\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGACACTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGGTTGTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA')

        submit = st.button('Submit Sequences')



    cols = st.columns([0.95, 0.1 , 0.95])

    with cols[0] : 
        with st.container() : 
            st.subheader('Upload the sequence file here👇')
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
        with st.container() : 
            st.subheader('Paste one sequence per line here👇')
            text_area = st.text_area(label='' , value = 'GAGGCATCGTTTTATCAGATGATAGTTTAATTAGTACGTGCAGCACCTTAAAGGATATAAGGGCCGGTAGAACATAACGC\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGACACTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA\nGAGGCCACTGTAAATAATGGTCAGAAGTGTTGTTATGGTTGTTTGCAAGGGTGTCTCCCAGTGTAGCGCCTCTCGCCCTA')
        

    button_cols = st.columns([0.95, 0.1, 0.95])
    with button_cols[-1] :
        with st.container() : 
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
    start_session = 1

    #with st.spinner('Loading deep transformer neural network model ...'):
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
        model, scaler,batch_size,session = load_model(model_conditions)
        K.set_session(session)
    
    if mode=="Expression" :
        sequences = list(input_df.iloc[:,0].values)
        single_sequence_input = 0 
        if(len(sequences) == 1) :
            single_sequence_input = 1 
            sequences = sequences+sequences
        X,_ = parse_seqs(sequences)

        with st.spinner('Predicting expression from sequence using model...'):
            K.set_session(session)
            with fitness_function_graph.as_default() : 
                with session.as_default(): 
                    Y_pred= scaler.inverse_transform(model.predict(X,batch_size = 1024,verbose = 0)).flatten()
        
                #Y_pred = evaluate_model(X, model, scaler, batch_size ,session, fitness_function_graph)
        st.success('Expression prediction complete !')

        expression_output_df = pd.DataFrame([ sequences , Y_pred ]  ).transpose()
        expression_output_df.columns = ['Sequence' , 'Expression']
        expression_output_df['Expression (percentile)']  = expression_output_df['Expression'].rank(pct=1)

        if single_sequence_input==1 :
            expression_output_df = pd.DataFrame(expression_output_df.loc[0,:]).T
        
        with st.container() : 
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
            with open(path_prefix+'Random_testdata_complex_media.txt') as f:
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



    if mode=="Evolvability Vector"  or mode=="Mutational Robustness" or mode=="Visualize Sequences and Generate Trajectories"\
    or mode=="Interpretability : In Silico Mutagenesis (ISM) score":

        sequences = list(input_df.iloc[:,0].values)
        single_sequence_input = 0 
        if(len(sequences) == 1) :
                single_sequence_input = 1 
                sequences = sequences+sequences
        X , sequences_flanked = parse_seqs(sequences)
        
        if mode=="Visualize Sequences and Generate Trajectories": 
            #with fitness_function_graph.as_default() : 
            #    st.write(np.array(scaler.inverse_transform(model.predict(seq2feature(sequences_flanked)))))
            

            st.header('Visualizing expression effects of mutation')
            st.write('')




            vis_reqs = st.expander('How to use this interface 👉', expanded=False)
            with vis_reqs : 
                st.write('Please click on a single mutation that you wish to introduce to the starting sequence or any subsequent sequence in the trajectory you create. The solid line denotes the mutation with the maximum expression and the dotted line denotes the mutation with the minumum expression. The first sequence is used if multiple sequences are entered above.')
            
            if 0:
                print_reqs = st.expander('How to print the complete trajectory👉', expanded=False)
                with print_reqs : 
                    st.session_state.print_trajectory = st.button('Just click here at the end of your experiment')
            st.write('')
            cmap_range = st.selectbox( "Select the color range scheme for the heatmap", ('Absolute', 'Relative'))
            if (cmap_range != st.session_state.cmap_range) :
                reset_state()
                st.session_state.cmap_range = 'Relative'
            
            if st.session_state.print_trajectory == 1 :  
                st.subheader('Trajectory')
                for i in range(len(st.session_state.mutation_list)):
                    st.subheader(st.session_state.mutation_list[i])
                    s,tmp_download_link,maxima,minima,df,source,p = plot_el_visualization(population_add_flank([st.session_state.seq_list[i]]))
                    st.bokeh_chart(p)
                st.session_state.print_trajectory = 0
                st.header('Start a new trajectory')

                reset_state()
            
            ### Reset if new input is entered


            if (st.session_state.input !=sequences_flanked ): 
                reset_state()

                
            ####BLOCK : Better not to put inside function
            st.session_state.counter=st.session_state.counter+1

            #st.session_state.counter
            #st.session_state.event_result_list
            #st.session_state.seq_list
            #st.session_state.mutation_list
            
            if (st.session_state.event_result_list==[]) :
                st.session_state.seq_list = st.session_state.seq_list +[population_remove_flank([sequences_flanked[0]])[0]]
                st.session_state.input = sequences_flanked
                st.session_state.mutation_list = st.session_state.mutation_list +['Input']
                s,tmp_download_link,maxima,minima,df,source,p = plot_el_visualization(sequences_flanked)
                #st.session_state.p_list = st.session_state.p_list + [p]
                st.session_state.download_list = st.session_state.download_list + [tmp_download_link]

            else :
                s,tmp_download_link,maxima,minima,df,source,p = plot_el_visualization(population_add_flank([st.session_state.seq_list[-1]]))
                    


            source.selected.js_on_change(
                "indices",
                CustomJS(
                    args=dict(source=source),
                    code="""
                    document.dispatchEvent(
                        new CustomEvent("TestSelectEvent", {detail: {indices: cb_obj.indices}})
                    )
                """,
                ),
            )
            
            st.subheader(st.session_state.mutation_list[-1])

            event_result = streamlit_bokeh_events(
                events="TestSelectEvent",
                bokeh_plot=p,
                key="foo",
                debounce_time=100,
                refresh_on_update=True
            )
            st.markdown(tmp_download_link, unsafe_allow_html=True)

            if 0 : 
                extrema_cols = st.columns([1, 1])
                with extrema_cols[0]:
                    maxima
                with extrema_cols[1]:
                    minima
            
            st.session_state.event_result_list =  st.session_state.event_result_list + [event_result]

        #if st.session_state.seq_list!=[] : 
        #    sequences_flanked = population_add_flank([ st.session_state.seq_list[-1] ])
        #    s,tmp_download_link,maxima,minima,df,source,p = plot_el_visualization(sequences_flanked)

            if ((st.session_state.counter%2) ==0) and (event_result is not None) : 
                event_result = st.session_state.event_result_list[-1]
                # TestSelectEvent was thrown
                if "TestSelectEvent" in event_result:
                    #st.subheader("Selected Points' Pandas Stat summary")
                    #if st.session_state.seq_list == [] : 

                    indices = event_result["TestSelectEvent"].get("indices", [])
                    #st.table(df.iloc[indices].describe())
                    index_list = [int(re.split('(\d+)',str(i))[1])-1 for i in df.iloc[indices]['position'].values]
                    index_list.reverse() ### Because clicking order is stored as stack
                    mutation_list = [ df.loc[i,'position'] + df.loc[i,'base'] for i in df.iloc[indices].index]
                    mutation_list.reverse()
                    newbase_list = [ df.loc[i,'base'] for i in df.iloc[indices].index]
                    newbase_list.reverse()
                        
                    
                    mutation = 'Input'
                    for index,m,newbase in zip(index_list,mutation_list,newbase_list) : 
                        mutation = mutation + '->' + m
                        new_sequences_unflanked = copy.deepcopy([i for i in s])
                        new_sequences_unflanked[index] = newbase
                        #st.write(new_sequences_unflanked[index] )
                        new_sequences_unflanked = ''.join(new_sequences_unflanked)
                        new_sequences_flanked = population_add_flank([new_sequences_unflanked])
                    
                    st.session_state.seq_list = st.session_state.seq_list +[new_sequences_unflanked]
                    st.session_state.mutation_list = st.session_state.mutation_list+[st.session_state.mutation_list[-1]+'->'+m]
                    #st.session_state.p_list = st.session_state.p_list + [p]
                    st.session_state.download_list = st.session_state.download_list + [tmp_download_link]
                    ###Reload Page to get correct plot
                    #from streamlit.script_runner import StopException, RerunException
                    #RerunException()  
                    st.experimental_rerun()
                    ###Reload Page to get correct plot


            st.subheader('Sequences in trajectory')
            st.session_state.seq_list
            st.subheader('Mutations in trajectory')
            st.session_state.mutation_list
            #st.session_state.p_list

            

            
            #####ENDBLOCK

                
                
  

            if 0 : 
                s,tmp_download_link, maxima,minima,source,output,df,p = plot_el_visualization(new_sequences_flanked)
                p_tuple = (mutation,s,tmp_download_link, maxima,minima,source,output,df,p )
                p_list = p_list + [p_tuple]
                
                st.header("Results")
                #st.write(event_result)
                for p_tuple in p_list:

                    st.subheader(p_tuple[0])
                    st.write(p_tuple[-1])
                    extrema_cols = st.columns([1, 1])
                    with extrema_cols[0]:
                        p_tuple[3]
                    with extrema_cols[1]:
                        p_tuple[4]
                    st.markdown(p_tuple[2], unsafe_allow_html=True)

                if 0 : 
                    st.header("Aggregated export of SVGs")
                    q = []
                    for p_tuple in p_list:
                        q = q+[[p_tuple[-1]]]
                    from bokeh.layouts import gridplot
                    st.bokeh_chart(gridplot(q))
                


        if mode=="Evolvability Vector"  or mode=="Mutational Robustness" :
            with st.spinner('Computing expression from sequence using the model...'):
                with fitness_function_graph.as_default() : 
                    with session.as_default(): 
                        Y_pred  = np.array(scaler.inverse_transform(model.predict(X)))
                #Y_pred = evaluate_model(X, model, scaler, batch_size ,session, fitness_function_graph)

            evolvability_output_df = pd.DataFrame([ sequences , Y_pred ]  ).transpose()
            evolvability_output_df.columns = ['Sequence' , 'Expression']##
            
            if mode=="Evolvability Vector" : 
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
                with st.container() : 
                    st.header('Results')
                    evolvability_output_df
                    tmp_download_link = download_link(evolvability_output_df, 'evolvability_output_df.csv', 'Click here to download the results as a CSV')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)


        if mode=="Interpretability : In Silico Mutagenesis (ISM) score" :

            def get_ISM_score(population) : 
                with fitness_function_graph.as_default() : 
                    with session.as_default(): 
                        population_fitness= scaler.inverse_transform(model.predict(seq2feature(list(population)), verbose = 0)).flatten()
                        args  = {'sequence_length' : 80 , 'nucleotide_frequency' :[0.25,0.25,0.25,0.25] , 'randomizer' : np.random } 
                        population_1bp_all_sequences = population_mutator(list(population) , args)
                        population_1bp_all_fitness= np.array(scaler.inverse_transform(model.predict(seq2feature(list(population_1bp_all_sequences)),batch_size = 1024, verbose = 0))).flatten()

                snpdev_dist = []
                for i in (range(len(population))) :   
                    original_fitness = population_fitness[i]
                    sequence = population[i]

                    exp_dist =population_1bp_all_fitness[3*args['sequence_length']*i:3*args['sequence_length']*(i+1)] -  original_fitness 
                    
                    snpdev_dist = snpdev_dist + [np.add.reduceat(exp_dist, np.arange(0, len(exp_dist), 3))]

                sequences = population
                ISM_scores = snpdev_dist
                return ISM_scores

            ism_reqs = st.expander('More information on ISM scores 👉', expanded=False)
            with ism_reqs : 
                st.write('* The ISM scores are computed using a definition equivalent to this [gist](https://gist.github.com/AvantiShri/2d166f201716d8d019c979b32dc70767).')
                st.write('* For more information (and a faster implementation) of ISM scores, please check out this [paper](https://doi.org/10.1101/2020.10.13.337147).')
        
            ISM_output_df = pd.DataFrame()
            ISM_output_df[ 'Sequence' ] = sequences
  
            with st.spinner('Computing ISM scores for sequences...'):
                ISM_scores = get_ISM_score(sequences_flanked)
            st.success('ISM scores computed !')
            ISM_output_df['ISM scores (Position-wise)'] = ISM_scores
            
            if single_sequence_input==1 :
                ISM_output_df = pd.DataFrame(ISM_output_df.loc[0,:]).T
            if 1 : 
                with st.container() : 
                    st.header('Results')
                    ISM_output_df
                    tmp_download_link = download_link(ISM_output_df, 'ISM_output_df.csv', 'Click here to download the results as a CSV')
                    st.markdown(tmp_download_link, unsafe_allow_html=True)


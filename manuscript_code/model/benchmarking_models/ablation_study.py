import sys
sys.path.insert(0, './')

from rr_aux import *



##Clear Memory 
tf.reset_default_graph()
tf.keras.backend.clear_session()
gc.collect()
##


NUM_GPU = len(get_available_gpus())
if(NUM_GPU>0) :
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

print(tf.__version__)
print(keras.__version__)

model_conditions = 'Glu'


#sys.argv[1] appended to beginning of path
dir_path=os.path.join('..','..','..','data',model_conditions)


###
sys.argv = sys.argv[1:]

print(sys.argv)





## Load the data matrix
with h5py.File(join(dir_path,'_trX.h5'), 'r') as hf:
    _trX = hf['_trX'][:]

with h5py.File(join(dir_path,'_trY.h5'), 'r') as hf:
    _trY = hf['_trY'][:]

with h5py.File(join(dir_path,'_vaX.h5'), 'r') as hf:
    _vaX = hf['_vaX'][:]

with h5py.File(join(dir_path,'_vaY.h5'), 'r') as hf:
    _vaY = hf['_vaY'][:]

with h5py.File(join(dir_path,'_teX.h5'), 'r') as hf:
    _teX = hf['_teX'][:]

with h5py.File(join(dir_path,'_teY.h5'), 'r') as hf:
    _teY = hf['_teY'][:]


_trX.shape , _trY.shape , _vaX.shape , _vaY.shape  , _teX.shape , _teY.shape



trX = _trX #np.concatenate((_trX, _trX_rc), axis = 1) #np.squeeze((_trX))#
vaX = _vaX # np.concatenate((_vaX, _vaX_rc), axis = 1) #np.squeeze((_vaX))#
teX = _teX # np.concatenate((_teX, _teX_rc), axis = 1)#np.squeeze((_teX))#



## Load the scaler function (scaler was trained on the synthesized data
scaler = sklearn.externals.joblib.load(join(dir_path,'scaler.save')) 




vaY = (scaler.transform(_vaY.reshape(1, -1))).reshape(_vaY.shape) #_vaY#
trY = (scaler.transform(_trY.reshape(1, -1))).reshape(_trY.shape) #_trY#
teY = (scaler.transform(_teY.reshape(1, -1))).reshape(_teY.shape) #_teY#


### If using generator, have a smaller val set for faster evaluation
if 0: 
    s_trX = np.vstack((trX , vaX))
    s_trY = np.vstack((trY , vaY))
    trX = s_trX[1000:,:]
    trY = s_trY[1000:,:]
    vaX = s_trX[0:1000,:]
    vaY = s_trY[0:1000,:]

print(trX.shape , trY.shape , vaX.shape , vaY.shape  , _teX.shape , _teY.shape)

input_shape = trX.shape


def fitness_function_model(model_params) :

    n_val_epoch = model_params['n_val_epoch']
    epochs= model_params['epochs']
    batch_size= model_params['batch_size']
    l1_weight= model_params['l1_weight']
    l2_weight= model_params['l2_weight']
    motif_conv_hidden= model_params['motif_conv_hidden']
    conv_hidden= model_params['conv_hidden']
    n_hidden= model_params['n_hidden']
    n_heads= model_params['n_heads']
    conv_width_motif= model_params['conv_width_motif']
    dropout_rate= model_params['dropout_rate']
    attention_dropout_rate= model_params['attention_dropout_rate']
    lr= model_params['lr']
    n_aux_layers= model_params['n_aux_layers']
    n_attention_layers= model_params['n_attention_layers']
    add_cooperativity_layer= model_params['add_cooperativity_layer']
    device_type = model_params['device_type']
    input_shape = model_params['input_shape']
    loss = model_params['loss']
    padding = model_params['padding']
    ablated_layer = model_params['ablated_layer']
    
    if(model_params['device_type']=='tpu'):
        input_layer = Input(batch_shape=(batch_size,input_shape[1],input_shape[2]))  #trX.shape[1:] #batch_shape=(batch_size,110,4)

    else :
        input_layer = Input(shape=input_shape[1:])  #trX.shape[1:] #


    #https://arxiv.org/pdf/1801.05134.pdf

    if (ablated_layer == 'conv1') :
        x_f = input_layer
        x_rc = input_layer
        
    else :
        x_f,x_rc = rc_Conv1D(motif_conv_hidden, conv_width_motif, padding=padding , \
                   kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight), kernel_initializer='he_normal' ,
                  data_format = 'channels_last' , use_bias=False)(input_layer)
        x_f = BatchNormalization()(x_f)
        x_rc = BatchNormalization()(x_rc)

        x_f = Activation('relu')(x_f)
        x_rc = Activation('relu')(x_rc)


    if (ablated_layer == 'conv2') :
        add_cooperativity_layer = False
    if(add_cooperativity_layer==True) : 
        x_f = Lambda(lambda x : K.expand_dims(x,axis=1))(x_f)
        x_rc = Lambda(lambda x : K.expand_dims(x,axis=1))(x_rc)

        x =Concatenate(axis=1)([x_f, x_rc] )

        x = keras.layers.ZeroPadding2D(padding = ((0,0 ),(int(conv_width_motif/2)-1,int(conv_width_motif/2))), 
                                          data_format = 'channels_last')(x)
        x = Conv2D(conv_hidden, (2,conv_width_motif), padding='valid' ,\
               kernel_regularizer  = l1_l2(l1=l1_weight, l2=l2_weight), kernel_initializer='he_normal' ,
              data_format = 'channels_last' , use_bias=False)(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Lambda(lambda x : K.squeeze(x,axis=1))(x)
        


    else:
        x =Add()([x_f, x_rc] )
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x)

    if (ablated_layer == 'conv3') :
        n_aux_layers = 0
    for i in range(n_aux_layers) : 
        #res_input = x
        x = Conv1D(conv_hidden, (conv_width_motif), padding=padding ,\
               kernel_regularizer  = l1_l2(l1=l1_weight, l2=l2_weight), kernel_initializer='he_normal' ,
              data_format = 'channels_last' , use_bias=False)(x) 
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        #x = Add()([res_input, x])


    if (ablated_layer == 'transformer') :
        n_attention_layers = 0    
    for i in range(n_attention_layers) : 
        mha_input = x
        x = MultiHeadAttention( head_num=n_heads,name='Multi-Head'+str(i),
                              kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight))(x) #### DO NOT MAX POOL or AVG POOL 
        if dropout_rate > 0.0:
            x = Dropout(rate=attention_dropout_rate)(x)
        else:
            x = x
        x = Add()([mha_input, x])
        x = LayerNormalization()(x)
        
        ff_input = x
        x  = FeedForward(units= n_heads, kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight))(x)
        if dropout_rate > 0.0:
            x = Dropout(rate=attention_dropout_rate)(x)
        else:
            x = x
        x = Add()([ff_input, x])
        x = LayerNormalization()(x)    


    if (ablated_layer != 'lstm') :
        x = Bidirectional(LSTM(n_heads, return_sequences=True,
                               kernel_regularizer  = l1_l2(l1=l1_weight, l2=l2_weight),
                               kernel_initializer='he_normal' , dropout = dropout_rate))(x)
        x = Dropout(dropout_rate)(x)


    if(len(x.get_shape())>2):
        x = Flatten()(x) 

        
    if (ablated_layer != 'dense') :

        x = Dense(int(n_hidden), 
                        kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight),
                        kernel_initializer='he_normal' , use_bias=True)(x)
        x = Activation('relu')(x) 
        x = Dropout(dropout_rate)(x) #https://arxiv.org/pdf/1801.05134.pdf


        x = Dense(int(n_hidden), kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight),
                        kernel_initializer='he_normal', use_bias=True )(x)
        x = Activation('relu')(x)
        x = Dropout(dropout_rate)(x) #https://arxiv.org/pdf/1801.05134.pdf

        
    output_layer = Dense(1, kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight),
                    activation='linear', kernel_initializer='he_normal', use_bias=True )(x) 


    model = Model(input_layer, output_layer)
    opt = tf.train.RMSPropOptimizer(lr) #tf.keras.optimizers.Adam(lr=lr)#
    

    model.compile(optimizer=opt, loss=loss,metrics=['mean_squared_error', 'cosine_similarity']) 
    
    return model


ablated_layer_list = sys.argv#['conv1' , 'conv2' , 'conv3' , 'transformer' , 'lstm' , 'dense' , 'None']

for ablated_layer in ablated_layer_list : 
    model_params = {
        'n_val_epoch' : 1000,
        'epochs' : 1,
        'batch_size': int(1024*1), # int(1024*3) , #64*55 fits , #best batch size is 1024
        'l1_weight': 0,#1e-6#1e-7#0.01 # l1 should always be zero
        'l2_weight': 0,#1e-7#0.01
        'motif_conv_hidden': 256,
        'conv_hidden': 64,
        'n_hidden': 64, #128
        'n_heads': 8,
        'conv_width_motif':30, ##30bp for yeast is required for capturing all motifs 
        'dropout_rate': 0.05,
        'lr':0.001,
        'add_cooperativity_layer': True,
        'n_aux_layers': 1,
        'n_attention_layers':2,
        'attention_dropout_rate' : 0,
        'device_type' : 'gpu', #'tpu'/'gpu'/'cpu'
        'input_shape' : input_shape,
        'loss' : 'mean_squared_error',
        'padding' : 'same',
        'ablated_layer' : ablated_layer }
    epochs = model_params['epochs']  
    batch_size =  model_params['batch_size']
    n_val_epoch =  model_params['n_val_epoch']
    epochs = model_params['epochs']



    test_variable = ablated_layer
    ### Save model params as csv
    w = csv.writer(open((test_variable+'_model_params.csv'), "w"))
    for key, val in model_params.items():
        w.writerow([key, val])

    ### Save model params as pickle
    f = open((test_variable+'_model_params.pkl'),"wb")
    pickle.dump(model_params,f)
    f.close()


    model=fitness_function_model(model_params)

    print(model.summary())

    model.fit(trX, trY, validation_data = (vaX[:100], vaY[:100]), batch_size=batch_size  , epochs=epochs  )
    #model.fit_generator(training_generator, validation_data = (teX[:100], teY[:100]),
    #epochs=epochs , steps_per_epoch = int(trX.shape[0]/(batch_size*n_val_epoch)) )


    def read_hq_testdata(filename) :

        with open(filename) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)

        sequences = [di[0] for di in d]

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
        print(seqdata_transformed.shape)


        expressions = [di[1] for di in d]
        expdata = np.asarray(expressions)
        expdata = expdata.astype('float')  

        return np.squeeze(seqdata_transformed),expdata

    X,Y = read_hq_testdata(os.path.join('..','..','..','data','Glu','HQ_testdata.txt'))
    Y = [float(x) for x in Y]
    Y_pred= model.predict(X, batch_size = 1024)
    Y_pred = [float(i[0]) for i in Y_pred]

    pcc = scipy.stats.pearsonr(Y,Y_pred )[0]
    print(pcc)

    df = pd.DataFrame({'Y' : Y , 'Y_pred' : Y_pred , 'pcc' : pcc})
    df.to_csv(ablated_layer+"_results.csv")
    model.save(ablated_layer+"_model.h5")



    def read_hq_testdata(filename) :

        with open(filename) as f:
            reader = csv.reader(f, delimiter="\t")
            d = list(reader)

        sequences = [di[0] for di in d]

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
        print(seqdata_transformed.shape)


        expressions = [di[1] for di in d]
        expdata = np.asarray(expressions)
        expdata = expdata.astype('float')  

        return np.squeeze(seqdata_transformed),expdata

    X,Y = read_hq_testdata(os.path.join('..','..','..','data','Glu','HQ_testdata.txt'))
    Y = [float(x) for x in Y]
    Y_pred= model.predict(X, batch_size = 1024)
    Y_pred = [float(i[0]) for i in Y_pred]

    pcc = scipy.stats.pearsonr(Y,Y_pred )[0]
    print(pcc)

    df = pd.DataFrame({'Y' : Y , 'Y_pred' : Y_pred , 'pcc' : pcc})
    df.to_csv(ablated_layer+"_results.csv")
    model.save(ablated_layer+"_model.h5")
    ##Clear Memory 
    tf.reset_default_graph()
    tf.keras.backend.clear_session()
    gc.collect()
    ##

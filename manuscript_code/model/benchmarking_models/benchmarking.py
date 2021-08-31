import BioinforDeepATT
from BioinforDeepATT.model.model import *
import sys
sys.path.insert(0, './')

from rr_aux import *


##Clear Memory 
#tf.reset_default_graph()
tf.keras.backend.clear_session()
gc.collect()
##

### Change this to 1 if you would like to save a new model.
save = 0

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


if 1: 


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



    trX = _trX[:int(_trX.shape[0]/1024)*1024] #np.concatenate((_trX, _trX_rc), axis = 1) #np.squeeze((_trX))#
    vaX = _vaX[:int(_vaX.shape[0]/1024)*1024] # np.concatenate((_vaX, _vaX_rc), axis = 1) #np.squeeze((_vaX))#
    teX = _teX[:int(_teX.shape[0]/1024)*1024] # np.concatenate((_teX, _teX_rc), axis = 1)#np.squeeze((_teX))#



    ## Load the scaler function (scaler was trained on the synthesized data
    scaler = sklearn.externals.joblib.load(join(dir_path,'scaler.save')) 




    vaY = (scaler.transform(_vaY.reshape(1, -1))).reshape(_vaY.shape)[:int(_vaY.shape[0]/1024)*1024] #_vaY#
    trY = (scaler.transform(_trY.reshape(1, -1))).reshape(_trY.shape)[:int(_trY.shape[0]/1024)*1024] #_trY#
    teY = (scaler.transform(_teY.reshape(1, -1))).reshape(_teY.shape)[:int(_teY.shape[0]/1024)*1024] #_teY#

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





import BioinforDeepATT
from BioinforDeepATT.model.model import *
define_DeepAtt_here = 1 

if define_DeepAtt_here == 1 :
    del Bidirectional, LSTM
    
from rr_aux import * ### Overrides DeepAtt() imports

def fitness_function_model(model_params) :

    batch_size= model_params['batch_size']
    l1_weight= model_params['l1_weight']
    l2_weight= model_params['l2_weight']
    lr= model_params['lr']
    device_type = model_params['device_type']
    input_shape = model_params['input_shape']
    loss = model_params['loss']
    model_name = model_params['model_name']
    
    if(model_params['device_type']=='tpu'):
        input_layer = Input(batch_shape=(batch_size,input_shape[1],input_shape[2]))  #trX.shape[1:] #batch_shape=(batch_size,110,4)

    else :
        input_layer = Input(shape = input_shape[1:] , batch_size = 1024)  #trX.shape[1:] #

    if model_name=='DeepAtt':
        if define_DeepAtt_here : 
            x = Conv1D(256, 30, padding='valid' ,\
                   kernel_regularizer  = l1_l2(l1=l1_weight, l2=l2_weight), kernel_initializer='he_normal' ,
                  data_format = 'channels_last' , activation='relu')(input_layer) 
            x = tf.keras.layers.MaxPool1D( pool_size=3, strides=3, padding='valid')(x)
            x= tf.keras.layers.Dropout(0.2)(x)
            x = Bidirectional(LSTM(16, return_sequences=True,kernel_initializer='he_normal'))(x)
            x = MultiHeadAttention( head_num=16)(x) 
            x = tf.keras.layers.Dropout(0.2)(x)
            x = Flatten()(x)

            x = keras.layers.Dense(
                    units=16,
                    activation='relu')(x)
            x = keras.layers.Dense(
                units=16,
                activation='relu')(x)
        else : 
            x = DeepAtt()
            x = x.call(input_layer)

    if model_name=='DanQ':
        x = DanQ()
        x = x.call(input_layer)

    if model_name=='DeepSEA':
        x = DeepSEA()
        x = x.call(input_layer)

    if(len(x.get_shape())>2):
        x = Flatten()(x) 
        
    output_layer = Dense(1, kernel_regularizer = l1_l2(l1=l1_weight, l2=l2_weight),
                    activation='linear', kernel_initializer='he_normal', use_bias=True )(x) 


    model = Model(input_layer, output_layer)
    opt = tf.compat.v1.train.AdamOptimizer(lr) #tf.keras.optimizers.Adam(lr=lr)#
    

    model.compile(optimizer=opt, loss=loss,metrics=['mean_squared_error', 'cosine_similarity']) 
    
    return model

model_name_list = sys.argv[1:]#['DeepAtt' , 'DeepSEA' , 'DanQ' ]
for model_name in model_name_list : 
    model_params = {
        'n_val_epoch' : 1000,
        'epochs' : 3, #3 used previously
        'batch_size': int(1024*1), # int(1024*3) , #64*55 fits , #best batch size is 1024
        'l1_weight': 0,#1e-6#1e-7#0.01 # l1 should always be zero
        'l2_weight': 0,#1e-7#0.01
        'lr':0.001,
        'device_type' : 'gpu', #'tpu'/'gpu'/'cpu'
        'input_shape' : input_shape,
        'loss' : 'mean_squared_error', 
        'model_name' : model_name}
    epochs = model_params['epochs']  
    batch_size =  model_params['batch_size']
    n_val_epoch =  model_params['n_val_epoch']
    epochs = model_params['epochs']



    if save :  ### Change to 1 if you would like to update the saved parameter file
        ### Save model params as csv
        w = csv.writer(open(os.path.join(model_name+'_model_params.csv'), "w"))
        for key, val in model_params.items():
            w.writerow([key, val])

        ### Save model params as pickle
        f = open(os.path.join(model_name+'_model_params.pkl'),"wb")
        pickle.dump(model_params,f)
        f.close()


    model=fitness_function_model(model_params)

    print(model.summary())

    model.fit(trX, trY, validation_data = (vaX[:1024], vaY[:1024]), batch_size=batch_size  , epochs=epochs  )
    #model.fit_generator(training_generator, validation_data = (teX[:100], teY[:100]),
    #epochs=epochs , steps_per_epoch = 10240)#int(trX.shape[0]/(batch_size*n_val_epoch)) )
    model.save_weights(model_name)

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
    Y_pred= evaluate_model(X, model, scaler)

    pcc = scipy.stats.pearsonr(Y,Y_pred )[0]
    print(pcc)
    
    df = pd.DataFrame({'Y' : Y , 'Y_pred' : Y_pred , 'pcc' : pcc})
    
    if save : ### Please change this to 1 if you would like to save the new results
        df.to_csv(model_name+"_results.csv")
        model.save(model_name+"_model.h5")
        model.save_weights(model_name)

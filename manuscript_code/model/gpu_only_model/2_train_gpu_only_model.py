#!/usr/bin/env python
# coding: utf-8

# Please use the transformer model for your future applications instead of the convolutional model in this directory. 
# The transformer model is available at: https://github.com/1edv/evolution/tree/master/manuscript_code/model/tpu_model
# For more details on why we recommend the transformer model instead of the model described here, please see: https://github.com/1edv/evolution/tree/master/manuscript_code/model/gpu_only_model#please-use-the-transformer-model-for-your-future-applications-instead-of-the-convolutional-model-in-this-directory

# ## This notebook allows the user to train their own version of the GPU model from scratch
# - This notebook can also be run using the `2_train_gpu_model.py` file in this folder. 
# - If you are a Reviewer using this with the Google Cloud VM we shared with you, running this on the command line in a separate terminal should automatically attach a GPU for the training.
# 
# 
# #### Notes
# - The training data for training the GPU model uses a separate file format. We have also uploaded an example training data ( the one we used for the complex media condition) in this format here so this training notebook will be fully functional on the Google Cloud VM and on CodeOcean(the file can be found in this directory and is used here below). As we have shown in the manuscript, the complex and defined media have highly correlated expression levels and doing the same for defined media will lead to equivalent prediction performance of the trained models.
# 
# - <b>Also, please note the PCC metric shown on the 'validation set' is not any of the test data we use in the paper. It is simply a held-out sample of the training data experiment as we explain elsewhere as well. This training data is significantly higher complexity and hence lead to a much lower number of 'read replicates' per given sequence. So we carry out separate experiments low-complexity library experiments to measure the test data.</b> 
# 
# - We verified that training the model works on this machine.

# ### Pre-process the training data for the GPU model

# In[1]:


import csv
import copy
import numpy as np
import multiprocessing as mp, ctypes
import time , csv ,pickle ,joblib , matplotlib  , multiprocessing,itertools
from joblib import Parallel, delayed 
from tqdm import tqdm

import argparse,pwd,os,numpy as np,h5py
from os.path import splitext,exists,dirname,join,basename
from os import makedirs
import matplotlib.pyplot as plt
import h5py
import tensorflow as tf, sys, numpy as np, h5py, pandas as pd
from tensorflow import nn
from tensorflow.contrib import rnn
from os.path import join,dirname,basename,exists,realpath
from os import makedirs
from tensorflow.examples.tutorials.mnist import input_data
import sklearn , scipy
from sklearn.metrics import *
from scipy.stats import *
import time
import os 
from tqdm import tqdm
import datetime
from datetime import datetime


# In[2]:



    
################################################Final one used
###GET ONE HOT CODE FROM SEQUENCES , parallel code, quite fast  
class OHCSeq:
    transformed = None
    data = None


def seq2feature(data):
    num_cores = multiprocessing.cpu_count()-2
    nproc = np.min([16,num_cores])
    OHCSeq.data=data
    shared_array_base = mp.Array(ctypes.c_bool, len(data)*len(data[0])*4)
    shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
    shared_array = shared_array.reshape(len(data),1,len(data[0]),4)
    #OHCSeq.transformed = np.zeros([len(data),len(data[0]),4] , dtype=np.bool )
    OHCSeq.transformed = shared_array


    pool = mp.Pool(nproc)
    r = pool.map(seq2feature_fill, range(len(data)))
    pool.close()
    pool.join()
    #myOHC.clear()
    return( OHCSeq.transformed)





def seq2feature_fill(i):
    mapper = {'A':0,'C':1,'G':2,'T':3,'N':None}
    ###Make sure the length is 110bp
    if (len(OHCSeq.data[i]) > 110) :
        OHCSeq.data[i] = OHCSeq.data[i][-110:]
    elif (len(OHCSeq.data[i]) < 110) : 
        while (len(OHCSeq.data[i]) < 110) :
            OHCSeq.data[i] = 'N'+OHCSeq.data[i]
    for j in range(len(OHCSeq.data[i])):
        OHCSeq.transformed[i][0][j][mapper[OHCSeq.data[i][j]]]=True 
    return i

########GET ONE HOT CODE FROM SEQUENCES , parallel code, quite fast  
################################################################


# ### Load the training data file

# In[3]:


model_conditions = 'defined' #'complex' or 'defined' or 'user_defined'
with open('./'+model_conditions+'_media_training_data.txt') as f: #replace with the path to your raw data if 'user_defined'
    reader = csv.reader(f, delimiter="\t")
    d = list(reader)	


# ### Extract the sequences and appropriately attach the constant flanks

# In[4]:


sequences = [di[0] for di in d]

### Append N's if the sequencing output has a length different from 17+80+13 (80bp with constant flanks)
for i in tqdm(range(0,len(sequences))) : 
    if (len(sequences[i]) > 110) :
        sequences[i] = sequences[i][-110:]
    if (len(sequences[i]) < 110) : 
        while (len(sequences[i]) < 110) :
            sequences[i] = 'N'+sequences[i]
            


# ### Convert the sequences to one hot code

# In[5]:


onehot_sequences = seq2feature(np.asarray(sequences))


# ### Get the reverse complement of the sequence
# Improved this implementation to make it faster for the readers to run in a single notebook 

# In[ ]:


tab = str.maketrans("ACTGN", "TGACN")

def reverse_complement_table(seq):
    return seq.translate(tab)[::-1]

rc_sequences = [reverse_complement_table(seq) for seq in tqdm(sequences)]

rc_onehot_sequences = seq2feature(np.array(rc_sequences))


# ### Extract the expression corresponding to the sequences

# In[ ]:


expressions = [di[1] for di in d]
expressions = np.asarray(expressions)
expressions = expressions.astype('float')  
expressions = np.reshape(expressions , [-1,1])


# ### Split training data into two groups _trX and _vaX but please note that this is not the test data !
# 

# In[ ]:




total_seqs = len(onehot_sequences)
_trX = onehot_sequences[int(total_seqs/10):]
_trX_rc = rc_onehot_sequences[int(total_seqs/10):]
_trY = expressions[int(total_seqs/10):]

_vaX = onehot_sequences[0:int(total_seqs/10)]
_vaX_rc = rc_onehot_sequences[0:int(total_seqs/10)]
_vaY = expressions[0:int(total_seqs/10)]





# ### Define hyperparameters and specify location for saving the model
# We have saved an example for training to test the file in the user models folder

# In[ ]:


##MODEL FILE SAVING ADDRESSES AND MINIBATCH SIZES

# Training
best_dropout = 0.8  
best_l2_coef = 0.0001 
best_lr = 0.0005     


_batch_size = 1024
_hyper_train_size = 2000
#_valid_size = 1024
_hidden = 256
_epochs = 5
_best_model_file = join('models_conditions',model_conditions+'_media','best_model.ckpt')
_best_model_file_hyper = join('models_conditions',model_conditions+'_media','hyper_search', 'best_model.ckpt')
for _file  in [_best_model_file, _best_model_file_hyper]:
    if not exists(dirname(_file)):
        makedirs(dirname(_file))


# ### Define Model Architecutre

# In[ ]:



####################################################################
####################################################################
### MODEL ARCHITECTURE
####################################################################
####################################################################
def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, stride=1):
    return tf.nn.conv2d(x, W, strides=[1, stride, stride, 1], padding='SAME')

def max_pool(x, stride=2, filter_size=2):
    return tf.nn.max_pool(x, ksize=[1, 1, 2, 1],
                        strides=[1, 1, 2, 1], padding='SAME')

def cross_entropy(y, y_real):
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = y, labels = y_real))

def build_two_fc_layers(x_inp, Ws, bs):
    h_fc1 = tf.nn.relu(tf.matmul(x_inp, Ws[0]) + bs[0])
    return tf.matmul(h_fc1, Ws[1]) + bs[1]



            
            


def cnn_model(X, hyper_params , scope):

    with tf.variable_scope(scope) : 
        global _hidden 
        conv1_filter_dim1 = 30
        conv1_filter_dim2 = 4
        conv1_depth = _hidden

        conv2_filter_dim1 = 30
        conv2_filter_dim2 = 1
        conv2_depth = _hidden








        W_conv1 = weight_variable([1,conv1_filter_dim1,conv1_filter_dim2,conv1_depth])
        conv1 = conv2d(X, W_conv1)    
        conv1 = tf.nn.bias_add(conv1, bias_variable([conv1_depth]))
        conv1 = tf.nn.relu(conv1)
        l_conv = conv1
        
        W_conv2 = weight_variable([conv2_filter_dim1,conv2_filter_dim2,conv1_depth, conv2_depth])
        conv2 = conv2d(conv1,W_conv2 )
        conv2 = tf.nn.bias_add(conv2, bias_variable([conv2_depth]))
        conv2 = tf.nn.relu(conv2)

        
        regularization_term = hyper_params['l2']* tf.reduce_mean(tf.abs(W_conv1)) + hyper_params['l2']* tf.reduce_mean(tf.abs(W_conv2)) 
        
        cnn_model_output = conv2

    return cnn_model_output , regularization_term 






def training(trX, trX_rc, trY, valX, valX_rc, valY, hyper_params, epochs, batch_size, best_model_file): 

    
    tf.reset_default_graph()
    
    global _hidden 

    lstm_num_hidden = _hidden
    fc_num_hidden = _hidden
    num_classes = 1
    num_bins = 256
    
    conv3_filter_dim1 = 30
    conv3_filter_dim2 = 1
    conv3_depth = _hidden
    
    conv4_filter_dim1 = 30
    conv4_filter_dim2 = 1
    conv4_depth = _hidden
        # Input and output

    X = tf.placeholder("float", [None, 1, 110, 4] )
    X_rc = tf.placeholder("float", [None, 1, 110, 4] )
    Y = tf.placeholder("float", [None,1] )
    dropout_keep_probability = tf.placeholder_with_default(1.0, shape=())


    #f is forward sequence 
    output_f , regularization_term_f =  cnn_model(X, {'dropout_keep':hyper_params['dropout_keep'],'l2':hyper_params['l2']} , "f")

    #rc is reverse complement of that sequence
    output_rc , regularization_term_rc =  cnn_model(X_rc, {'dropout_keep':hyper_params['dropout_keep'],'l2':hyper_params['l2']} , "rc")
    
    
    ### CONCATENATE output_f and output_rc
    concatenated_f_rc = tf.concat([output_f , output_rc], -1)
    ###
    
    W_conv3 = weight_variable([conv3_filter_dim1,conv3_filter_dim2,2*_hidden,conv3_depth])
    conv3 = conv2d(concatenated_f_rc,W_conv3 )
    conv3 = tf.nn.bias_add(conv3, bias_variable([conv3_depth]))
    conv3 = tf.nn.relu(conv3)

    W_conv4 = weight_variable([conv4_filter_dim1,conv4_filter_dim2,conv3_depth,conv4_depth])
    conv4 = conv2d(conv3,W_conv4 )
    conv4 = tf.nn.bias_add(conv4, bias_variable([conv4_depth]))
    conv4 = tf.nn.relu(conv4)


    conv_feat_map_x = 110   
    conv_feat_map_y =  1   
    h_conv_flat = tf.reshape(conv4, [-1, conv_feat_map_x * conv_feat_map_y * lstm_num_hidden])
    #FC-1
    
    W_fc1 = weight_variable([conv_feat_map_x * conv_feat_map_y * lstm_num_hidden , fc_num_hidden])
    b_fc1 = bias_variable([fc_num_hidden])
    h_fc1 = tf.nn.relu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)
    #Dropout for FC-1
    h_fc1 = tf.nn.dropout(h_fc1, dropout_keep_probability)

    
    #FC-2
    W_fc2 = weight_variable([fc_num_hidden , num_bins])
    b_fc2 = bias_variable([num_bins])
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
    #Dropout for FC-2
    h_fc2 = tf.nn.dropout(h_fc2, dropout_keep_probability)


    #FC-3
    W_fc3 = weight_variable([num_bins, num_classes])
    b_fc3 = bias_variable([num_classes])
    h_fc3 = tf.matmul(h_fc2, W_fc3) + b_fc3 

    regularization_term = hyper_params['l2']* tf.reduce_mean(tf.abs(W_fc3)) + hyper_params['l2']* tf.reduce_mean(tf.abs(W_fc2)) + hyper_params['l2']* tf.reduce_mean(tf.abs(W_fc1)) + hyper_params['l2']* tf.reduce_mean(tf.abs(W_conv3))+ regularization_term_f + regularization_term_rc +hyper_params['l2']* tf.reduce_mean(tf.abs(W_conv4))



    with tf.variable_scope("out") :
        output = h_fc3
        model_output = tf.identity(output, name="model_output")

        ##########

        loss = tf.losses.mean_squared_error( Y , model_output ) + regularization_term
        cost = loss 
        model_cost = tf.identity(cost, name="model_cost")
        ##########
        pcc = tf.contrib.metrics.streaming_pearson_correlation(model_output,Y)
        model_pcc = tf.identity(pcc, name="model_pcc")
        ##########
        mse = tf.losses.mean_squared_error( Y , model_output )
        model_mse = tf.identity(mse, name="model_mse")
        ##########
        total_error = tf.reduce_sum(tf.square(tf.subtract(Y, tf.reduce_mean(Y))))
        unexplained_error = tf.reduce_sum(tf.square(tf.subtract(Y, model_output)))
        R_squared = tf.subtract(tf.constant(
    1,
    dtype=tf.float32), tf.div(unexplained_error, total_error))
        model_R_squared = tf.identity(R_squared, name="model_R_squared")

        ##########

        


    tf.summary.scalar("cost", model_cost)
    tf.summary.scalar("pcc", model_pcc[0])
    tf.summary.scalar("mse", model_mse)
    tf.summary.scalar("R_squared", R_squared)

    summary_op = tf.summary.merge_all()
    
    train_op = tf.train.AdamOptimizer(hyper_params['lr']).minimize(cost)


    start = time.time()
    best_cost = float("inf") 
    best_r2 = float(0) 

    
    batches_per_epoch = int(len(trX)/batch_size)
    num_steps = int(epochs * batches_per_epoch)

    sess = tf.Session()
    init=tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    #init = tf.global_variables_initializer()
    sess.run(init)
    
    #clear the logs directory
    now = datetime.now()
    #writer = tf.summary.FileWriter(join('user_models','logs' , now.strftime("%Y%m%d-%H%M%S") ), sess.graph)

    print('Initializing variables...')

    epoch_loss = 0
    epoch_pcc = 0
    epoch_mse = 0
    epoch_R_squared = 0

    
   
    
    
    
    for step in tqdm(range(num_steps)):
    
    
    
        offset = (step * batch_size) % (trX.shape[0] - batch_size)
        batch_x = trX[offset:(offset + batch_size), :]
        batch_x_rc = trX_rc[offset:(offset + batch_size), :]
        batch_y = trY[offset:(offset + batch_size)]

        
        feed_dict = {X: batch_x, X_rc: batch_x_rc ,  Y: batch_y , dropout_keep_probability : hyper_params['dropout_keep'] }
        
        _, batch_loss , batch_pcc , batch_mse, batch_R_squared , summary = sess.run([train_op, cost , pcc , mse, R_squared , summary_op], feed_dict=feed_dict)
        
        batch_R_squared = batch_pcc[0]**2
        epoch_loss += batch_loss
        epoch_pcc += batch_pcc[0]
        epoch_mse += batch_mse        
        epoch_R_squared += batch_R_squared    
        
        #writer.add_summary(summary, step)

        
        
        if ( (step % batches_per_epoch == 0) and step/batches_per_epoch!=0):

            epoch_loss /= batches_per_epoch
            epoch_pcc /= batches_per_epoch
            epoch_mse /= batches_per_epoch
            epoch_R_squared /= batches_per_epoch
            
            print('')
            print( '')
            print( '')
            print( '')
            print( 'Training - Avg batch loss at epoch %d: %f' % (step/batches_per_epoch, epoch_loss))
            print( 'Training - PCC : %f' % epoch_pcc)
            print( 'Training - MSE : %f' % epoch_mse)
            print( 'Training - R_squared : %f' % epoch_pcc**2)
            
            epoch_loss = 0
            epoch_pcc = 0
            epoch_mse = 0
            epoch_R_squared = 0  
            
            #Randomized validation subset start
            randomize  =  np.random.permutation(len(valX))
            vaX = valX[randomize,:]
            vaX_rc = valX_rc[randomize,:]
            vaY = valY[randomize,:]
            
            #valX = vaX[0:valid_size,:] 
            #valX_rc = vaX_rc[0:valid_size,:] 
            #valY = vaY[0:valid_size,:]            
            
            #with tf.device('/cpu:0'):
            #validation_cost , validation_acc , summary = sess.run([cost , accuracy , summary_op], feed_dict={X: valX , X_rc: valX_rc, Y: valY})
            
            #### teX_output contains TESTED SEQUENCES FOR VALIDATION SET
            va_batch_size = 1024
            (q,r) = divmod(vaX.shape[0] , va_batch_size)
            i=0
            vaX_output = []
            while(i <= q ) :
                if(i< q  ) :
                    temp_result_step1=sess.run([model_output], feed_dict={X: vaX[va_batch_size*i:va_batch_size*i+va_batch_size,:], X_rc: vaX_rc[va_batch_size*i:va_batch_size*i+va_batch_size,:]  ,Y: vaY[va_batch_size*i:va_batch_size*i+va_batch_size,:]}) 
                    temp_result_step2=[float(x) for x in temp_result_step1[0]]
                    #print temp_result_step2

                    vaX_output = vaX_output + temp_result_step2
                    i = i+1

                elif (i==q) :
                    temp_result_step1 = sess.run([model_output], feed_dict={X: vaX[va_batch_size*i:,:], X_rc: vaX_rc[va_batch_size*i:,:]  ,Y: vaY[va_batch_size*i:,:]})
                    temp_result_step2=[float(x) for x in temp_result_step1[0]]
                    #print "here"
                    vaX_output = vaX_output + temp_result_step2
                    i = i+1

            #### RETURN TESTED SEQUENCES FOR VALIDATION SET
            vaY = [float(x) for x in vaY]
            validation_mse = sklearn.metrics.mean_squared_error(vaY , vaX_output )

            validation_pcc = scipy.stats.pearsonr(vaY , vaX_output )
            validation_r2  = validation_pcc[0]**2
        
        
        
            #for tensorboard
            print('')
            print( 'Full Validation Set - MSE : %f' % validation_mse)
            print( 'Full Validation Set - PCC : %f' % validation_pcc[0])
            print( 'Full Validation Set - R_squared : %f' % validation_r2)

            if(best_r2 < validation_r2) :
                #
                
                #SAVER 
                saver = tf.train.Saver()               
                saver.save(sess, "%s"%best_model_file )

                #
                best_loss = validation_mse
                best_cost = validation_mse
                best_r2 = validation_r2

                
    print( "Training time: ", time.time() - start)
    

    return best_r2


# ### Train the model. 
# Note that the model autosaves in the path defined above

# In[ ]:




print('\n', training(_trX,_trX_rc, _trY, _vaX,_vaX_rc, _vaY,                     {'dropout_keep':best_dropout,'l2':best_l2_coef, 'lr':best_lr},        _epochs, _batch_size , _best_model_file))


# In[ ]:





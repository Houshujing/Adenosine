import lasagne
from lasagne.layers.shape import PadLayer
from lasagne.layers.merge import ConcatLayer
from lasagne.layers import InputLayer, DenseLayer, NonlinearityLayer, ReshapeLayer, dropout
from lasagne.nonlinearities import softmax, rectify
#from lasagne.layers.normalization import BatchNormLayer, batch_norm
from lasagne.layers.base import Layer
from lasagne.regularization import regularize_network_params, l2, l1
import pandas as pd

import time
import theano
import numpy as np
from theano.tensor import *
import theano.tensor as T
import gc


#then import my own modules
import seqHelper
import lasagneModelsFingerprints


#get the name of files we need, which is the csv files of the activity...
# expr_filename = '../../data/csv_files/logSolubilityTest.csv'
#expr_filename= '/home/jianping/Desktop/wangmukuo/Dual_antagonist/A1/csv_files/insupset_a1.csv'
expr_filename= '/data/home/jianping/Desktop/adenosine/input_data.csv'
#and the name of the fingerprints
#fingerprint_filename = '/home/jianping/Desktop/wangmukuo/Dual_antagonist/A1/fingerprint/train/druglike_control_fingerprints_2048.csv'
fingerprint_filename = '/data/home/jianping/Desktop/adenosine/dnn/output/fingerprint/dnn_fingerprints_2048.csv'
#then get all the hyperparameters as well
batch_size = 250
learning_rate = 0.001
num_epochs = 500
fingerprint_dim = 2048
random_seed = 2020
l2_regularization_lambda = 0.0001
# final_layer_type = lasagne.nonlinearities.linear
final_layer_type = lasagne.nonlinearities.softmax
output_dim = 2
dropout_prob = 0.2
start_time = str(time.ctime()).replace(':','-').replace(' ','_')


#this is a list that can arbitrarily make the neural neural on top of the CNN fingerprint
#it is a list that holds the dimensions of your neural network
# neural_net=[1000]
neural_net = [3000, 2000, 1000]
#neural_net = [4000,3000,2000,1000]

#for no neural network on top, leave it empty
#neural_net=[]


#then make the name of the output to save for testing
neural_net_present = 'True'
if neural_net == []:
    neural_net_present = 'False'


#if expr_filename == '/home/jianping/Desktop/wangmukuo/Dual_antagonist/A1/csv_files/insupset_a1.csv':
if expr_filename == '/data/home/jianping/Desktop/adenosine/input_data.csv':
    test_type = 'drug_likeness'

# progress_filename = '../../output/NN_control-'+neural_net_present+'_'+test_type+'_'+start_time+'.csv'
progress_train_filename = '/data/home/jianping/Desktop/adenosine/deep_learn/ecfpdnn/output/progress_files/train/NN_ecfp_train-' + neural_net_present + '_' + test_type + '.csv'
progress_val_filename = '/data/home/jianping/Desktop/adenosine/deep_learn/ecfpdnn/output/progress_files/val/NN_ecfp_val-' + neural_net_present + '_' + test_type + '.csv'
progress_train_output_df = pd.DataFrame()
progress_val_output_df = pd.DataFrame()


#read in our drug data
smiles_to_prediction,smiles_to_fingerprint \
     = seqHelper.read_in_ecfp_data(expr_filename,fingerprint_filename)

#then get some variables ready to set up my model
experiment_names = smiles_to_prediction.keys()

#get my random training and test set
# test_num = int(float(len(experiment_names))*0.2)
# train_num = int(float(len(experiment_names))*0.8)
# test_list, train_list = seqHelper.gen_rand_train_test_data(experiment_names, test_num, random_seed)

test_num = int(float(len(experiment_names))*0.2)
train_num = int(float(len(experiment_names))*0.7)
val_num = int(float(len(experiment_names))*0.1)
train_list, val_list, test_list = seqHelper.gen_rand_train_val_test_data(experiment_names, test_num, val_num, random_seed)


#define my theano variables
input_fingerprints = fmatrix('input_fingerprints')
# target_vals = fvector('output_data')   #### need to be fix
target_vals = ivector('output_data')

#get my model output
nn_model = lasagneModelsFingerprints.buildControlNN(input_fingerprints,
    fingerprint_dim, output_dim, final_layer_type, dropout_prob, neural_net)

print "Number of parameters:",lasagne.layers.count_params(nn_model['output'])
print "batch size", batch_size
# OUTPUT = open(progress_filename, 'w')
# OUTPUT.write("NUM_PARAMS,"+str(lasagne.layers.count_params(nn_model['output']))+'\n')
# OUTPUT.write("EPOCH,RMSE,MSE\n")
# OUTPUT.close()

#get my training prediction
train_prediction = lasagne.layers.get_output(nn_model['output'],deterministic=False)
# train_prediction = train_prediction.flatten()
# train_loss = lasagne.objectives.squared_error(target_vals,train_prediction)
train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_vals)


#get my loss with l2 regularization
l2_loss = regularize_network_params(nn_model['output'],l2)
train_cost = T.mean(train_loss) + l2_loss*l2_regularization_lambda
train_acc = T.mean(T.eq(T.argmax(train_prediction, axis=1), target_vals), dtype=theano.config.floatX)


#then my parameters and updates from theano
params = lasagne.layers.get_all_params(nn_model['output'], trainable=True)
updates = lasagne.updates.adam(train_cost, params, learning_rate=learning_rate)

#pull out my test predictions
test_prediction = lasagne.layers.get_output(nn_model['output'],deterministic=True)
# test_prediction = test_prediction.flatten()
# test_cost= lasagne.objectives.squared_error(target_vals,test_prediction)
test_cost = lasagne.objectives.categorical_crossentropy(test_prediction, target_vals)
test_cost = test_cost.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_vals), dtype=theano.config.floatX)


#then get my training and test functions
train_func = theano.function([input_fingerprints,target_vals], \
    [train_prediction,train_cost, train_acc], updates=updates, allow_input_downcast=True)

test_func = theano.function([input_fingerprints,target_vals], \
    [test_prediction,test_cost, test_acc], allow_input_downcast=True)

print "compiled functions"

test_acc_list = []
test_cost_list = []
test_epo_list = []
train_acc_list = []
train_cost_list = []
train_epo_list = []
val_acc_list = []
val_cost_list = []
val_epo_list = []


for epoch in xrange(num_epochs):
    print epoch
    progress_filename_train = '/data/home/jianping/Desktop/adenosine/dnn/batch250/log_files/train/drug_like_train-' + str(
        epoch) + neural_net_present + '_' + test_type + '.csv'
    progress_filename_val = '/data/home/jianping/Desktop/adenosine/dnn/batch250/log_files/val/drug_like_val-' + str(
        epoch) + neural_net_present + '_' + test_type + '.csv'

    output_train_df = pd.DataFrame()
    output_val_df = pd.DataFrame()

    #generate the training minibatch
    expr_list_of_lists = seqHelper.gen_batch_list_of_lists(train_list,batch_size,(random_seed+epoch))

    for counter,experiment_list in enumerate(expr_list_of_lists):
        temp_train_df = pd.DataFrame()

        #generate my minibatch parameters for X and Y
        x_fing, y_val = seqHelper.gen_batch_XY_control(experiment_list,\
            smiles_to_prediction,smiles_to_fingerprint)

        #run a training iteration
        # train_output_pred,train_error = train_func(x_fing, y_val)
        train_prediction, train_cost, train_acc = train_func(x_fing, y_val)

        temp_train_df['structure'] = experiment_list
        temp_train_df['train_prediction'] = list(train_prediction)
        temp_train_df['y_value'] = y_val
        output_train_df = pd.concat([output_train_df, temp_train_df])

        train_cost_list.append(train_cost)
        train_acc_list.append(train_acc)
        train_epo_list.append(epoch)
    output_train_df.to_csv(progress_filename_train, mode='a')

    #then run my test
    # test_error_list = []

    if epoch % 1 == 0:

        #generate the test minibatch
        expr_list_of_lists = seqHelper.gen_batch_list_of_lists(val_list,batch_size,(random_seed+epoch))

        #then run through the minibatches
        for experiment_list in expr_list_of_lists:
            temp_df = pd.DataFrame()

            x_fing, y_val = seqHelper.gen_batch_XY_control(experiment_list,\
                smiles_to_prediction,smiles_to_fingerprint)


            #get out the test predictions
            # test_output_pred,test_error_output = test_func(x_fing, y_val)
            val_prediction_output, val_cost_output, val_acc_output = test_func(x_fing, y_val)

            temp_df['structure'] = experiment_list
            temp_df['y_value'] = y_val
            temp_df['val_prediction'] = list(val_prediction_output)

            val_acc_list.append(val_acc_output)
            val_cost_list.append(val_cost_output)
            val_epo_list.append(epoch)
            # test_cost_list += list(test_cost_output).to_list()
            # test_acc_list += test_acc_output.to_list()
            # epoch_list += epoch.to_list()
            output_val_df = pd.concat([output_val_df, temp_df])
        output_val_df.to_csv(progress_filename_val, mode='a')


progress_val_output_df['val_cost'] = val_cost_list
progress_val_output_df['val_acc'] = val_acc_list
progress_val_output_df['epoch'] = val_epo_list
progress_val_output_df.to_csv(progress_val_filename, mode='a')

progress_train_output_df['train_cost'] = train_cost_list
progress_train_output_df['train_acc'] = train_acc_list
progress_train_output_df['epoch'] = train_epo_list
progress_train_output_df.to_csv(progress_train_filename, mode='a')

###test###
expr_list_of_lists = seqHelper.gen_batch_list_of_lists(test_list,batch_size,random_seed)
progress_filename_test = '/data/home/jianping/Desktop/adenosine/dnn/batch250/log_files/test/drug_like_test-' + neural_net_present + '_' + test_type  + '.csv'
output_test_df = pd.DataFrame()




for counter, experiment_list in enumerate(expr_list_of_lists):
    temp_test_df = pd.DataFrame()
    # print counter, experiment_list
    x_fing, y_val = seqHelper.gen_batch_XY_control(experiment_list, smiles_to_prediction, smiles_to_fingerprint)

    # get out the test predictions
    # test_output_pred,test_error_output = test_func(x_fing, y_val)
    test_prediction_output, test_cost_output, test_acc_output = test_func(x_fing, y_val)

    temp_test_df['structure'] = experiment_list
    temp_test_df['test_prediction'] = list(test_prediction_output)
    temp_test_df['y_value'] = y_val
    output_test_df = pd.concat([output_test_df, temp_test_df])

    test_cost_list.append(test_cost)
    test_acc_list.append(test_acc)
output_test_df.to_csv(progress_filename_test, mode='a')


# ### ChemDiv test

tcm_expr_filename = '/data/home/jianping/Desktop/adenosine/chemdiv/avaliablechemdiv2020.csv'
tcm_fingerprint_filename = '/data/home/jianping/Desktop/adenosine/dnn/fingerprint/dual_fingerprints_2048.csv'

tcm_smiles_to_prediction, tcm_smiles_to_fingerprint \
     = seqHelper.read_in_ecfp_data(tcm_expr_filename, tcm_fingerprint_filename)

tcm_test_list = []
for line in open(tcm_expr_filename):
    tcm_test_list.append(line.strip().split(',')[1])
# print tcm_test_list


expr_list_of_lists = seqHelper.gen_batch_list_of_lists(tcm_test_list, batch_size,random_seed)
progress_filename_test = '/data/home/jianping/Desktop/adenosine/dnn/ecfp_screen_test/test/drug_like_yatcm_test_' + neural_net_present + '_' + test_type  + '.csv'
output_test_df = pd.DataFrame()
for counter, experiment_list in enumerate(expr_list_of_lists):
    temp_test_df = pd.DataFrame()
    # print counter, experiment_list
    x_fing, y_val = seqHelper.gen_batch_XY_control(experiment_list, tcm_smiles_to_prediction, tcm_smiles_to_fingerprint)

    # get out the test predictions
    # test_output_pred,test_error_output = test_func(x_fing, y_val)
    test_prediction_output, test_cost_output, test_acc_output = test_func(x_fing, y_val)

    temp_test_df['structure'] = experiment_list
    temp_test_df['test_prediction'] = list(test_prediction_output)
    temp_test_df['y_value'] = y_val
    output_test_df = pd.concat([output_test_df, temp_test_df])

    test_cost_list.append(test_cost)
    test_acc_list.append(test_acc)
output_test_df.to_csv(progress_filename_test, mode='a')

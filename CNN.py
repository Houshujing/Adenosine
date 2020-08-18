import lasagne
from lasagne.regularization import regularize_network_params, l2, l1
import time
import theano
import numpy as np
from theano.tensor import *
import theano.tensor as T
import pandas as pd
import gc
import sys
sys.setrecursionlimit(50000)

#then import my own modules
import seqHelper
import lasagneModelsFingerprints


# expr_filename = '../../data/csv_files/logSolubilityTest.csv'
# expr_filename = '../../data/csv_files/chembl_2.csv'   # my code
expr_filename = '/data/home/jianping/Desktop/adenosine/input_data.csv'
# fingerprint_filename = '../../data/temp/logSolubilityInput_withRDKITidx.pkl'
# fingerprint_filename = '../../data/temp/chembl_2withRDKITidx.pkl' ### my code
fingerprint_filename = '/data/home/jianping/Desktop/adenosine/cnn/fingerprints/cnn_fingerprints.pkl'


#some hyperparameters of the job
batch_size =200
# batch_size = 7

#this is the dimension of the output fingerprint
fingerprint_dim = 265
#this is the dimension of the hiddens of the fingerprint
#the length of the list determines the number of layers for the molecule conv net
fingerprint_network_architecture=[500]*5

#some hyperparameters
learning_rate = 0.001
num_epochs = 500
# num_epochs = 2
random_seed = 2020
output_dim = 2
input_index_dim = 6
l2_regularization_lambda = 0.0001
final_layer_type = lasagne.nonlinearities.softmax
start_time = str(time.ctime()).replace(':','-').replace(' ','_')

#this is a list that can arbitrarily make the neural neural on top of the CNN fingerprint
#it is a list that holds the dimensions of your neural network
#neural_net=[1000]
neural_net = [3000,2000,1000]

#for no neural network on top, leave it empty
#neural_net=[]


#then make the name of the output to save for testing
neural_net_present = 'True'
if neural_net == []:
    neural_net_present = 'False'

#define the name of the output so I can save my predictions
# if expr_filename == '../../data/csv_files/logSolubilityTest.csv':
#     test_type = 'solubility'
# progress_filename = '../../log_files/CNN_fingerprint_NN-'+neural_net_present+'_'+test_type+'_'+start_time+'.csv'


# my code
# if expr_filename == '../../druglike_data/csv_files/decoys1_training_fix.csv':
#     test_type = 'drug-like'
# progress_train_filename = '../../druglike_cnn_add_test_7_2_1/output/CNN_fingerprint_train_NN-' + neural_net_present + '_' + test_type  + '.csv'
# progress_val_filename = '../../druglike_cnn_add_test_7_2_1/output/CNN_fingerprint_val_NN-' + neural_net_present + '_' + test_type + '.csv'
# progress_train_output_df = pd.DataFrame()
# progress_val_output_df = pd.DataFrame()


if expr_filename == '/data/home/jianping/Desktop/adenosine/input_data.csv':
    test_type = 'drug-like'
progress_train_filename = '/data/home/jianping/Desktop/adenosine/deep_learn/cnn/output/train/CNN_fingerprint_train_NN-' + neural_net_present + '_' + test_type + '.csv'
progress_val_filename = '/data/home/jianping/Desktop/adenosine/deep_learn/cnn/output/val/CNN_fingerprint_val_NN-' + neural_net_present + '_' + test_type + '.csv'
progress_train_output_df = pd.DataFrame()
progress_val_output_df = pd.DataFrame()




# progress_filename_2='../../log_files/drug_like_test-' + neural_net_present + '_' + test_type + '_' + start_time + '.csv'
#read in our drug data from seqHelper function also in this folder
smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
    smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask,\
    smiles_to_rdkit_list,max_atom_len,max_bond_len,num_atom_features,num_bond_features\
     = seqHelper.read_in_data(expr_filename,fingerprint_filename)

#grab the names of the experiments so I can make random test and train data
experiment_names = smiles_to_measurement.keys()
# print experiment_names

#get my random training and test set
test_num = int(float(len(experiment_names))*0.2)
train_num = int(float(len(experiment_names))*0.7)
val_num = int(float(len(experiment_names))*0.1)
train_list, val_list, test_list = seqHelper.gen_rand_train_val_test_data(experiment_names, test_num, val_num, random_seed)

#define my theano variables
input_atom = ftensor3('input_atom')
input_atom_index = itensor3('input_atom_index')
input_bonds = ftensor3('input_bonds')
input_bond_index = itensor3('input_mask_attn')
input_mask = fmatrix('input_mask_attn')
target_vals = ivector('output_data')
# target_vals = fmatrix('output_data')

#get my model output
cnn_model = lasagneModelsFingerprints.buildCNNFingerprint(input_atom, input_bonds, \
    input_atom_index, input_bond_index, input_mask, max_atom_len, max_bond_len, num_atom_features, \
    num_bond_features, input_index_dim, fingerprint_dim, batch_size, output_dim, final_layer_type, \
    fingerprint_network_architecture,neural_net)

print "Number of parameters:",lasagne.layers.count_params(cnn_model['output'])
print "batch size",batch_size
# OUTPUT = open(progress_filename, 'a')
# OUTPUT.write("NUM_PARAMS,"+str(lasagne.layers.count_params(cnn_model['output']))+'\n')
# OUTPUT.write("EPOCH,ACC,COST\n")
# OUTPUT.close()
# OUTPUT_test = open(progress_filename_2, 'w')
# OUTPUT_test.write("NUM_PARAMS,"+str(lasagne.layers.count_params(cnn_model['output']))+'\n')
# OUTPUT_test.write("EPOCH,RMSE,MSE\n")
# OUTPUT_test.close()

#get the output of the model
train_prediction = lasagne.layers.get_output(cnn_model['output'],deterministic=False)
# print train_prediction
#flatten the prediction
# train_prediction = train_prediction.flatten()
#define the loss as the trained error
# train_loss = lasagne.objectives.squared_error(target_vals,train_prediction)
train_loss = lasagne.objectives.categorical_crossentropy(train_prediction, target_vals)

#regularize the network with L2 regularization
l2_loss = regularize_network_params(cnn_model['output'],l2)
train_cost = T.mean(train_loss) + l2_loss*l2_regularization_lambda
train_acc = T.mean(T.eq(T.argmax(train_prediction, axis=1), target_vals), dtype=theano.config.floatX)


#get the parameters and updates from lasagne
params = lasagne.layers.get_all_params(cnn_model['output'], trainable=True)
updates = lasagne.updates.adam(train_cost, params, learning_rate=learning_rate)

#do this also for the test data
test_prediction = lasagne.layers.get_output(cnn_model['output'],deterministic=True)
# test_prediction = test_prediction.flatten()
# test_cost= lasagne.objectives.squared_error(target_vals,test_prediction)
test_cost = lasagne.objectives.categorical_crossentropy(test_prediction, target_vals)
test_cost = test_cost.mean()
test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_vals), dtype=theano.config.floatX)

#define our functions for train and test
# train_func = theano.function([input_atom,input_bonds,input_atom_index,\
#     input_bond_index,input_mask,target_vals], [train_cost], updates=updates, allow_input_downcast=True)

train_func = theano.function([input_atom,input_bonds,input_atom_index,\
    input_bond_index,input_mask,target_vals], [train_prediction, train_cost, train_acc], updates=updates, allow_input_downcast=True)

# test_func = theano.function([input_atom,input_bonds,input_atom_index,\
#     input_bond_index,input_mask,target_vals], [test_prediction,test_cost], allow_input_downcast=True)


# my code
test_func = theano.function([input_atom,input_bonds,input_atom_index,\
    input_bond_index,input_mask,target_vals], [test_prediction, test_cost, test_acc], allow_input_downcast=True)

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
    progress_filename_train = '/data/home/jianping/Desktop/adensine/cnn/batch200/log_files/train/drug_like_train-' + str(epoch) + neural_net_present + '_' + test_type + '.csv'
    progress_filename_val = '/data/home/jianping/Desktop/adenosine/cnn/batch200/log_files/val/drug_like_val-' + str(epoch) + neural_net_present + '_' + test_type  + '.csv'
    #OUTPUT_train = open(progress_filename_train, 'a')
    output_train_df = pd.DataFrame()
    output_val_df = pd.DataFrame()
    # output_test_df['structure'] = []
    # output_test_df['y_value'] = []
    # output_test_df['test_prediction'] = []

    #this function makes a list of lists that is the minibatch
    expr_list_of_lists = seqHelper.gen_batch_list_of_lists(train_list,batch_size,(random_seed+epoch))
    #print len(train_list),batch_size,(random_seed+epoch)
    # print expr_list_of_lists
    #then loop through the minibatches

    #print len(expr_list_of_lists)
    for counter,experiment_list in enumerate(expr_list_of_lists):
        temp_train_df = pd.DataFrame()
        #print counter, experiment_list
        x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val = seqHelper.gen_batch_XY_reg(experiment_list,\
            smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
            smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask)
        train_prediction, train_cost, train_acc = train_func(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y_val)

        temp_train_df['structure'] = experiment_list
        temp_train_df['train_prediction'] = list(train_prediction)
        temp_train_df['y_value'] = y_val
        output_train_df = pd.concat([output_train_df, temp_train_df])

        train_cost_list.append(train_cost)
        train_acc_list.append(train_acc)
        train_epo_list.append(epoch)
    output_train_df.to_csv(progress_filename_train, mode='a')

    # print experiment_list,train_prediction, y_val
        # print pd.DataFrame(str(counter),str(experiment_list) , str(y_val) ,str(train_prediction) )
    # print len(y_val),len(train_prediction)

        #run a training iteration
        # train_error = train_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)

        # print train_prediction
    # test_cost_list = []
    # test_acc_list = []
    # epoch_list = []
    #every certain number of epochs, run the test data as well
    if epoch % 1 == 0:
        expr_list_of_lists = seqHelper.gen_batch_list_of_lists(val_list,batch_size,(random_seed+epoch))
        # print expr_list_of_lists
        for experiment_list in expr_list_of_lists:
            temp_df = pd.DataFrame()
            x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val = seqHelper.gen_batch_XY_reg(experiment_list,\
                smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
                smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask)

            #run the test output
            # test_prediction_output, test_error_output = test_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)
            val_prediction_out, val_cost_output, val_acc_output = test_func(x_atom,x_bonds,x_atom_index,x_bond_index,x_mask,y_val)
            # print len(y_val), len(test_prediction_out),len(experiment_list)
            # print experiment_list,test_prediction_out,y_val

            temp_df['structure'] = experiment_list
            temp_df['y_value'] = y_val
            temp_df['val_prediction'] = list(val_prediction_out)


            val_acc_list.append(val_acc_output)
            val_cost_list.append(val_cost_output)
            val_epo_list.append(epoch)
            # test_cost_list += list(test_cost_output).to_list()
            # test_acc_list += test_acc_output.to_list()
            # epoch_list += epoch.to_list()
            output_val_df = pd.concat([output_val_df, temp_df])
        output_val_df.to_csv(progress_filename_val, mode='a')

expr_list_of_lists = seqHelper.gen_batch_list_of_lists(test_list,batch_size,random_seed)
progress_filename_test = '/data/home/jianping/Desktop/adenosine/cnn/batch200/log_files/test/drug_like_test-' + neural_net_present + '_' + test_type  + '.csv'
output_test_df = pd.DataFrame()
for counter, experiment_list in enumerate(expr_list_of_lists):
    temp_test_df = pd.DataFrame()
    # print counter, experiment_list
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y_val = seqHelper.gen_batch_XY_reg(experiment_list, \
                                                                                            smiles_to_measurement,
                                                                                            smiles_to_atom_info,
                                                                                            smiles_to_bond_info, \
                                                                                            smiles_to_atom_neighbors,
                                                                                            smiles_to_bond_neighbors,
                                                                                            smiles_to_atom_mask)
    test_prediction, test_cost, test_acc = test_func(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y_val)

    temp_test_df['structure'] = experiment_list
    temp_test_df['test_prediction'] = list(test_prediction)
    temp_test_df['y_value'] = y_val
    output_test_df = pd.concat([output_test_df, temp_test_df])

    test_cost_list.append(test_cost)
    test_acc_list.append(test_acc)
output_test_df.to_csv(progress_filename_test, mode='a')


#####ChemDiv######

#tcm_expr_filename = '/home/jianping/Desktop/wangmukuo/Dual_antagonist/Chemdiv/ch1000.csv'
tcm_expr_filename ='/data/home/jianping/Desktop/adenosine/chemdiv.csv'
#tcm_fingerprint_filename = '/home/jianping/Desktop/wangmukuo/Dual_antagonist/Chemdiv/fingerprint/ch1000.pkl'
tcm_fingerprint_filename = '/data/home/jianping/Desktop/adenosine/cnn/fingerprints/fingerprint_chemdiv.pkl'
smiles_to_measurement,smiles_to_atom_info,smiles_to_bond_info,\
    smiles_to_atom_neighbors,smiles_to_bond_neighbors,smiles_to_atom_mask,\
    smiles_to_rdkit_list,max_atom_len,max_bond_len,num_atom_features,num_bond_features\
     = seqHelper.read_in_data(tcm_expr_filename,tcm_fingerprint_filename)

tcm_test_list = pd.read_csv(tcm_expr_filename, header=None)[1]
# print(tcm_test_list.columns)
print tcm_test_list


expr_list_of_lists = seqHelper.gen_batch_list_of_lists(tcm_test_list, batch_size, random_seed)
progress_filename_test = '/data/home/jianping/Desktop/adenosine/cnn/screen_result/drug_like_chemdiv_screen-' + neural_net_present + '_' + test_type  + '.csv'
output_test_df = pd.DataFrame()
for counter, experiment_list in enumerate(expr_list_of_lists):
    temp_test_df = pd.DataFrame()
    # print counter, experiment_list
    x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y_val = seqHelper.gen_batch_XY_reg(experiment_list, \
                                                                                            smiles_to_measurement,
                                                                                            smiles_to_atom_info,
                                                                                            smiles_to_bond_info, \
                                                                                            smiles_to_atom_neighbors,
                                                                                            smiles_to_bond_neighbors,
                                                                                            smiles_to_atom_mask)
    test_prediction, test_cost, test_acc = test_func(x_atom, x_bonds, x_atom_index, x_bond_index, x_mask, y_val)

    temp_test_df['structure'] = experiment_list
    temp_test_df['test_prediction'] = list(test_prediction)
    temp_test_df['y_value'] = y_val
    output_test_df = pd.concat([output_test_df, temp_test_df])

    test_cost_list.append(test_cost)
    test_acc_list.append(test_acc)
output_test_df.to_csv(progress_filename_test, mode='a')



#######

progress_val_output_df['val_cost'] = val_cost_list
progress_val_output_df['val_acc'] = val_acc_list
progress_val_output_df['epoch'] = val_epo_list
progress_val_output_df.to_csv(progress_val_filename, mode='a')

progress_train_output_df['train_cost'] = train_cost_list
progress_train_output_df['train_acc'] = train_acc_list
progress_train_output_df['epoch'] = train_epo_list
progress_train_output_df.to_csv(progress_train_filename, mode='a')
np.savez('/data/home/jianping/Desktop/wangmukuo/dual_new/deep_learn/cnn/model.npz', *lasagne.layers.get_all_param_values(cnn_model['output']))

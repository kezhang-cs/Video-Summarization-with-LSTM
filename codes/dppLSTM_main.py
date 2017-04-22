"""
Video Summarization with Long Short-term Memory
4 datasets used in our task: OVP, Youtube, SumMe, TVSum: Pre-train on OVP, Youtube and SumMe/TVSum, test on TVSum/SumMe
Ke Zhang
Apr, 2017
"""

import os
import numpy
import theano
import h5py
from tools import data_loader
from optimizer.adam_opt import adam_opt
from layers.summ_dppLSTM import summ_dppLSTM


def train(model_idx, train_set, val_set, lr=0.001, n_iters=100, minibatch=10, valid_period=1, model_saved = ''):

    print('... training')
    model_save_dir = '../models/' + model_idx + '/'
    if os.path.exists(model_save_dir):
        os.system('rm -r %s' % model_save_dir)
    os.mkdir(model_save_dir)

    # build model
    print('... building model')
    model = summ_dppLSTM(model_file = model_saved)
    train_seq = data_loader.SequenceDataset(train_set, batch_size=None, number_batches=minibatch)
    valid_seq = data_loader.SequenceDataset(val_set, batch_size=None, number_batches=len(val_set[0]))

    # train model
    adam_opt(model, train_seq, valid_seq, model_save_dir = model_save_dir, minibatch = minibatch,
             valid_period = valid_period, n_iters=n_iters, lr=lr)

def inference(model_file, model_idx, test_set, test_dir, te_idx):

    print('... inference')
    if os.path.exists(model_file) == 0:
        print('model doesn\'t exist')
        return

    model = summ_dppLSTM(model_file = model_file)
    res = test_dir + '/' + model_idx + '_inference.h5'
    f = h5py.File(res, 'w')
    h_func = theano.function(inputs=[model.inputs[0]], outputs=model.classify_mlp.h[-1])
    h_func_k = theano.function(inputs=[model.inputs[0]], outputs=model.kernel_mlp.h[-1])
    cFrm = []
    for i in xrange(len(test_set[0])):
        cFrm.append(test_set[0][i].shape[0])
    xf = h5py.File(model_file, 'r')
    xf.keys()

    pred = []
    pred_k = []
    for seq in test_set[0]:
        pred.append(h_func(seq))
        pred_k.append(h_func_k(seq))
    pred = numpy.concatenate(pred, axis=0)
    pred_k = numpy.concatenate(pred_k, axis=0)

    f['pred'] = pred
    f['pred_k'] = pred_k
    f['cFrm'] = cFrm
    f['idx'] = te_idx

    f.close()

if __name__ == '__main__':

    dataset_testing = 'SumMe' # testing dataset: SumMe or TVSum
    model_type = 2 # 1 for vsLSTM and 2 for dppLSTM, please refer to the readme file for more detail
    model_idx = 'dppLSTM_' + dataset_testing + '_' + model_type.__str__()

    # load data
    print('... loading data')
    train_set, val_set, val_idx, test_set, te_idx = data_loader.load_data(data_dir = '../data/', dataset_testing = dataset_testing, model_type = model_type)
    model_file = '../models/model_trained_' + dataset_testing

    """
    Uncomment the following line if you want to train the model
    """
    # train(model_idx = model_idx, train_set = train_set, val_set = val_set, model_saved = model_file)

    inference(model_file=model_file, model_idx = model_idx, test_set=test_set, test_dir='./res_LSTM/', te_idx=te_idx)
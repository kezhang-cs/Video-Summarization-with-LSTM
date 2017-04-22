__author__ = 'kezhang'

import sys
import h5py
import json
import numpy
import theano
import scipy.io as sio

def load_data(data_dir = '../data/SOY/', dataset_testing = 'TVSum', model_type = 1):
    
    train_set = [[], [], [], []]
    
    [feature, label, weight] = load_dataset_h5(data_dir, 'OVP', model_type)
    label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
    train_set[0].extend(feature)
    train_set[1].extend(label)
    train_set[2].extend(label_tmp)
    train_set[3].extend(weight)

    [feature, label, weight] = load_dataset_h5(data_dir, 'Youtube', model_type)
    label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
    train_set[0].extend(feature)
    train_set[1].extend(label)
    train_set[2].extend(label_tmp)
    train_set[3].extend(weight)

    val_set = [[], [], [], []]
    val_idx = []
    test_set = [[], [], [], []]
    te_idx = []

    if dataset_testing == 'TVSum':
        [feature, label, weight] = load_dataset_h5(data_dir, 'TVSum', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        test_set[0].extend(feature)
        test_set[1].extend(label)
        test_set[2].extend(label_tmp)
        test_set[3].extend(weight)
        te_idx.extend(range(50))
        [feature, label, weight] = load_dataset_h5(data_dir, 'SumMe', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        rand_idx = numpy.random.permutation(25)
        for i in xrange(25):
            if i <= 15:
                train_set[0].append(feature[rand_idx[i]])
                train_set[1].append(label[rand_idx[i]])
                train_set[2].append(label_tmp[rand_idx[i]])
                train_set[3].append(weight[rand_idx[i]])
            else:
                val_set[0].append(feature[rand_idx[i]])
                val_set[1].append(label[rand_idx[i]])
                val_set[2].append(label_tmp[rand_idx[i]])
                val_set[3].append(weight[rand_idx[i]])
                val_idx.append(rand_idx[i])

    elif dataset_testing == 'SumMe':
        [feature, label, weight] = load_dataset_h5(data_dir, 'SumMe', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        test_set[0].extend(feature)
        test_set[1].extend(label)
        test_set[2].extend(label_tmp)
        test_set[3].extend(weight)
        te_idx.extend(range(25))
        [feature, label, weight] = load_dataset_h5(data_dir, 'TVSum', model_type)
        label_tmp = [numpy.where(l)[0].astype('int32') for l in label]
        rand_idx = numpy.random.permutation(50)
        for i in xrange(50):
            if i <= 30:
                train_set[0].append(feature[rand_idx[i]])
                train_set[1].append(label[rand_idx[i]])
                train_set[2].append(label_tmp[rand_idx[i]])
                train_set[3].append(weight[rand_idx[i]])
            else:
                val_set[0].append(feature[rand_idx[i]])
                val_set[1].append(label[rand_idx[i]])
                val_set[2].append(label_tmp[rand_idx[i]])
                val_set[3].append(weight[rand_idx[i]])
                val_idx.append(rand_idx[i])

    for i in xrange(len(train_set[0])):
        train_set[0][i] = numpy.transpose(train_set[0][i])
        train_set[1][i] = train_set[1][i].flatten().astype(theano.config.floatX)
        train_set[2][i] = train_set[2][i].flatten().astype('int32')
        train_set[3][i] = train_set[3][i]

    for i in xrange(len(val_set[0])):
        val_set[0][i] = numpy.transpose(val_set[0][i])
        val_set[1][i] = val_set[1][i].flatten().astype(theano.config.floatX)
        val_set[2][i] = val_set[2][i].flatten().astype('int32')
        val_set[3][i] = val_set[3][i]

    for i in xrange(len(test_set[0])):
        test_set[0][i] = numpy.transpose(test_set[0][i])
        test_set[1][i] = test_set[1][i].flatten().astype(theano.config.floatX)
        test_set[2][i] = test_set[2][i].flatten().astype('int32')
        test_set[3][i] = test_set[3][i]

    return train_set, val_set, val_idx, test_set, te_idx

def load_dataset_h5(data_dir, dataset, label_type):
    # loading data from a dataset in the format of hdf5 file
    feature = []
    label = []
    weight = []
    file_name = data_dir + '/Data_' + dataset + '_google_p5.h5'
    f = h5py.File(file_name)
    vid_ord = numpy.sort(numpy.array(f['/ord']).astype('int32').flatten())
    for i in vid_ord:
        feature.append(numpy.matrix(f['/fea_' + i.__str__()]).astype(theano.config.floatX))
        label.append(numpy.array(f['/gt_' + label_type.__str__() + '_' + i.__str__()]).astype(theano.config.floatX).flatten())
        weight.append(numpy.array(label_type - 1.0).astype(theano.config.floatX))
    f.close()

    return feature, label, weight

def load_data_mat(data_dir, idx):

    mat_contents = sio.loadmat(data_dir)
    feature = mat_contents['fea' + idx]
    label = mat_contents['fea' + idx]
    return feature, label

class SequenceDataset:
  '''Slices, shuffles and manages a small dataset for the HF optimizer.'''

  def __init__(self, data, batch_size, number_batches, minimum_size=10):
    '''SequenceDataset __init__

  data : list of lists of numpy arrays
    Your dataset will be provided as a list (one list for each graph input) of
    variable-length tensors that will be used as mini-batches. Typically, each
    tensor is a sequence or a set of examples.
  batch_size : int or None
    If an int, the mini-batches will be further split in chunks of length
    `batch_size`. This is useful for slicing subsequences or provide the full
    dataset in a single tensor to be split here. All tensors in `data` must
    then have the same leading dimension.
  number_batches : int
    Number of mini-batches over which you iterate to compute a gradient or
    Gauss-Newton matrix product.
  minimum_size : int
    Reject all mini-batches that end up smaller than this length.'''
    self.current_batch = 0
    self.number_batches = number_batches
    self.items = []

    for i_sequence in xrange(len(data[0])):
      if batch_size is None:
        self.items.append([data[i][i_sequence] for i in xrange(len(data))])
      else:
        for i_step in xrange(0, len(data[0][i_sequence]) - minimum_size + 1, batch_size):
          self.items.append([data[i][i_sequence][i_step:i_step + batch_size] for i in xrange(len(data))])

    self.shuffle()

  def shuffle(self):
    numpy.random.shuffle(self.items)

  def iterate(self, update=True):
    for b in xrange(self.number_batches):
      yield self.items[(self.current_batch + b) % len(self.items)]
    if update: self.update()

  def update(self):
    if self.current_batch + self.number_batches >= len(self.items):
      self.shuffle()
      self.current_batch = 0
    else:
      self.current_batch += self.number_batches

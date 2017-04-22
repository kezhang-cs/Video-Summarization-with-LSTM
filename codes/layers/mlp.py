import h5py
import numpy
import theano
import theano.tensor as T


class mlp(object):
    """
    An implement of mlp. Support two initialization methods:
    1. create a new one from the given settings
    2. read the model parameters from a given file
    """
    def __init__(self, layers=[-1, -1], model_file=None, layer_name='mlp', inputs=None, net_type='tanh'):
        self.layer_name = layer_name
        self.layers = layers
        # create a new model
        if model_file == None:
            self.W = []
            self.b = []
            for i in xrange(len(layers)-1):
                self.W.append(theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (layers[i], layers[i+1])).astype(theano.config.floatX)))
                self.W[-1].name = self.layer_name + '_W' + str(i)
                self.b.append(theano.shared(numpy.zeros(layers[i+1]).astype(theano.config.floatX)))
                self.b[-1].name = self.layer_name + '_b' + str(i)
        # read from model_file
        else:
            # read model file, in hdf5 format
            f = h5py.File(model_file)
            self.W = []
            self.b = []
            self.layers = []
            i = 0
            while True:
                if self.layer_name+'_W'+str(i) in f:
                    self.W.append(theano.shared(numpy.array(f[self.layer_name+'_W'+str(i)]).astype(theano.config.floatX)))
                    self.W[-1].name = self.layer_name + '_W' + str(i)
                    self.layers.append(self.W[-1].get_value().shape[0])
                    self.b.append(theano.shared(numpy.array(f[self.layer_name+'_b'+str(i)]).astype(theano.config.floatX)))
                    self.b[-1].name = self.layer_name + '_b' + str(i)
                    i += 1
                else:
                    self.layers.append(self.W[-1].get_value().shape[1])
                    break
            # close the hdf5 model file
            f.close()

        self.params = []
        self.params.extend(self.W)
        self.params.extend(self.b)

        assert inputs != None
        x = inputs[0]
        self.inputs = [x]
        self.h = [x]

        def relu(x):
            return x * (x > 0)

        for i in xrange(len(self.layers)-2):
            self.h.append(T.nnet.sigmoid(T.dot(self.h[i], self.W[i]) + self.b[i]))

        if net_type == 'tanh':
            self.h.append(T.tanh(T.dot(self.h[-1], self.W[-1]) + self.b[-1]))
        elif net_type == 'softmax':
            self.h.append(T.nnet.softmax(T.dot(self.h[-1], self.W[-1]) + self.b[-1]))
        elif net_type == 'sigmoid':
            self.h.append(T.nnet.sigmoid(T.dot(self.h[-1], self.W[-1]) + self.b[-1]))
        elif net_type == 'linear':
            self.h.append(T.dot(self.h[-1], self.W[-1]) + self.b[-1])
        else:
            print('Unsupported MLP Type !!')

    def save_to_file(self, file_dir, file_index=-1):
        file_name = file_dir + self.layer_name + '.h5'
        if file_index >= 0:
            file_name = file_name + '.' + str(file_index)
        f = h5py.File(file_name)
        for p in self.params:
            f[p.name] = p.get_value()
        f.close()

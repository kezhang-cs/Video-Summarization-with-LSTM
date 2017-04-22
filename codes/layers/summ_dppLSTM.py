import h5py
import numpy
import theano
import theano.tensor as T
from mlp import mlp


class summ_dppLSTM(object):
    """
    bidirectional LSTM units: h_backwards + h_forwards to MLP
    """
    def __init__(self, nx=-1, nh=-1, nout=-1, model_file=None, layer_name='sumLSTM_bid', inputs=None):
        self.layer_name = layer_name
        # input video
        if inputs == None:
            video = T.matrix('video')
            label = T.vector('label')
            labelS = T.ivector('labelS')
            dpp_weight = T.dscalar('dpp_weight')
        else:
            video = inputs[0]
            label = inputs[1]
            labelS = inputs[2]
            dpp_weight = inputs[3]

        # create a new model
        if model_file == None:
            # image feature projection
            self.c_init_mlp = mlp(layers=[nx, nh], layer_name='c_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            self.h_init_mlp = mlp(layers=[nx, nh], layer_name='h_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            # input gate
            self.Wi = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype(theano.config.floatX))
            self.Wi.name = self.layer_name + '_Wi'
            self.bi = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype(theano.config.floatX))
            self.bi.name = self.layer_name + '_bi'
            # input modulator
            self.Wc = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype(theano.config.floatX))
            self.Wc.name = self.layer_name + '_Wc'
            self.bc = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype(theano.config.floatX))
            self.bc.name = self.layer_name + '_bc'
            # forget gate
            self.Wf = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype(theano.config.floatX))
            self.Wf.name = self.layer_name + '_Wf'
            self.bf = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype(theano.config.floatX))
            self.bf.name = self.layer_name + '_bf'
            # output gate
            self.Wo = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, (nx+nh, nh)).astype(theano.config.floatX))
            self.Wo.name = self.layer_name + '_Wo'
            self.bo = theano.shared(0.02 * numpy.random.uniform(-1.0, 1.0, nh).astype(theano.config.floatX))
            self.bo.name = self.layer_name + '_bo'
        # read from model_file
        else:
            # read model file, in hdf5 format
            f = h5py.File(model_file)
            # image feature projection
            self.c_init_mlp = mlp(model_file=model_file, layer_name='c_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            self.h_init_mlp = mlp(model_file=model_file, layer_name='h_init_mlp', inputs=[T.mean(video, axis=0)], net_type='tanh')
            # input gate
            self.Wi = theano.shared(numpy.array(f[self.layer_name+'_Wi']).astype(theano.config.floatX))
            self.Wi.name = self.layer_name + '_Wi'
            self.bi = theano.shared(numpy.array(f[self.layer_name+'_bi']).astype(theano.config.floatX))
            self.bi.name = self.layer_name + '_bi'
            # input modulator
            self.Wc = theano.shared(numpy.array(f[self.layer_name+'_Wc']).astype(theano.config.floatX))
            self.Wc.name = self.layer_name + '_Wc'
            self.bc = theano.shared(numpy.array(f[self.layer_name+'_bc']).astype(theano.config.floatX))
            self.bc.name = self.layer_name + '_bc'
            # forget gate
            self.Wf = theano.shared(numpy.array(f[self.layer_name+'_Wf']).astype(theano.config.floatX))
            self.Wf.name = self.layer_name + '_Wf'
            self.bf = theano.shared(numpy.array(f[self.layer_name+'_bf']).astype(theano.config.floatX))
            self.bf.name = self.layer_name + '_bf'
            # output gate
            self.Wo = theano.shared(numpy.array(f[self.layer_name+'_Wo']).astype(theano.config.floatX))
            self.Wo.name = self.layer_name + '_Wo'
            self.bo = theano.shared(numpy.array(f[self.layer_name+'_bo']).astype(theano.config.floatX))
            self.bo.name = self.layer_name + '_bo'
            # close the hdf5 model file
            f.close()

        # record the size information
        self.nx = self.c_init_mlp.W[0].get_value().shape[0]
        self.nh = self.c_init_mlp.b[-1].get_value().shape[0]
        # add all above into params
        self.params = [self.Wi, self.bi,
                       self.Wc, self.bc,
                       self.Wf, self.bf,
                       self.Wo, self.bo]
        self.params.extend(self.c_init_mlp.params)
        self.params.extend(self.h_init_mlp.params)

        # initializing memory cell and hidden state
        self.c0 = self.c_init_mlp.h[-1]
        self.h0 = self.h_init_mlp.h[-1]

        # go thru the sequence
        # forwards
        ([self.c, self.h], updates) = theano.scan(fn=self.one_step, sequences=[video], outputs_info=[self.c0, self.h0])
        self.c0_back = self.c_init_mlp.h[-1]
        self.h0_back = self.h_init_mlp.h[-1]
        # backwards
        ([self.c_back, self.h_back], updates) = theano.scan(fn=self.one_step, sequences=[video[::-1, :]], outputs_info=[self.c0_back, self.h0_back])
        self.h = T.concatenate([self.h, self.h_back[::-1, :]], axis=1)
        self.h = T.concatenate([video, self.h], axis=1)
        # predicted probobility
        if model_file == None:
            self.classify_mlp = mlp(layers=[self.nx + 2*self.nh, nh, 1],
                                    layer_name='classify_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')

            self.kernel_mlp = mlp(layers=[self.nx + 2*self.nh, nh, nout],
                                    layer_name='kernel_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')
        else:
            self.classify_mlp = mlp(model_file=model_file,
                                    layer_name='classify_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')

            self.kernel_mlp = mlp(model_file=model_file,
                                    layer_name='kernel_mlp',
                                    # inputs=[self.h[-1, :]],
                                    inputs =[self.h],
                                    net_type='linear')

        self.nout = self.classify_mlp.b[-1].get_value().shape[0]
        self.params.extend(self.classify_mlp.params)
        self.pred = self.classify_mlp.h[-1]

        self.nout_k = self.kernel_mlp.b[-1].get_value().shape[0]
        self.params.extend(self.kernel_mlp.params)
        self.pred_k = self.kernel_mlp.h[-1]

        kv = self.pred_k # kv means kernel_vector
        qv = self.pred
        K_mat = T.dot(kv, kv.T)
        Q_mat = T.outer(qv, qv)
        L = K_mat * Q_mat
        Ly = L[labelS, :][:, labelS]

        dpp_loss = (- (T.log(T.nlinalg.Det()(Ly)) - T.log(T.nlinalg.Det()(L + T.identity_like(L)))))
        if not T.isnan(dpp_loss):
            loss = T.mean(T.sqr(self.pred.flatten() - label)) + dpp_weight * dpp_loss
        else:
            loss = T.mean(T.sqr(self.pred.flatten() - label)) + dpp_weight * T.nlinalg.Det()(Ly + T.identity_like(Ly)) # when the dpp_loss is nan, just randomly fill in a number

        acc = T.log(T.nlinalg.Det()(L + T.identity_like(L)))

        self.inputs = [video, label, labelS, dpp_weight]
        self.costs = [loss, acc]

    def one_step(self, x_t, c_tm1, h_tm1):
        x_and_h = T.concatenate([x_t, h_tm1], axis=0)
        i_t = T.nnet.sigmoid(T.dot(x_and_h, self.Wi) + self.bi)
        c_tilde = T.tanh(T.dot(x_and_h, self.Wc) + self.bc)
        f_t = T.nnet.sigmoid(T.dot(x_and_h, self.Wf) + self.bf)
        o_t = T.nnet.sigmoid(T.dot(x_and_h, self.Wo) + self.bo)
        c_t = i_t * c_tilde + f_t * c_tm1
        h_t = o_t * T.tanh(c_t)
        return [c_t, h_t]

    def save_to_file(self, file_dir, file_index=-1):
        file_name = file_dir + self.layer_name + '.h5'
        if file_index >= 0:
            file_name = file_name + '.' + str(file_index)
        f = h5py.File(file_name)
        for p in self.params:
            f[p.name] = p.get_value()
        f.close()


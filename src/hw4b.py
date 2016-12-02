"""
Source Code for Homework 4.b of ECBM E4040, Fall 2016, Columbia University

"""

import os
import timeit
import inspect
import sys
import numpy
from collections import OrderedDict
import copy
import random
import theano
import theano.tensor as T
from theano.tensor.nnet import conv2d
from theano.tensor.signal import downsample

from hw4_utils import contextwin, shared_dataset, load_data, shuffle, conlleval, check_dir
from hw4_nn import myMLP, train_nn
import sys
sys.setrecursionlimit(5000)

def gen_parity_pair(nbit, num, is_rnn=0):
    """
    Generate binary sequences and their parity bits

    :type nbit: int
    :param nbit: length of binary sequence

    :type num: int
    :param num: number of sequences

    """
    if is_rnn==0:
        X = numpy.random.randint(2, size=(num, nbit))
        Y = numpy.mod(numpy.sum(X, axis=1), 2)
        
        return X, Y
    elif is_rnn==1:
        X = numpy.random.randint(2, size=(num, nbit))
        Y = numpy.empty((num,nbit))
        Y[:,0] = X[:,0]%2
        for i in range(1,nbit):
            Y[:,i] = (numpy.sum(X[:,0:(i+1)], axis=1))
            Y[:,i] = Y[:,i]%2
        return X, Y.astype(int)
    else:
        print ("Unsupported Type")
        sys.exit(0)
#TODO: implement RNN class to learn parity function
class RNN(object):
    """ Elman Neural Net Model Class
    """
    def __init__(self, nh, nc, ne, cs):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.wx = theano.shared(name='wx',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (cs, nh))
                                .astype(theano.config.floatX))
        self.wh = theano.shared(name='wh',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bh = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))
        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        # bundle
        self.params = [self.wx, self.wh, self.w,self.bh, self.b, self.h0]

        # as many columns as context window size
        # as many lines as words in the sentence
        x = T.matrix()
        y_sentence = T.ivector('y_sentence')  # labels


        def recurrence(x_t, h_tm1):
            h_t = T.nnet.sigmoid(T.dot(x_t, self.wx) + T.dot(h_tm1, self.wh) + self.bh)
            
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b)
            return [h_t, s_t]

        [h, s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0, None],
                                n_steps=x.shape[0])

        p_y_given_x_sentence = s[:,0,:]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')

        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)[T.arange(y_sentence.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred)
        self.sentence_train = theano.function(inputs=[x, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              allow_input_downcast=True)
        # self.normalize = theano.function(inputs=[],
        #                                  updates={self.emb:
        #                                           self.emb /
        #                                           T.sqrt((self.emb**2)
        #                                           .sum(axis=1))
        #                                           .dimshuffle(0, 'x')})
        # self.normal = normal

    def train(self, x, y, window_size, learning_rate):

        cwords = contextwin(x, window_size)
        words = list(map(lambda x: numpy.asarray(x).astype('int32'), cwords))
        labels = y

        self.sentence_train(words, labels, learning_rate)
        # if self.normal:
        #     self.normalize()

    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))    
        


#TODO: implement LSTM class to learn parity function
class LSTM(object):
    def __init__(self, nh, nc):
        """Initialize the parameters for the RNNSLU

        :type nh: int
        :param nh: dimension of the hidden layer

        :type nc: int
        :param nc: number of classes

        :type ne: int
        :param ne: number of word embeddings in the vocabulary

        :type de: int
        :param de: dimension of the word embeddings

        :type cs: int
        :param cs: word window context size

        :type normal: boolean
        :param normal: normalize word embeddings after each update or not.

        """
        # parameters of the model
        self.wi = theano.shared(name='wi',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.ui = theano.shared(name='ui',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wf = theano.shared(name='wf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.uf = theano.shared(name='uf',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wc = theano.shared(name='wc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.uc = theano.shared(name='uc',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.wo = theano.shared(name='wo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (1, nh))
                                .astype(theano.config.floatX))
        self.uo = theano.shared(name='uo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))
        self.vo = theano.shared(name='vo',
                                value=0.2 * numpy.random.uniform(-1.0, 1.0,
                                (nh, nh))
                                .astype(theano.config.floatX))

        self.w = theano.shared(name='w',
                               value=0.2 * numpy.random.uniform(-1.0, 1.0,
                               (nh, nc))
                               .astype(theano.config.floatX))
        self.bi = theano.shared(name='bi',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bf = theano.shared(name='bf',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bc = theano.shared(name='bc',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.bo = theano.shared(name='bh',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.b = theano.shared(name='b',
                               value=numpy.zeros(nc,
                               dtype=theano.config.floatX))

        self.h0 = theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))
        self.c0= theano.shared(name='h0',
                                value=numpy.zeros(nh,
                                dtype=theano.config.floatX))


        




        # bundle
        self.params = [self.wi, self.wf, self.wc, self.wo, self.w, 
                       self.ui, self.uf, self.uc, self.uo,
                       self.vo, self.bi, self.bc, self.bf, self.bo, self.b, self.h0, self.c0]

        # as many columns as context window size
        # as many lines as words in the sentence
        x = T.matrix()
        y_sentence = T.ivector('y_sentence')  # labels

        def recurrence(x_t, h_tm1,c_tm1):
            
            i_t = T.nnet.sigmoid(T.dot(x_t, self.wi) + T.dot(h_tm1, self.ui)  + self.bi)
            f_t = T.nnet.sigmoid(T.dot(x_t, self.wf) + T.dot(h_tm1, self.uf) + self.bf)
            c_telda_t= T.tanh(T.dot(x_t, self.wc)+ T.dot(h_tm1, self.uc) + self.bc)
            c_t = f_t * c_tm1 + i_t * c_telda_t
            o_t = T.nnet.sigmoid(T.dot(x_t, self.wo)+ T.dot(h_tm1, self.uo) + T.dot(c_t, self.vo)  + self.bo)
            h_t = o_t * T.tanh(c_t)
            s_t = T.nnet.softmax(T.dot(h_t, self.w) + self.b) 
            return [h_t, c_t, s_t]
        [h,c,s], _ = theano.scan(fn=recurrence,
                                sequences=x,
                                outputs_info=[self.h0,self.c0, None],
                                n_steps=x.shape[0])


        
        p_y_given_x_sentence = s[:, 0, :]
        y_pred = T.argmax(p_y_given_x_sentence, axis=1)

        # cost and gradients and learning rate
        lr = T.scalar('lr')
        error= T.mean(T.neq(y_pred, y_sentence))
        sentence_nll = -T.mean(T.log(p_y_given_x_sentence)
                               [T.arange(x.shape[0]), y_sentence])
        sentence_gradients = T.grad(sentence_nll, self.params)
        sentence_updates = OrderedDict((p, p - lr*g)
                                       for p, g in
                                       zip(self.params, sentence_gradients))

        # theano functions to compile
        self.classify = theano.function(inputs=[x], outputs=y_pred)
        self.classify_error=theano.function(inputs=[x,y_sentence], outputs=error, allow_input_downcast=True )
        self.sentence_train = theano.function(inputs=[x, y_sentence, lr],
                                              outputs=sentence_nll,
                                              updates=sentence_updates,
                                              allow_input_downcast=True)

    
    
    def save(self, folder):
        for param in self.params:
            numpy.save(os.path.join(folder,
                       param.name + '.npy'), param.get_value())

    def load(self, folder):
        for param in self.params:
            param.set_value(numpy.load(os.path.join(folder,
                            param.name + '.npy')))
    



#TODO: build and train a MLP to learn parity function
def test_mlp_parity(n_bit=8, n_hidden=400, learning_rate=0.01,
    L1_reg=0.00, L2_reg=0.001, n_hiddenLayers=1, n_epochs=400, batch_size=100, verbose= True):
    
    # generate datasets
    train_set = gen_parity_pair(n_bit, 1000)
    valid_set = gen_parity_pair(n_bit, 500)
    test_set  = gen_parity_pair(n_bit, 100)

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = shared_dataset(train_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    test_set_x, test_set_y = shared_dataset(test_set)

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] // batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] // batch_size

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print('... building the model')

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    rng = numpy.random.RandomState(1234)

    # construct the MLP class

    
    
    classifier = myMLP(
        rng=rng,
        input=x,
        n_in=n_bit,
        n_hidden=n_hidden,
        n_hiddenLayers=n_hiddenLayers,
        n_out=2)

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = (
        classifier.negative_log_likelihood(y)
        + L1_reg * classifier.L1
        + L2_reg * classifier.L2_sqr
    )

    # compiling a Theano function that computes the mistakes that are made
    # by the model on a minibatch
    test_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: test_set_x[index * batch_size:(index + 1) * batch_size],
            y: test_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    validate_model = theano.function(
        inputs=[index],
        outputs=classifier.errors(y),
        givens={
            x: valid_set_x[index * batch_size:(index + 1) * batch_size],
            y: valid_set_y[index * batch_size:(index + 1) * batch_size]
        }
    )

    # compute the gradient of cost with respect to theta (sotred in params)
    # the resulting gradients will be stored in a list gparams
    gparams = [T.grad(cost, param) for param in classifier.params]

    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs

    # given two lists of the same length, A = [a1, a2, a3, a4] and
    # B = [b1, b2, b3, b4], zip generates a list C of same size, where each
    # element is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3), (a4, b4)]
    updates = [
        (param, param - learning_rate * gparam)
        for param, gparam in zip(classifier.params, gparams)
    ]

    # compiling a Theano function `train_model` that returns the cost, but
    # in the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(
        inputs=[index],
        outputs=cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]
        }
    )

    ###############
    # TRAIN MODEL #
    ###############
    print('... training')
    train_nn(train_model=train_model, validate_model=validate_model, test_model=test_model,
        n_train_batches=n_train_batches, n_valid_batches=n_valid_batches,
        n_test_batches=n_test_batches,n_bit=n_bit,
        n_epochs=n_epochs,verbose = True)

    
    

    
#TODO: build and train a RNN to learn parity function
def test_rnn_parity(**kwargs):
    """
    Wrapper function for training and testing RNNSLU

    :type fold: int
    :param fold: fold index of the ATIS dataset, from 0 to 4.

    :type lr: float
    :param lr: learning rate used (factor for the stochastic gradient.

    :type nepochs: int
    :param nepochs: maximal number of epochs to run the optimizer.

    :type win: int
    :param win: number of words in the context window.

    :type nhidden: int
    :param n_hidden: number of hidden units.

    :type emb_dimension: int
    :param emb_dimension: dimension of word embedding.

    :type verbose: boolean
    :param verbose: to print out epoch summary or not to.

    :type decay: boolean
    :param decay: decay on the learning rate if improvement stop.

    :type savemodel: boolean
    :param savemodel: save the trained model or not.

    :type normal: boolean
    :param normal: normalize word embeddings after each update or not.

    :type folder: string
    :param folder: path to the folder where results will be stored.

    """


    # process input arguments
    param = {
        'n_bit': 8,
        'lr': 0.1970806646812754,
        'verbose': True,
        'decay': True,
        'win': 7,
        'nhidden': 3,
        'seed': 100,
        'nepochs': 60,
        'savemodel': True,
        'is_rnn':1,
        'folder':'../result_rnn'
        }

    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))

    # create result folder if not exists
    check_dir(param['folder'])

    # load the dataset
    print('... loading the dataset')
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])
    
    
    train_x, train_y = gen_parity_pair(param['n_bit'], 1000,param['is_rnn'])
    valid_x, valid_y = gen_parity_pair(param['n_bit'], 500, param['is_rnn'])
    test_x, test_y  = gen_parity_pair(param['n_bit'], 100,param['is_rnn'])

    vocsize = 2
    nclasses = 2
    nsentences = 1000

    
    print('... building the model')
    rnn = RNN(
        nh=param['nhidden'],
        nc=nclasses,
        ne= vocsize,
        cs=param['win']
    )

    # train with early stopping on validation set
    print('... training')
    best_f1 = -numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):

        param['ce'] = e
        tic = timeit.default_timer()

        for i, (x, y) in enumerate(zip(train_x, train_y)):
            rnn.train(numpy.asarray(x), numpy.asarray(y), param['win'], param['clr'])
            sys.stdout.flush()

        # evaluation and prediction
        # predictions_test = [map(lambda x: idx2label[x],
        #                     rnn.classify(numpy.asarray(
        #                     contextwin(x, param['win'])).astype('int32')))
        #                     for x in test_lex]
        
        predictions_valid = numpy.asarray([rnn.classify(numpy.asarray( 
                            contextwin(x, param['win'])).astype('int32')) 
                            for x in valid_x ])
        predictions_test = numpy.asarray([rnn.classify(numpy.asarray( 
                           contextwin(x, param['win'])).astype('int32'))
                           for x in test_x ])

        
        res_valid = numpy.mean((valid_y[:,param['n_bit']-1]==predictions_valid[:, param['n_bit']-1]))
        res_test = numpy.mean((test_y[:,param['n_bit']-1]==predictions_test[:, param['n_bit']-1]))


        if res_valid > best_f1:
            
            if param['savemodel']:
                rnn.save(param['folder'])

            best_rnn = copy.deepcopy(rnn)
            best_f1 = res_valid

            if param['verbose']:
                print('NEW BEST: epoch '+str(e),
                      'Validation Error '+ str((1.0-res_valid)*100) + '%',
                      'Test Error '+str((1.0-res_test)*100) + '%')

            param['vf1'], param['tf1'] = res_valid, res_test
            param['be'] = e

        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.5
            rnn = best_rnn

        if param['clr'] < 1e-5:
            break

    print('Best Result : epoch '+ str(param['be']),
                      'Validation Error '+ str((1.0-param['vf1'])*100) + '%',
                      'Best Test Error '+str((1.0-param['tf1'])*100) + '%', 'on '+str(param['n_bit'])+' bit sequences')

    


    
    
#TODO: build and train a LSTM to learn parity function
def test_lstm_parity(**kwargs):

    param = {
        'n_bit':8,
        'lr': 0.1,
        'verbose': True,
        'decay': True,
        'nhidden': 2,
        'seed': 345,
        'nepochs': 80, 
        'savemodel':True,
        'is_rnn':1,
        'folder':'../result_lstm'      
        }

    param_diff = set(kwargs.keys()) - set(param.keys())
    if param_diff:
        raise KeyError("invalid arguments:" + str(tuple(param_diff)))
    param.update(kwargs)

    if param['verbose']:
        for k,v in param.items():
            print("%s: %s" % (k,v))
    
    # generate datasets
    train_set = gen_parity_pair(param['n_bit'], 1000, param['is_rnn'])
    valid_set = gen_parity_pair(param['n_bit'], 500, param['is_rnn'])
    test_set  = gen_parity_pair(param['n_bit'], 100, param['is_rnn'])

    # Convert raw dataset to Theano shared variables.
    train_set_x, train_set_y = train_set
    valid_set_x, valid_set_y = valid_set
    test_set_x, test_set_y = test_set

    train_set_x = numpy.asarray(train_set_x)
    train_set_y = numpy.asarray(train_set_y).astype('int32')
    valid_set_x = numpy.asarray(valid_set_x)
    valid_set_y = numpy.asarray(valid_set_y).astype('int32')
    test_set_x = numpy.asarray(test_set_x)
    test_set_y = numpy.asarray(test_set_y).astype('int32')

    
    # instanciate the model
    numpy.random.seed(param['seed'])
    random.seed(param['seed'])

    print('... building the model')
    
    nclasses = 2
    nsentences=1000

    lstm = LSTM(
        nh=param['nhidden'],
        nc=nclasses)

    # train with early stopping on validation set
    print('... training')
    best_valid_error = numpy.inf
    param['clr'] = param['lr']
    for e in range(param['nepochs']):

        # shuffle
        #shuffle([train_lex, train_ne, train_y], param['seed'])

        param['ce'] = e
        tic = timeit.default_timer()
        

        for i, (x, y) in enumerate(zip(train_set_x, train_set_y)):
            lstm.sentence_train(x.reshape(x.shape[0],1), y, param['clr'])
            # print('[learning] epoch %i >> %2.2f%%' % (
            #     e, (i + 1) * 100. / nsentences))
            # print('completed in %.2f (sec) <<\r' % (timeit.default_timer() - tic))
            sys.stdout.flush()

        
        test_error = [lstm.classify_error(x.reshape(x.shape[0],1),y) for i, (x, y) in enumerate(zip(test_set_x, test_set_y))]
        test_error = numpy.mean(test_error)
        valid_error = [lstm.classify_error(x.reshape(x.shape[0],1),y) for i, (x, y) in enumerate(zip(valid_set_x, valid_set_y))]
        valid_error = numpy.mean(valid_error)
        


        




        if valid_error < best_valid_error:

            if param['savemodel']:
                lstm.save(param['folder'])

            best_lstm = copy.deepcopy(lstm)
            best_valid_error = valid_error
            best_test_error = test_error
            if param['verbose']:
                print('NEW BEST: epoch', e, 'validation error', best_valid_error,
                'test error', best_test_error)

            
            param['be'] = e

        else:
            if param['verbose']:
                print('')

        # learning rate decay if no improvement in 10 epochs
        if param['decay'] and abs(param['be']-param['ce']) >= 10:
            param['clr'] *= 0.75


        if param['clr'] < 1e-6:
            print 'param[''clr''] < 1e-5'
            break

    if param['verbose']:
        print('BEST result : epoch', e,
        'best validation error', best_valid_error,
        'test error', best_test_error, 'on '+str(param['n_bit'])+' bit sequences')
    
if __name__ == '__main__':
    test_mlp_parity()

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn, rnn_cell, nn_ops
from tensorflow.python.ops import math_ops, array_ops
from sys import stderr
import re
import nltk
import sys
import copy

def rnn_decoder(decoder_inputs, initial_state, cell, Ws, bs, vocab_size, batch_size, memory_dim):
    with tf.variable_scope("rnn_decoder"):
        state = initial_state
        outputs = []
        for i, inp in enumerate(decoder_inputs):
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            output, state = cell(inp, state)
            output = array_ops.reshape(output,[1,-1])
            output = tf.slice(output,[0,0],[1,memory_dim])
            y = tf.nn.softmax(tf.matmul(output, Ws) + bs)
            outputs.append(y)
    return outputs, state


def loss (examples, targets):
    log_perp_list = []
    for example, target in zip(examples, targets):
        target = array_ops.reshape(target, [-1])
        example = array_ops.reshape(example, [-1])
        const = tf.constant(1e-12)
        crossent = tf.reduce_mean(-tf.reduce_sum((example+const)*tf.log(target+const)))
        log_perp_list.append(crossent)
    log_perps = math_ops.add_n(log_perp_list)
    total_size = len(targets)
    total_size += 1e-12
    log_perps /= total_size
    return log_perps

 
class BaseCreativeMap(object):
    def __init__(self,
                 seq_length,
                 vocab_size,
                 stack_dimension,
                 batch_size):
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.memory_dim = vocab_size

        self.enc_inp = [tf.placeholder(tf.float32, shape=(vocab_size,batch_size), name="enc_inp%i" % t)
                        for t in range(seq_length)]

        self.dec_inp = self.enc_inp[:-1] + [tf.zeros_like(self.enc_inp[0], dtype=np.float32, name="GO")]


        single_enc_cell = rnn_cell.LSTMCell(self.memory_dim, state_is_tuple=False)
        self.enc_cell = rnn_cell.MultiRNNCell([single_enc_cell]*stack_dimension, state_is_tuple=True)
        _, encoder_state = rnn.rnn(self.enc_cell, self.enc_inp, dtype=tf.float32)

        single_dec_cell = rnn_cell.LSTMCell(self.memory_dim, state_is_tuple=False)
        self.dec_cell = rnn_cell.MultiRNNCell([single_dec_cell]*stack_dimension, state_is_tuple=True)


        self.Ws = tf.Variable(tf.random_uniform([self.memory_dim, self.vocab_size],0,0.1))
        self.bs = tf.Variable(tf.random_uniform([self.vocab_size],-0.1,0.1))

        
        self.dec_outputs, self.dec_state = rnn_decoder(self.dec_inp, encoder_state, self.dec_cell, self.Ws, self.bs, vocab_size, batch_size, self.memory_dim)

        self.labels = [tf.placeholder(tf.float32, [vocab_size, batch_size], name = 'LABEL%i' % t)
                        for t in range(seq_length)]
        self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
                        for labels_t in self.labels]
        self.loss = loss(self.labels, self.dec_outputs)
        
        self.train_op = tf.train.AdamOptimizer(1e-3).minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())


    def __setitem(self,X,Y):
        X= np.transpose(X,(1,2,0))
        X= X[::-1,:,:]
        Y= np.transpose(Y,(1,2,0))
        feed_dict = {self.enc_inp[t]: X[t] for t in range(self.seq_length)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.seq_length)})
        _, loss_t = self.sess.run([self.train_op, self.loss], feed_dict)
        sys.stdout.flush()
        return loss_t

    
    def __getitem__(self,X):
        X= np.transpose(X,(1,2,0))
        X= X[::-1,:,:]
        feed_dict = {self.enc_inp[t]: X[t] for t in range(self.seq_length)}
        dec_outputs_batch = self.sess.run(self.dec_outputs, feed_dict)
        return dec_outputs_batch

    
    def save (self, filename):
        saver = tf.train.Saver()
        saver.save (self.sess, filename)

    def load (self, filename):
        saver = tf.train.Saver()
        saver.restore (self.sess, filename)

    def __valid_inputs_or_throw (self, X):
        return True
        
    def train(self, training_set_input, training_set_output, batch_size):
        trace_max = open ('trace_max','w')
        trace_min = open ('trace_min','w')
        training_set_size = len(training_set_input)
        total_number_of_batches = training_set_size/batch_size
        if total_number_of_batches < 50:
            stderr.write('Warning: Number of batches is too small! ' 
                         'Please increase the training set '
                         'or decrease the batch size.')
        for r in range(30000):
            training_vect_loss = []
            for j in range(total_number_of_batches):
                loss = self.__setitem(
                    [training_set_input[i+j] for i in range(batch_size)], 
                    [training_set_output[i+j] for i in range(batch_size)]
                )
                training_vect_loss.append(loss)
            min_loss = min(training_vect_loss)
            max_loss = max(training_vect_loss)
            trace_max.write(str(max_loss)+'\n')
            trace_min.write(str(min_loss)+'\n')
            trace_min.flush()
            trace_max.flush()
            if max_loss < 0.01 and max_loss - min_loss < 0.005:
                break
            if max_loss < -130 and max_loss - min_loss < 1:
                break
        training_set_input = []
        training_set_output = []



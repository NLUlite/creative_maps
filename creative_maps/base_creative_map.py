import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from sys import stderr
import re
import nltk
import sys
import gensim

class BaseCreativeMap(object):
    def __init__(self,
                 seq_length,
                 vocab_size,
                 stack_dimension,
                 embedding_matrix = None):
        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = 300
        self.memory_dim = 100
        self.enc_inp = [tf.placeholder(tf.int32, shape=(None,),
                                       name="inp%i" % t)
                        for t in range(seq_length)]
        self.labels = [tf.placeholder(tf.int32, shape=(None,),
                                      name="labels%i" % t)
                       for t in range(seq_length)]
        self.weights = [tf.ones_like(labels_t, dtype=tf.float32)
                        for labels_t in self.labels]
        self.dec_inp = ([tf.zeros_like(self.enc_inp[0], dtype=np.int32, name="GO")]
                        + self.enc_inp[:-1])
        single_cell = rnn_cell.LSTMCell(self.memory_dim, state_is_tuple=False)
        self.cell = rnn_cell.MultiRNNCell([single_cell]*stack_dimension, state_is_tuple=True)
        self.dec_outputs, self.dec_memory = seq2seq.embedding_rnn_seq2seq(
            self.enc_inp, self.dec_inp, self.cell, self.vocab_size, self.vocab_size, self.embedding_dim)
        self.loss = seq2seq.sequence_loss(self.dec_outputs, self.labels, self.weights, self.vocab_size)
        self.learning_rate = 0.05
        self.momentum = 0.9
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    self.momentum)
        self.train_op = self.optimizer.minimize(self.loss)

        if embedding_matrix is not None:
            with tf.device('/gpu:0') and tf.variable_scope("embedding_rnn_seq2seq/RNN/EmbeddingWrapper", reuse=True):
                embedding_encoder = tf.get_variable("embedding",[vocab_size,self.embedding_dim])

            with tf.device('/gpu:0') and tf.variable_scope("embedding_rnn_seq2seq/embedding_rnn_decoder", reuse=True):
                embedding_decoder = tf.get_variable("embedding",[vocab_size,self.embedding_dim])

        self.sess.run(tf.initialize_all_variables())
        tf.scalar_summary("loss", self.loss)
        self.summary_op = tf.merge_all_summaries()

        if embedding_matrix is not None:
            self.sess.run(embedding_encoder.assign(embedding_matrix))
            self.sess.run(embedding_decoder.assign(embedding_matrix))

    def __setitem(self,X,Y):
        X = np.fliplr(X)
        X = np.array(X).T
        Y = np.array(Y).T
        feed_dict = {self.enc_inp[t]: X[t] for t in range(self.seq_length)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.seq_length)})

        _, loss_t, summary = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict)
        sys.stdout.flush()
        return loss_t, summary

    
    def __getitem__(self,X):
        X = np.fliplr(X)
        X = np.array(X).T
        feed_dict = {self.enc_inp[t]: X[t] for t in range(self.seq_length)}
        dec_outputs_batch = self.sess.run(self.dec_outputs, feed_dict)
        return np.array ([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]).T

    
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
                loss, _ = self.__setitem(
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
            if max_loss < 0.05 and max_loss - min_loss < 0.01:
                break
        training_set_input = []
        training_set_output = []



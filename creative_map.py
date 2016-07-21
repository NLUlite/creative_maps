import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
from sys import stderr
import re
import sys
        
class CreativeMap:
    def __init__(self, seq_length, vocab_size, stack_dimension):
        self.sess = tf.InteractiveSession()

        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = 1
        self.memory_dim = 500
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
        single_cell = rnn_cell.GRUCell(self.memory_dim)
        self.cell = rnn_cell.MultiRNNCell([single_cell]*stack_dimension)
#        self.dec_outputs, self.dec_memory = seq2seq.embedding_tied_rnn_seq2seq(
#            self.enc_inp, self.dec_inp, self.cell, self.vocab_size, self.embedding_size)
        self.dec_outputs, self.dec_memory = seq2seq.embedding_rnn_seq2seq(
            self.enc_inp, self.dec_inp, self.cell, self.vocab_size, self.vocab_size, self.embedding_dim)
        self.loss = seq2seq.sequence_loss(self.dec_outputs, self.labels, self.weights, self.vocab_size)
        self.learning_rate = 0.05
        self.momentum = 0.9
        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate,
                                                    self.momentum)
        self.train_op = self.optimizer.minimize(self.loss)
        self.sess.run(tf.initialize_all_variables())
        tf.scalar_summary("loss", self.loss)
        self.summary_op = tf.merge_all_summaries()
        
    def __setitem__(self,X,Y):
        X = np.fliplr(X)
        X = np.array(X).T
        Y = np.array(Y).T
        feed_dict = {self.enc_inp[t]: X[t] for t in range(self.seq_length)}
        feed_dict.update({self.labels[t]: Y[t] for t in range(self.seq_length)})

        _, loss_t, summary = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict)
#        print loss_t
        sys.stdout.flush()
        return loss_t, summary

    
    def __getitem__(self,X):
        X = np.fliplr(X)
        X = np.array(X).T
        feed_dict = {self.enc_inp[t]: X[t] for t in range(self.seq_length)}
        dec_outputs_batch = self.sess.run(self.dec_outputs, feed_dict)
        return np.array ([logits_t.argmax(axis=1) for logits_t in dec_outputs_batch]).T

    
    def save (self, filename):
        # save the model
        saver = tf.train.Saver()
        saver.save (self.sess, filename)

    def load (self, filename):
        # save the model
        saver = tf.train.Saver()
        saver.restore (self.sess, filename)


class StringCreativeMap (CreativeMap):
    def __init__(self, max_seq_length, stack_dimension=2, batch_size=1):
        self.END_CHAR = '\0'
        self.FILLING_CHAR = ' '
        self.max_seq_length = max_seq_length
        vocab_size = 256
        CreativeMap.__init__(self, max_seq_length, vocab_size, stack_dimension)
        self.training_set_input = []
        self.training_set_output = []
        self.batch_size = batch_size
        
    def __convert_string_to_ints (self, X):
        if X[-1] != self.END_CHAR:
            X = X + self.END_CHAR
        X = X.ljust(self.max_seq_length)
        X = [ord(X[i]) for i in range (len(X))]
        return X

    def __convert_ints_to_string (self, X):
        X = ''.join([chr(X[i]) for i in range (len(X))])
        X, _, _ = X.partition(self.END_CHAR)
        return X

    def __valid_inputs_or_throw (self, X):
        if not isinstance(X,str):
            raise ValueError ("The arguments to setitem must be strings.")
        X = X.split(self.END_CHAR)[0]
        if len(X) >= self.max_seq_length:
            raise ValueError ("The arguments must be shorter than max_seq_length.")

    def __setitem__(self,X,Y):
        self.__valid_inputs_or_throw (X)
        self.__valid_inputs_or_throw (Y)
        X = self.__convert_string_to_ints (X)
        Y = self.__convert_string_to_ints (Y)
        self.training_set_input.append(X)
        self.training_set_output.append(Y)

    def __getitem__(self,X):
        self.__valid_inputs_or_throw (X)
        X = self.__convert_string_to_ints (X)
        output_in_integers = CreativeMap.__getitem__(self, [X])
        return self.__convert_ints_to_string (output_in_integers[0])
        
    def train(self):
        training_set_size = len(self.training_set_input)
        total_number_of_batches = training_set_size/self.batch_size
        if total_number_of_batches < 50:
            stderr.write('Warning: Number of batches is too small! ' 
                         'Please increase the training set '
                         'or decrease the batch size.')
        for j in range(total_number_of_batches):
            CreativeMap.__setitem__(
                self, 
                [self.training_set_input[i+j] for i in range(self.batch_size)], 
                [self.training_set_output[i+j] for i in range(self.batch_size)]
            )
        self.training_set_input = []
        self.training_set_output = []
        
class WordsCreativeMap (CreativeMap):
    def __init__(self, max_seq_length, input_vect, output_vect, stack_dimension=2, batch_size=1):
        self.END_CHAR = 0
        self.FILLING_CHAR = ' '
        self.max_seq_length = max_seq_length
        vocab_size = len(input_vect)+10
        CreativeMap.__init__(self, max_seq_length, vocab_size, stack_dimension)
        self.training_set_input = []
        self.training_set_output = []
        self.batch_size = batch_size
        self.input_vect = input_vect
        self.output_vect = output_vect

    def __get_int_from_list(self, vect, word):
        index = 0
        try:
            index = vect.index(word)
        except:
            index = 1
        return index
        
    def __convert_input_string_to_ints (self, X):
        words = re.findall(r'\w+', X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(self.input_vect,words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        for i in range(self.max_seq_length - len(X)):
            X = X + [0]
        return X

    def __convert_output_string_to_ints (self, X):
        words = re.findall(r'\w+', X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(self.output_vect,words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        for i in range(self.max_seq_length - len(X)):
            X = X + [0]
        return X

    
    def __convert_output_ints_to_string (self, X):
        X = ' '.join([self.output_vect[X[i]] for i in range (len(X))])
#        X, _, _ = X.partition(self.END_CHAR)
        return X

    def __valid_inputs_or_throw (self, X):
#        if not isinstance(X,str):
#            raise ValueError ("The arguments to setitem must be strings.")
#        X = X.split(self.END_CHAR)[0]
#        if len(X) >= self.max_seq_length:
#            raise ValueError ("The arguments must be shorter than max_seq_length.")
        return True

    def __setitem__(self,X,Y):
        self.__valid_inputs_or_throw (X)
        self.__valid_inputs_or_throw (Y)
        X = self.__convert_input_string_to_ints (X)
        Y = self.__convert_output_string_to_ints (Y)
        self.training_set_input.append(X)
        self.training_set_output.append(Y)

    def __getitem__(self,X):
        self.__valid_inputs_or_throw (X)
        X = self.__convert_input_string_to_ints (X)
        output_in_integers = CreativeMap.__getitem__(self, [X])
        return self.__convert_output_ints_to_string (output_in_integers[0])
        
    def train(self):
        training_set_size = len(self.training_set_input)
        total_number_of_batches = training_set_size/self.batch_size
        if total_number_of_batches < 50:
            stderr.write('Warning: Number of batches is too small! ' 
                         'Please increase the training set '
                         'or decrease the batch size.')
        for r in range(3000):
            training_vect_loss = []
            for j in range(total_number_of_batches):
                loss, _ = CreativeMap.__setitem__(
                    self, 
                    [self.training_set_input[i+j] for i in range(self.batch_size)], 
                    [self.training_set_output[i+j] for i in range(self.batch_size)]
                )
                training_vect_loss.append(loss)
            print max(training_vect_loss)
            if max(training_vect_loss) < 0.05:
                break
        self.training_set_input = []
        self.training_set_output = []



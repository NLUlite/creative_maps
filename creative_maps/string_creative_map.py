import numpy as np
from base_creative_map import BaseCreativeMap        

ZERO = 0

def get_vector(i,vocab_size):
    vect = [ZERO]*vocab_size
    vect[i] = 1.
    return vect



class StringCreativeMap (BaseCreativeMap):
    def __init__(self, max_seq_length, stack_dimension=1, batch_size=1):
        self.END_CHAR = '\0'
        self.FILLING_CHAR = ' '
        self.max_seq_length = max_seq_length
        self.vocab_size = 256
        BaseCreativeMap.__init__(self, max_seq_length, self.vocab_size, stack_dimension, batch_size)
        self.training_set_input = []
        self.training_set_output = []
        self.batch_size = batch_size
        
    def __convert_string_to_ints (self, sentence):
        integer_vector = [get_vector(ord(sentence[i]),self.vocab_size) for i in range (len(sentence))]
        max_length = self.max_seq_length
        if len(integer_vector) >= max_length:
            integer_vector = integer_vector[:max_length]
        for i in range(max_length - len(integer_vector)):
            integer_vector.append(get_vector(0,self.vocab_size))
        return integer_vector

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
        dec_output= BaseCreativeMap.__getitem__(self, [X])
        output_in_integers = np.array([logits_t.argmax(axis=1) for logits_t in dec_output]).T
        return self.__convert_ints_to_string (output_in_integers[0])
        
    def train (self):
        super(StringCreativeMap,self).train(self.training_set_input, self.training_set_output, self.batch_size)

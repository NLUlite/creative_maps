from base_creative_map import BaseCreativeMap
import gensim
import nltk


import numpy as np

class EmbeddedCreativeMap (BaseCreativeMap):
    def __init__(self, max_seq_length, model, stack_dimension=1, batch_size=1):
        self.model = model
        self.max_seq_length = max_seq_length
        self.vocab_size = self.model.syn0.shape[1]
        BaseCreativeMap.__init__(self, max_seq_length, self.vocab_size, stack_dimension, batch_size)
        self.training_set_input = []
        self.training_set_output = []
        self.batch_size = batch_size


    def __get_int_from_list(self, word):
        index = self.model['.']
        try:
            index = self.model[word]
        except:
            index = self.model['.']
        return index

    def __convert_input_string_to_ints (self, X):
        tokenizer = nltk.tokenize.TweetTokenizer()
        words = tokenizer.tokenize(X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        for i in range(self.max_seq_length - len(X)):
            X.append(self.model['.'])
        return X

    def __convert_output_string_to_ints (self, X):
        tokenizer = nltk.tokenize.TweetTokenizer()
        words = tokenizer.tokenize(X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        for i in range(self.max_seq_length - len(X)):
            X.append(self.model['.'])
        return X

    
    def __convert_output_ints_to_string (self, X):
        X = ' '.join([self.model.most_similar([X[i]],[],topn=1)[0][0] for i in range(len(X))])
        return X

    def __setitem__(self,X,Y):
        X = self.__convert_input_string_to_ints (X)
        Y = self.__convert_output_string_to_ints (Y)
        self.training_set_input.append(X)
        self.training_set_output.append(Y)

    def __getitem__(self,X):
        X = self.__convert_input_string_to_ints (X)
        output_in_integers = BaseCreativeMap.__getitem__(self, [X])
        output_in_integers = np.transpose(output_in_integers,[1,0,2])
        return self.__convert_output_ints_to_string (output_in_integers[0])
        
    def train (self):
        super(EmbeddedCreativeMap,self).train(self.training_set_input, self.training_set_output, self.batch_size)

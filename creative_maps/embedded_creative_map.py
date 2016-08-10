from base_creative_map import BaseCreativeMap
import nltk


class EmbeddedCreativeMap (BaseCreativeMap):
    def __init__(self, max_seq_length, model, stack_dimension=2, batch_size=1):
        self.END_CHAR = 1
        self.FILLING_CHAR = 0
        self.model = model
        self.max_seq_length = max_seq_length
        vocab_size = len(self.model.vocab)
        BaseCreativeMap.__init__(self, max_seq_length, vocab_size, stack_dimension, self.model.syn0)
        self.training_set_input = []
        self.training_set_output = []
        self.batch_size = batch_size


    def __get_int_from_list(self, word):
        index = 0
        try:
            index = self.model.vocab[word].index
        except:
            index = 1
        return index
        
    def __convert_input_string_to_ints (self, X):
        tokenizer = nltk.tokenize.TweetTokenizer()
        words = tokenizer.tokenize(X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        X.append(self.END_CHAR)
        for i in range(self.max_seq_length - len(X)):
            X = X + [self.FILLING_CHAR]
        return X

    def __convert_output_string_to_ints (self, X):
        tokenizer = nltk.tokenize.TweetTokenizer()
        words = tokenizer.tokenize(X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        X.append(self.END_CHAR)
        for i in range(self.max_seq_length - len(X)):
            X = X + [self.FILLING_CHAR]
        return X

    
    def __convert_output_ints_to_string (self, X):
        X = ' '.join([self.model.index2word[X[i]] for i in range (len(X))])
        return X

    def __setitem__(self,X,Y):
        X = self.__convert_input_string_to_ints (X)
        Y = self.__convert_output_string_to_ints (Y)
        self.training_set_input.append(X)
        self.training_set_output.append(Y)

    def __getitem__(self,X):
        X = self.__convert_input_string_to_ints (X)
        output_in_integers = BaseCreativeMap.__getitem__(self, [X])
        return self.__convert_output_ints_to_string (output_in_integers[0])
        
    def train (self):
        super(EmbeddedCreativeMap,self).train(self.training_set_input, self.training_set_output, self.batch_size)

from base_creative_map import BaseCreativeMap
import nltk

class WordsCreativeMap (BaseCreativeMap):
    def __init__(self, max_seq_length, input_vect, output_vect, stack_dimension=2, batch_size=1):
        self.END_CHAR = 0
        self.FILLING_CHAR = ' '
        self.max_seq_length = max_seq_length
        vocab_size = len(input_vect)+10
        BaseCreativeMap.__init__(self, max_seq_length, vocab_size, stack_dimension)
        self.training_set_input = []
        self.training_set_output = []
        self.batch_size = batch_size
        self.input_vect = input_vect
        self.output_vect = output_vect
        self.tokenizer = nltk.tokenize.TweetTokenizer()

    def __get_int_from_list(self, vect, word):
        index = 0
        try:
            index = vect.index(word)
        except:
            index = 1
        return index
        
    def __convert_input_string_to_ints (self, X):
        words = self.tokenizer.tokenize(X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(self.input_vect,words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        for i in range(self.max_seq_length - len(X)):
            X = X + [0]
        return X

    def __convert_output_string_to_ints (self, X):
        words = self.tokenizer.tokenize(X)
        words = [word.lower() for word in words]
        X = [self.__get_int_from_list(self.output_vect,words[i]) for i in range (len(words))]
        if len(X) >= self.max_seq_length:
            X = X[:self.max_seq_length-1]
        for i in range(self.max_seq_length - len(X)):
            X = X + [0]
        return X
    
    def __convert_output_ints_to_string (self, X):
        X = ' '.join([self.output_vect[X[i]] for i in range (len(X))])
        return X

    def __setitem__(self,X,Y):
        X = self.__convert_input_string_to_ints (X)
        Y = self.__convert_output_string_to_ints (Y)
        self.training_set_input.append(X)
        self.training_set_output.append(Y)

    def __getitem__(self,X):
        X = self.__convert_input_string_to_ints (X)
        output_in_integers = CreativeMap.__getitem__(self, [X])
        return self.__convert_output_ints_to_string (output_in_integers[0])
        
    def train (self):
        super(WordsCreativeMap,self).train(self.training_set_input, self.training_set_output, self.batch_size)

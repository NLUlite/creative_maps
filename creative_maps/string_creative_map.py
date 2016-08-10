from base_creative_map import BaseCreativeMap        

class StringCreativeMap (BaseCreativeMap):
    def __init__(self, max_seq_length, stack_dimension=2, batch_size=1):
        self.END_CHAR = '\0'
        self.FILLING_CHAR = ' '
        self.max_seq_length = max_seq_length
        vocab_size = 256
        BaseCreativeMap.__init__(self, max_seq_length, vocab_size, stack_dimension)
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
        output_in_integers = BaseCreativeMap.__getitem__(self, [X])
        return self.__convert_ints_to_string (output_in_integers[0])
        
    def train (self):
        super(StringCreativeMap,self).train(self.training_set_input, self.training_set_output, self.batch_size)

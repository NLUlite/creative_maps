from creative_map import WordsCreativeMap        

max_seq_length = 30

cmap = WordsCreativeMap (max_seq_length,
                         [l.strip('\n') for l in open('en_words').readlines()],
                         [l.strip('\n') for l in open('nl_words').readlines()])
cmap.load ("strings_words.nn")

inp = 'Resumption of the session'
print inp, '->', cmap [inp]



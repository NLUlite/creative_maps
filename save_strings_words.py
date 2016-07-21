from creative_map import WordsCreativeMap

max_seq_length = 30
        
cmap = WordsCreativeMap (max_seq_length,
                         [l.strip('\n') for l in open('en_words').readlines()],
                         [l.strip('\n') for l in open('nl_words').readlines()],
                         3, 10)
en_lines = open('../europarl-v7.nl-en.en').readlines()
nl_lines = open('../europarl-v7.nl-en.nl').readlines()

for t in range(500000):
    cmap[en_lines[t]] = nl_lines[t]

cmap.train ()
cmap.save ("strings_words.nn")


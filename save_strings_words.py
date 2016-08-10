from creative_maps.words_creative_map import WordsCreativeMap

max_seq_length = 30
        
cmap = WordsCreativeMap (max_seq_length,
                         [l.strip('\n') for l in open('data/en_words').readlines()],
                         [l.strip('\n') for l in open('data/nl_words').readlines()],
                         2, 2)
en_lines = open('data/europarl-v7.nl-en.en').readlines()
nl_lines = open('data/europarl-v7.nl-en.nl').readlines()

for t in range(6):
    cmap[en_lines[t]] = nl_lines[t]

cmap.train ()
cmap.save ("strings_words.nn")


from creative_maps.embedded_creative_map import EmbeddedCreativeMap
import nltk, gensim
import codecs
max_seq_length = 30
        
cmap = EmbeddedCreativeMap (max_seq_length,
                            gensim.models.Word2Vec.load_word2vec_format('data/deps.words.txt',binary=False),
                            1, 1)
en_lines = codecs.open('data/ordered_lines.txt', encoding='utf8', errors='ignore').readlines()

for t in range(5):
    from_line = str(en_lines[t])
    to_line = str(en_lines[t+1])
    tokenizer = nltk.tokenize.TweetTokenizer()
    num_from = len(tokenizer.tokenize(from_line))
    num_to = len(tokenizer.tokenize(to_line))

    if num_from < max_seq_length and num_to < max_seq_length:        
        cmap[from_line] = to_line

try:
    cmap.train ()
except:
    print 'Training stopped: '

cmap.save ("strings_embedded.nn")


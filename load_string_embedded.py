from creative_maps.embedded_creative_map import EmbeddedCreativeMap        
import gensim

max_seq_length = 30

cmap = EmbeddedCreativeMap (max_seq_length,
                            gensim.models.Word2Vec.load_word2vec_format('data/deps.words.txt',binary=False))
cmap.load ("strings_embedded.nn")

inp = 'Your orders, Mr Vereker?'
print inp, '->', cmap [inp]



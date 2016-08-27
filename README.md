## Creative maps

Creative maps are a simple way to create sequence to sequence learning
with Tensorflow. RNNs are used as associative
containers that map a string to another one.

One can train the network as in the following

```bash
cmap ["hi"] = "ciao"
cmap ["good morning"] = "buongiorno"
```

and then retrieve the results with

```
pring cmap ["hi"]
```

The network learns the corresponence between Hi and Ciao. Moreover, if
the trainset is large enough, it starts recognizing patterns in the
data and provide answers to strings it has not been trained for.

## Example for character to character translation

Run the following file to train the creative map with two instances

```bash 
python save_strings.py 
```

The code in this script reads as

```python
from creative_map import StringCreativeMap

max_seq_length = 10
        
cmap = StringCreativeMap (max_seq_length)
cmap["hello"] = "world"
cmap["ciao"]  = "mondo"
cmap.train ()
cmap.save ("strings.nn")
```

After this step one can run

```bash
python load_string.py
```

which brings the result

```bash
hello world
ciao mondo
---
hi moodd
hallo world
bonjour moodd
```

The first two lines are the reply based on the trained data. The last
three are the results of untrained queries. The nework is able to
understand that "hallo" and "hello" are related, and give the same
answer. In the other two cases the network simply creates a reply from
scratch - based on the only two training examples.


## Example for translation with word embeddings

There is an additional map that uses embeddings. The following example
uses the word embeddings deps.words.txt, the first one in these
[https://github.com/3Top/word2vec-api#where-to-get-a-pretrained-models](pre-trained
models). 


```python
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
    
    cmap[from_line] = to_line

cmap.train ()

cmap.save ("strings_embedded.nn")
```

The previous lines train the map using the first 6 lines of a
sequence of movie scripts, as provided in
[www.cs.cornell.edu/~cristian/Cornell_Movie-Dialogs_Corpus.html](the
Cornell movie dialog corpus). You can use more than 6 lines :)


## Status 

Good:

1) The char2char map (StringCreativeMap) and embedding map
(EmbeddingCreativeMap) work! They can learn from the examples,
even for long sentences.

Bad:

1) Slow upon training. A better method for batch learning needs to be
added to the code.

2) The decoder does not use attention. 


## References
[https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html)

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

## Example

Run the following file to train the creative map with two instances

```bash 
python save_strings.py 
```

The code in this script reads as

```python
from creative_map import StringCreativeMap

max_seq_length = 10
        
cmap = StringCreativeMap (max_seq_length)
for t in range(100):
    cmap["hello"] = "world"
    cmap["ciao"]  = "mondo"
cmap.train ()
cmap.save ("strings.nn")
```

After this step one can run

```bash
python load_strings.py
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

## Status 

This is only a working prototype. It works rather well for short char
to char sequences. Improvements are on their way!


## References
[https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html](https://www.tensorflow.org/versions/master/tutorials/seq2seq/index.html)

[https://worksheets.codalab.org/rest/bundle/0xaae3b837f50e4dbdb9161d657df74831/contents/blob/main.html](https://worksheets.codalab.org/rest/bundle/0xaae3b837f50e4dbdb9161d657df74831/contents/blob/main.html)
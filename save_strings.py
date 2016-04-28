from creative_map import StringCreativeMap

max_seq_length = 20

def train (cmap):
    for t in range(200):
        cmap["hello"] = "world"
        cmap["ciao"]  = "mondo"
        
cmap = StringCreativeMap (max_seq_length)
train (cmap)
cmap.save ("strings.nn")


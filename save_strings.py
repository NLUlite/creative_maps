from creative_map import StringCreativeMap

max_seq_length = 10
        
cmap = StringCreativeMap (max_seq_length)
for t in range(100):
    cmap["hello"] = "world"
    cmap["ciao"]  = "mondo"
cmap.train ()
cmap.save ("strings.nn")


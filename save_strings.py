from creative_maps.string_creative_map import StringCreativeMap

max_seq_length = 10
        
cmap = StringCreativeMap (max_seq_length)
cmap["hello"] = "world"
cmap["ciao"]  = "mondo"

try:
    cmap.train ()
except:
    pass

cmap.save ("strings.nn")

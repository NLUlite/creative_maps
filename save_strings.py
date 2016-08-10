from creative_maps.string_creative_map import StringCreativeMap

max_seq_length = 10
        
cmap = StringCreativeMap (max_seq_length)
cmap["hello"] = "world"
cmap["ciao"]  = "mondo"

cmap.train ()
cmap.save ("strings.nn")


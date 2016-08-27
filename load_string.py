from creative_maps.string_creative_map import StringCreativeMap

max_seq_length = 10

cmap = StringCreativeMap (max_seq_length)
cmap.load ("strings.nn")

inp = "hello"
print inp, cmap [inp]

inp = "ciao"
print inp, cmap [inp]

print '---'

inp = "hi"
print inp, cmap [inp]

inp = "hallo"
print inp, cmap [inp]

inp = "bonjour"
print inp, cmap [inp]



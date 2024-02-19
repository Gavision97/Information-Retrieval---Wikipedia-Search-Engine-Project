from inverted_index_gcp import InvertedIndex

inverted=InvertedIndex.read_index("pkl files","index_title")
print(inverted.posting_locs )

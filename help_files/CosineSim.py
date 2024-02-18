import math

import numpy as np
from contextlib import closing
from inverted_index_gcp import MultiFileReader
from collections import Counter
from collections import defaultdict
import pandas as pd

class CosineSim:

    def __init__(self, index, DL, index_type, doc_norm):
        self.index = index
        self.DL = DL
        #self.page_rank = page_rank
        self.doc_norm = doc_norm
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.index_type = index_type
        self.bucket_name = ""##########""

    def read_posting_list(self, index, w):
        TUPLE_SIZE = 6
        TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
        with closing(MultiFileReader(self.bucket_name)) as reader:
            locs = index.posting_locs[w]
            b = reader.read(locs, index.df[w] * TUPLE_SIZE, self.index_type)
            posting_list = []
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))

        return posting_list

    def get_posting_gen(self, index, query):
        """
        This function returning the generator working with posting list.
        Parameters:
        ----------
        index: inverted index
        """
        w_pls_dict = {}
        for term in query:
            temp_pls = self.read_posting_list(index, term)
            w_pls_dict[term] = temp_pls
        return w_pls_dict


    def calcCosineSim(self, query, index, N=100):

        epsilon = .0000001

        counter = Counter(query)

        w_pls_dict = self.get_posting_gen(self.index, query)
        words = tuple(w_pls_dict.keys())
        pls = tuple(w_pls_dict.values())

        normQuery = math.sqrt(sum(x**2 for x in counter.values()))

        simDict = defaultdict(float)

        for token in np.unique(query):
            if token in index.df.keys():  # avoid terms that do not appear in the index.

                list_of_doc = pls[words.index(token)]

                for doc_id, freq in list_of_doc:

                    mone = counter[token] * freq
                    mechane = normQuery * self.doc_norm[doc_id]

                    simDict[doc_id] = simDict.get(doc_id, 0) + mone/mechane

        finalList = self.get_top_n(simDict, N)

        return finalList

    def get_top_n(self, sim_dict, N=3):
        """
        Sort and return the highest N documents according to the cosine similarity score.
        Generate a dictionary of cosine similarity scores
        Parameters:
        -----------
        sim_dict: a dictionary of similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))
        N: Integer (how many documents to retrieve). By default N = 3
        Returns:
        -----------
        a ranked list of pairs (doc_id, score) in the length of N.
        """

        return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1],
                      reverse=True)[
               :N]
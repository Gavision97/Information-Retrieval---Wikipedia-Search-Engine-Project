import math
import numpy as np
from inverted_index_gcp import *
from collections import Counter
from collections import defaultdict

class CosineSim:

    def __init__(self, index, DL, index_type, doc_norm):
        self.index = index
        self.DL = DL
        self.doc_norm = doc_norm
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.index_type = index_type
        self.bucket_name = "inverted_indexes_bucket"

    def read_posting_list(self, index, w):
        TUPLE_SIZE = 6
        with closing(MultiFileReader(self.index_type, self.bucket_name)) as reader:
            locs = index.posting_locs[w]
            b = reader.read(locs, index.df[w] * TUPLE_SIZE)
            posting_list = []
            for i in range(index.df[w]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))

        return posting_list

    def get_posting_gen(self, query):
        """
        Retrieves the posting lists for the given query terms.

        Args:
            query (list): A list of query terms.

        Returns:
            dict: A dictionary where keys are query terms and values are their corresponding posting lists.
        """
        w_pls_dict = {}

        for term in query:
            temp_pls = self.read_posting_list(self.index, term)
            w_pls_dict[term] = temp_pls

        return w_pls_dict

    def l2_norm(self, query):
        """
        Computes the L2 norm of the query vector.

        Args:
            query (list): A list representing the query vector.

        Returns:
            float: The L2 norm of the query vector.
        """
        word_counts = Counter(query)

        counts_list = [word_counts[word] ** 2 for word in set(query)]
        l2_sum = 0
        for word_num in counts_list:
            l2_sum += word_num
        l2 = math.sqrt(l2_sum)
        return l2


    def calcCosineSim(self, query, N=100):
        """
        Calculates the cosine similarity between the query and documents in the index.

        Args:
            query (list): A list representing the query.
            N (int, optional): Number of top documents to return. Defaults to 100.

        Returns:
            list: A list of tuples containing document IDs and their corresponding cosine similarity scores.
        """
        simDict = defaultdict(float)
        counter = Counter(query)
        w_pls_dict = self.get_posting_gen(query)

        words = tuple(w_pls_dict.keys())
        pls = tuple(w_pls_dict.values())

        # Normalize the query by L2 norm technique.
        normQuery = self.l2_norm(query)

        for token in np.unique(query):
            if token in self.index.df.keys():
                list_of_doc = pls[words.index(token)]

                for doc_id, freq in list_of_doc:
                    numerator = counter[token] * freq
                    denominator = normQuery * self.doc_norm[doc_id]
                    simDict[doc_id] = simDict.get(doc_id, 0) + numerator/denominator

        '''
        Generate list of tuples containing the document ID and its corresponding rounded similarity score.
              The list is sorted in descending order of similarity scores
        '''
        return sorted([(doc_id, round(score, 5)) for doc_id, score in simDict.items()], key=lambda x: x[1],reverse=True)[:N]

import math
import numpy as np
from contextlib import closing
from inverted_index_gcp import MultiFileReader
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

    def normalize_cosine_similarity(self, cosine_sim_dict):
        # Find the maximum and minimum
        max_rank = max(cosine_sim_dict.values())
        min_rank = min(cosine_sim_dict.values())

        normalized_dict = {}
        for doc_id, rank in cosine_sim_dict.items():
            # Scale the rank to the range of 0 to 1
            normalized_rank = (rank - min_rank) / (max_rank - min_rank)
            normalized_dict[doc_id] = normalized_rank

        return normalized_dict

    def read_posting_list(self, index, w):
        """
        Reads the posting list for a given term from the index.

        Args:
            index (Index): The index containing the posting lists.
            w (str): The term for which the posting list is to be retrieved.

        Returns:
            list: A list of tuples representing the posting list for the given term.
                  Each tuple contains a document ID and its corresponding term frequency.

        Raises:
            KeyError: If the term `w` is not found in the posting lists.
        """
        TUPLE_SIZE = 6
        with closing(MultiFileReader(self.index_type, self.bucket_name)) as reader:
            try:
                locs = index.posting_locs[w]
                b = reader.read(locs, index.df[w] * TUPLE_SIZE)
                posting_list = []
                for i in range(index.df[w]):
                    doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                    tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                    posting_list.append((doc_id, tf))
            except KeyError:
                posting_list = []  # Returning an empty list as default

        return posting_list

    def get_posting_gen(self, query):
        """
        This function returning the generator working with posting list.
        Parameters:
        ----------
        index: inverted index
        """
        w_pls_dict = {}
        for term in query:
            temp_pls = self.read_posting_list(self.index, term)
            w_pls_dict[term] = temp_pls
        return w_pls_dict

    def l2_norm(self, query):
        word_counts = Counter(query)

        # Get the counts of each unique word as a list
        l2_sum = sum(count ** 2 for count in word_counts.values())
        l2 = math.sqrt(l2_sum)
        return l2

    def score(self, query, N, page_rank_on_title_index_dict):
        """
          Calculates the cosine similarity between the query and documents in the index.

          Args:
              query (list): A list representing the query.
              N (int, optional): Number of top documents to return. Defaults to 100.

          Returns:
              tuple: A tuple containing two elements:
                  - A list of tuples containing document IDs and their corresponding cosine similarity scores.
                  - A dictionary containing document IDs as keys and their corresponding cosine similarity scores as values.
          """
        w_pls_dict = self.get_posting_gen(query)

        normQuery = self.l2_norm(query)
        simDict = defaultdict(float)
        counter = Counter(query)

        words = tuple(w_pls_dict.keys())
        pls = tuple(w_pls_dict.values())

        for token in np.unique(query):
            if token in self.index.df.keys():
                list_of_doc = pls[words.index(token)]
                for doc_id, freq in list_of_doc:
                    if doc_id in page_rank_on_title_index_dict.keys():
                        denominator = normQuery * self.doc_norm[doc_id]
                        numerator = counter[token] * freq
                        simDict[doc_id] = simDict.get(doc_id, 0) + numerator / denominator
        
        print(f'MAX SIM VALUES -> {max(simDict.values())}')
        print(f'COSINE SIM DICS -> {simDict.items()}')

        normalized_simDict = self.normalize_cosine_similarity(simDict)
        '''
        Generate list of tuples containing the document ID and its corresponding rounded similarity score.
              The list is sorted in descending order of similarity scores
        '''

        ls_res = sorted([(doc_id, round(score, 5)) for doc_id, score in normalized_simDict.items()], key=lambda x: x[1],
        reverse=True)[:N]

        print(f'Best values N by COSINE SIM -->> {ls_res}')

        return ls_res

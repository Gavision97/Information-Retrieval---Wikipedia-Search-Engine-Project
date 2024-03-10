import math
import threading
from time import time

import numpy as np
from contextlib import closing
from inverted_index_gcp import MultiFileReader
import multiprocessing

class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75
    index: inverted index
    """

    def __init__(self, index, DL, index_type, name, page_rank, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.name = name
        self.page_rank = page_rank
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.index_type = index_type
        self.bucket_name = "inverted_indexes_bucket"

    def read_posting_list(self, index, w):
        global w_pls_dict
        """
        Reads the posting list for a given term from the index.

        Args:
            index (Index): The index object containing the posting locations and document frequencies.
            w (str): The term for which the posting list is to be retrieved.

        Returns:
            list: A list of tuples containing document IDs and their corresponding term frequencies for the given term.

        Note:
            This method assumes that the index has been initialized and contains valid posting locations and document frequencies.
            It also assumes that the index_type and bucket_name attributes have been set appropriately.

        Raises:
            KeyError: If the term 'w' is not found in the index.

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
                    if self.index_type == "body_stem_index_directory":
                        if tf / float(self.DL[doc_id]) > 1 / 215:
                            posting_list.append((doc_id, tf))
                    else:
                        posting_list.append((doc_id, tf))
            except KeyError:
                posting_list = []  # Returning an empty list as default
        #return posting_list
        w_pls_dict[w]=posting_list

    def get_posting_gen(self, query):
        global w_pls_dict
        """
        Retrieve posting lists for a given query.

        Args:
            query (list): A list of query terms for which posting lists are to be retrieved.

        Returns:
            dict: A dictionary containing posting lists for each term in the query.
        """
        w_pls_dict = {}
        tttime=time()
        jobs=[]
        for term in query:
            thread=threading.Thread(target=self.read_posting_list,args=(self.index, term,))
            jobs.append(thread)
            #w_pls_dict[term] = self.read_posting_list(self.index, term)
            thread.start()
        for j in jobs:
            j.join()
        print(f'BM25 Adding to dic after all processed time -> {(time() - tttime)}')
        return w_pls_dict


    def calc_idf(self, list_of_tokens):
        """
        Calculate the inverse document frequency (IDF) for a list of tokens.

        Args:
            list_of_tokens (list): A list of tokens (terms) for which IDF is to be calculated.

        Returns:
            dict: A dictionary containing the IDF values for each token in the input list.
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search_(self, query, N=3):
        """
        Search for documents based on the given query.

        Args:
            query (list): A list of query terms.
            N (int, optional): The number of top documents to return. Defaults to 3.

        Returns:
            list: A list of tuples containing document IDs and their corresponding BM25 scores.
        """
        w_pls_dict = self.get_posting_gen(query)
        words = tuple(w_pls_dict.keys())
        pls = tuple(w_pls_dict.values())
        self.idf = self.calc_idf(query)
        term_frequencies_dict = {}

        for term in query:
            if term in self.index.df:
                term_frequencies_dict[term] = dict(pls[words.index(term)])
        candidates = []
        for term in np.unique(query):
            if term in words:
                current_list = (pls[words.index(term)])
                candidates += current_list
        candidates = np.unique([c[0] for c in candidates])


        if self.name == 'title':
            return sorted([(doc_id,
                            round(self._score(query, doc_id, term_frequencies_dict), 5))
                           for doc_id in candidates], key=lambda x: x[1], reverse=True)[:N]

        # For body index, we first verify that page rank is not zero, in order to avoid
        # documents that has high BM25 and no page rank.
        return sorted([(doc_id,
                        round(self._score(query, doc_id, term_frequencies_dict), 5) if self.page_rank.get(doc_id, 0) > 0 else 0)
                       for doc_id in candidates], key=lambda x: x[1], reverse=True)[:N]

    def _score(self, query, doc_id, term_frequencies_dict):
        """
        Calculate the BM25 score for a given document and query.
        Args:
            query (list): A list of query terms.
            doc_id (int): The ID of the document to score.
            term_frequencies_dict (dict): A dictionary containing term frequencies for each query term in the document.

        Returns:
            float: The BM25 score for the document.
        """
        score = 0.0
        if doc_id not in self.DL.keys():
            return -math.inf
        doc_len = self.DL[doc_id]

        for term in query:
            try:
                if doc_id in term_frequencies_dict[term]:
                    freq = term_frequencies_dict[term][doc_id]
                    numerator = self.idf[term] * freq * (self.k1 + 1)
                    denominator = freq + self.k1 * (1 - self.b + (self.b * doc_len / self.AVGDL))
                    score += (numerator / denominator)

            # In case 'term' is not in 'term_frequencies_dict'
            except KeyError as e:
                score += 0

        return score


def merge_results_(title_scores, body_scores, title_weight=0.5, text_weight=0.5, page_rank=None,page_views=None, N=3):
    """
    Merge search results from title and body scores, applying weights and considering PageRank.

    Args:
        page_views: (dict): of pages id and num of views
        title_scores (list): A list of tuples containing document IDs and scores from the title search.
        body_scores (list): A list of tuples containing document IDs and scores from the body search.
        title_weight (float): The weight to be applied to title scores (default is 0.5).
        text_weight (float): The weight to be applied to body scores (default is 0.5).
        page_rank (dict): A dictionary containing PageRank scores for each document (default is None).
        N (int): The number of top results to return (default is 3).

    Returns:
        list: A list of tuples containing document IDs and merged scores, sorted by score in descending order.

    Note:
        This method merges search results from title and body searches, applying specified weights to each score.
        If PageRank scores are provided, they are incorporated into the final score with a square root transformation.
        Documents are sorted based on the merged scores, and the top N results are returned.
    """
    merged_lst = []

    ts = [(doc_id, score * title_weight) for doc_id, score in title_scores]
    bs = [(doc_id, score * text_weight) for doc_id, score in body_scores]
    title_dict = {}
    body_dict = {}
    for doc_id, score in ts:
        title_dict.setdefault(doc_id, []).append(score)
    for doc_id, score in bs:
        body_dict.setdefault(doc_id, []).append(score)
    inter = set(title_dict.keys()) & set(body_dict.keys())
    diff = (set(title_dict.keys()) | set(body_dict.keys())) - (set(title_dict.keys()) & set(body_dict.keys()))

    if len(diff) > 0:
        res_list = []
        for key in list(diff):
            if key in title_dict.keys():
                res_list.append((key, title_dict[key][0] + math.sqrt(page_rank.get(key, 0))))
                res_list.append((key, title_dict[key][0]))
            else:
                res_list.append((key, body_dict[key][0]))
        merged_lst.extend(sorted(res_list, key=lambda x: x[1], reverse=True))

    for doc_id in inter:
        merged_lst.append(
            (doc_id, title_dict[doc_id][0] + body_dict[doc_id][0] + (math.sqrt(page_rank.get(doc_id, 0)))))
    return sorted(merged_lst, key=lambda x: x[1], reverse=True)[:N]

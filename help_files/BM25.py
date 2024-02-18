import math

import numpy as np
from contextlib import closing
from inverted_index_gcp import MultiFileReader


def get_candidate_documents(query_to_search, words, pls):
    candidates = []
    for term in np.unique(query_to_search):
        if term in words:
            current_list = (pls[words.index(term)])
            candidates += current_list
    candidates = [i[0] for i in candidates]
    return np.unique(candidates)


# When preprocessing the data have a dictionary of document length for each document saved in a variable called `DL`.
class BM25:
    """
    Best Match 25.
    ----------
    k1 : float, default 1.5
    b : float, default 0.75
    index: inverted index
    """

    def __init__(self, index, DL, index_type,page_rank, k1=1.5, b=0.75):
        self.b = b
        self.k1 = k1
        self.index = index
        self.DL = DL
        self.page_rank = page_rank
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.index_type = index_type
        self.bucket_name = "##########"

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
                if self.index_type == "_body" or self.index_type == "_body_stem":
                    if tf / float(self.DL[doc_id]) > 1 / 215:
                        posting_list.append((doc_id, tf))
                else:
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

    def calc_idf(self, list_of_tokens):
        """
        This function calculate the idf values according to the BM25 idf formula for each term in the query.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        Returns:
        -----------
        idf: dictionary of idf scores. As follows:
                                                    key: term
                                                    value: bm25 idf score
        """
        idf = {}
        for term in list_of_tokens:
            if term in self.index.df.keys():
                n_ti = self.index.df[term]
                idf[term] = math.log(1 + (self.N - n_ti + 0.5) / (n_ti + 0.5))
            else:
                pass
        return idf

    def search(self, query, N=3):
        """
        This function calculate the bm25 score for given query and document.
        We need to check only documents which are 'candidates' for a given query.
        This function return a dictionary of scores as the following:
                                                                    key: query_id
                                                                    value: a ranked list of pairs (doc_id, score) in the length of N.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        Returns:
        -----------
        score: float, bm25 score.
        """
        # YOUR CODE HERE
        w_pls_dict = self.get_posting_gen(self.index, query)
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
        return sorted([(doc_id, round(self._score(query, doc_id, term_frequencies_dict), 5)) for doc_id in candidates], key=lambda x: x[1], reverse=True)[:N]

    def _score(self, query, doc_id, term_frequencies_dict):
        """
        This function calculate the bm25 score for given query and document.
        Parameters:
        -----------
        query: list of token representing the query. For example: ['look', 'blue', 'sky']
        doc_id: integer, document id.
        Returns:
        -----------
        score: float, bm25 score.
        """
        score = 0.0
        if doc_id not in self.DL.keys():
            return -math.inf
        doc_len = self.DL[doc_id]
        page_rank_max = self.page_rank["3434750"]
        for term in query:
            if doc_id in term_frequencies_dict[term]:
                freq = term_frequencies_dict[term][doc_id]
                numerator = self.idf[term] * freq * (self.k1 + 1)
                denominator = freq + self.k1 * (1 - self.b + (self.b * doc_len / self.AVGDL))
                score += (numerator / denominator)

                # pageScore = self.page_rank.get(doc_id, 1)
                #
                # if pageScore > 0:
                #     score += (numerator / denominator) + 1.6 * math.log(pageScore, 10)
                # else:
                #     score += (numerator / denominator)
        return score


def merge_results(title_scores, body_scores, title_weight=0.5, text_weight=0.5, page_rank=None, N=3):
    """
    This function merge and sort documents retrieved by its weighte score (e.g., title and body).
    Parameters:
    -----------
    title_scores: a dictionary build upon the title index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    body_scores: a dictionary build upon the body/text index of queries and tuples representing scores as follows:
                                                                            key: query_id
                                                                            value: list of pairs in the following format:(doc_id,score)
    title_weight: float, for weigted average utilizing title and body scores
    text_weight: float, for weigted average utilizing title and body scores
    N: Integer. How many document to retrieve. This argument is passed to topN function. By default N = 3, for the topN function.
    Returns:
    -----------
    dictionary of querires and topN pairs as follows:
                                                        key: query_id
                                                        value: list of pairs in the following format:(doc_id,score).
    """
    # YOUR CODE HERE
    merged_lst = []
    maxPage = page_rank["3434750"]

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
                res_list.append((key, title_dict[key][0]))
            else:
                res_list.append((key, body_dict[key][0]))
        merged_lst.extend(sorted(res_list, key=lambda x: x[1], reverse=True))

    for doc_id in inter:
        merged_lst.append((doc_id, title_dict[doc_id][0] + body_dict[doc_id][0] + (page_rank.get(doc_id, 0) / maxPage)))
    return sorted(merged_lst, key=lambda x: x[1], reverse=True)[:N]

import math

import numpy as np
from contextlib import closing
from inverted_index_gcp import MultiFileReader
from collections import Counter
from collections import defaultdict
from inverted_index_gcp import *
import pandas as pd


class CosinSimilarity:

    def __init__(self, index, DL, index_type) -> None:
        self.index = index
        self.DL = DL
        self.N = len(DL)
        self.AVGDL = sum(DL.values()) / self.N
        self.index_type = index_type
        #self.bucket_name = ""##########""

    def get_posting_iter(self):
        """
        This function returning the iterator working with posting list.

        Parameters:
        ----------
        index: inverted index
        """
        words, pls = zip(*self.index.posting_lists_iter())
        return words, pls

    def generate_query_tfidf_vector(self, query_to_search):
        """
        Generate a vector representing the query. Each entry within this vector represents a tfidf score.
        The terms representing the query will be the unique terms in the index.

        We will use tfidf on the query as well.
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the query.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        Returns:
        -----------
        vectorized query with tfidf scores
        """

        epsilon = .0000001

        total_vocab_size = len(self.index.term_total)
        Q = np.zeros((total_vocab_size))

        term_vector = list(self.index.term_total.keys())
        counter = Counter(query_to_search)

        for token in np.unique(query_to_search):
            if token in self.index.term_total.keys():  # avoid terms that do not appear in the index.
                tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
                df = self.index.df[token]
                idf = math.log((len(self.DL)) / (df + epsilon), 10)  # smoothing

                try:
                    ind = term_vector.index(token)
                    Q[ind] = tf * idf
                except:
                    pass
        return Q

    def get_candidate_documents_and_scores(self, query_to_search, words, pls):
        """
        Generate a dictionary representing a pool of candidate documents for a given query. This function will go through every token in query_to_search
        and fetch the corresponding information (e.g., term frequency, document frequency, etc.') needed to calculate TF-IDF from the posting list.
        Then it will populate the dictionary 'candidates.'
        For calculation of IDF, use log with base 10.
        tf will be normalized based on the length of the document.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.

        words,pls: iterator for working with posting.

        Returns:
        -----------
        dictionary of candidates. In the following format:
                                                                   key: pair (doc_id,term)
                                                                   value: tfidf score.
        """
        candidates = {}
        for term in np.unique(query_to_search):
            if term in words:
                list_of_doc = pls[words.index(term)]
                normlized_tfidf = [(doc_id, (freq / self.DL[str(doc_id)]) * math.log(len(self.DL) / self.index.df[term], 10)) for doc_id, freq in list_of_doc]

                for doc_id, tfidf in normlized_tfidf:
                    candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

        return candidates

    def generate_document_tfidf_matrix(self, query_to_search, words, pls):
        """
        Generate a DataFrame `D` of tfidf scores for a given query.
        Rows will be the documents candidates for a given query
        Columns will be the unique terms in the index.
        The value for a given document and term will be its tfidf score.

        Parameters:
        -----------
        query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                         Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

        index:           inverted index loaded from the corresponding files.


        words,pls: iterator for working with posting.

        Returns:
        -----------
        DataFrame of tfidf scores.
        """

        total_vocab_size = len(self.index.term_total)
        candidates_scores = self.get_candidate_documents_and_scores(query_to_search, self.index, words, pls)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
        unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])

        D = np.zeros((len(unique_candidates), total_vocab_size))
        D = pd.DataFrame(D)

        D.index = unique_candidates
        D.columns = self.index.term_total.keys()

        for key in candidates_scores:
            tfidf = candidates_scores[key]
            doc_id, term = key
            D.loc[doc_id][term] = tfidf

        return D

    def cosine_similarity(self, D, Q):
        """
        Calculate the cosine similarity for each candidate document in D and a given query (e.g., Q).
        Generate a dictionary of cosine similarity scores
        key: doc_id
        value: cosine similarity score

        Parameters:
        -----------
        D: DataFrame of tfidf scores.

        Q: vectorized query with tfidf scores

        Returns:
        -----------
        dictionary of cosine similarity score as follows:
                                                                    key: document id (e.g., doc_id)
                                                                    value: cosine similarty score.
        """
        cosine_dict = {}
        for i, row in D.iterrows():
            row = row.to_numpy()
            cosine_score = sum(row * Q) / math.sqrt(sum(np.power(row, 2)) * sum(np.power(Q, 2)))
            cosine_dict[i] = cosine_score
        return cosine_dict

    def get_topN_score_for_queries(self, queries_to_search, N=100):
        """
        Generate a dictionary that gathers for every query its topN score.

        Parameters:
        -----------
        queries_to_search: a dictionary of queries as follows:
                                                            key: query_id
                                                            value: list of tokens.
        index:           inverted index loaded from the corresponding files.
        N: Integer. How many documents to retrieve. This argument is passed to the topN function. By default N = 3, for the topN function.

        Returns:
        -----------
        return: a dictionary of queries and topN pairs as follows:
                                                            key: query_id
                                                            value: list of pairs in the following format:(doc_id, score).
        """
        q_top_n = {}
        words, pls = self.get_posting_iter()
        for qid, tokens in queries_to_search.items():
            sim_dict = self.cosine_similarity(self.generate_document_tfidf_matrix(tokens, words, pls), self.generate_query_tfidf_vector(tokens))

            q_top_n[qid] = sorted([(doc_id, round(score,5)) for doc_id, score in sim_dict.items()], key = lambda x: x[1],reverse=True)[:N]

        return q_top_n

import math
import numpy as np
import time
from inverted_index_gcp import *
from collections import Counter
from collections import defaultdict


class PageRankSim:
    def __init__(self, index, index_type, page_rank):
        self.index = index
        self.page_rank = page_rank
        self.index_type = index_type
        self.bucket_name = "inverted_indexes_bucket"

    def normalize_page_rank_dict(self, page_rank_dict):
        """
        Normalize the page rank dictionary by scaling the values to the range of 0 to 1 using MinMax algorithm.

        Args:
            page_rank_dict (dict): A dictionary containing document IDs as keys and their corresponding page ranks as values.

        Returns:
            dict: A normalized dictionary where the values are scaled to the range of 0 to 1.
        """
        # Find the maximum and minimum page ranks
        max_rank = max(page_rank_dict.values())
        min_rank = min(page_rank_dict.values())
        zero = False
        if max_rank == min_rank:
            zero = True
        # Normalize the page ranks
        normalized_dict = {}
        for doc_id, rank in page_rank_dict.items():
            # Scale the rank to the range of 0 to 1
            if zero:
                normalized_dict[doc_id] = 0
                continue
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

    def get_top_N_docs_by_title_and_page_rank(self, query, DL_title, N):
        """
        Retrieves the top N documents based on their title page rank and relevance to the query.

        Args:
            query (list): A list representing the query terms.
            N (int): Number of top documents to return.
            cosine_sim_on_body_index_dict (dict): A dictionary containing document IDs as keys and their corresponding cosine similarity scores as values.

        Returns:
            list: A list of tuples containing the top N document IDs and their corresponding normalized page rank values.
        """
        alpha = 0.00001
        page_rank_tf_dict = {}
        temp_page_rank_dict = {}
        visited_docs = set()  # Set to keep track of visited dic_ids, in order to add page rank once
        w_pls_dict = self.get_posting_gen(query)

        for term in query:
            if term in w_pls_dict.keys():
                for doc_id, tf in w_pls_dict[term]:
                    page_rank_tf_dict[doc_id] = page_rank_tf_dict.get(doc_id, 0) + 1

        for doc_id_ in page_rank_tf_dict.keys():
            temp_page_rank_dict[doc_id_] = self.page_rank.get(doc_id_, 0) + alpha

        normalized_temp_page_rank_dict = self.normalize_page_rank_dict(temp_page_rank_dict)

        for doc_id__ in page_rank_tf_dict.keys():
            page_rank_tf_dict[doc_id__] = (page_rank_tf_dict[doc_id__] + normalized_temp_page_rank_dict[
                doc_id__]) / DL_title.get(doc_id, 1)

        normalized_page_rank_dict = self.normalize_page_rank_dict(page_rank_tf_dict)

        # Normalize the values to be in the range of 0-1
        norm_page_rank_title_dic = self.normalize_page_rank_dict(page_rank_tf_dict)

        # Return top N doc_id with their corresponding page rank value in (doc_in, page_rank) in sorted list
        ls_res = sorted(normalized_page_rank_dict.items(), key=lambda x: x[1], reverse=True)[:N]

        sorted_dict = {doc_id: score for doc_id, score in ls_res}

        return (ls_res, sorted_dict)


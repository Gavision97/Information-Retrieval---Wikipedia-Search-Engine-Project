import math
from collections import Counter
from collections import defaultdict
import numpy as np
import pandas as pd
from contextlib import closing
from inverted_index_gcp import MultiFileReader

bucket_name = ""##########""

def read_posting_list(inverted, w, index_type):
    TUPLE_SIZE = 6
    TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
    with closing(MultiFileReader(bucket_name)) as reader:
        locs = inverted.posting_locs[w]
        b = reader.read(locs, inverted.df[w] * TUPLE_SIZE, index_type)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i*TUPLE_SIZE:i*TUPLE_SIZE+4], 'big')
            tf = int.from_bytes(b[i*TUPLE_SIZE+4:(i+1)*TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
    return posting_list


def get_posting_gen(index, query, index_type=""):
    """
    This function returning the generator working with posting list.
    Parameters:
    ----------
    index: inverted index
    """
    w_pls_dict = {}
    for term in query:
        temp_pls = read_posting_list(index, term, index_type)
        w_pls_dict[term] = temp_pls
    return w_pls_dict

def generate_query_tfidf_vector(query_to_search, index, DL):
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
    total_vocab_size = len(np.unique(query_to_search))
    Q = np.zeros((total_vocab_size))
    term_vector = list(np.unique(query_to_search))
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] /len(query_to_search) # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(DL))/(df + epsilon), 10)  # smoothing
            try:
                ind = term_vector.index(token)
                Q[ind] = tf *idf
            except:
                pass
    return Q


def get_candidate_documents_and_scores(query_to_search, index, words, pls, DL):
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
    words,pls: generator for working with posting.
    Returns:
    -----------
    dictionary of candidates. In the following format:
                                                               key: pair (doc_id,term)
                                                               value: tfidf score.
    """
    candidates = {}
    N = len(DL)
    for term in np.unique(query_to_search):
        if term in words:
            list_of_doc = pls[words.index(term)]

            normlized_tfidf = [(doc_id, (freq / DL[doc_id]) * math.log(N / index.df[term], 10)) for doc_id, freq in
                               list_of_doc]

            for doc_id, tfidf in normlized_tfidf:
                candidates[(doc_id, term)] = candidates.get((doc_id, term), 0) + tfidf

    return candidates


def generate_document_tfidf_matrix(query_to_search, index, words, pls, DL):
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
    words,pls: generator for working with posting.
    Returns:
    -----------
    DataFrame of tfidf scores.
    """

    total_vocab_size = len(np.unique(query_to_search))
    candidates_scores = get_candidate_documents_and_scores(query_to_search, index, words, pls, DL)  # We do not need to utilize all document. Only the docuemnts which have corrspoinding terms with the query.
    unique_candidates = np.unique([doc_id for doc_id, freq in candidates_scores.keys()])
    D = np.zeros((len(unique_candidates), total_vocab_size))
    D = pd.DataFrame(D)

    D.index = unique_candidates
    D.columns = query_to_search

    for key in candidates_scores:
        tfidf = candidates_scores[key]
        doc_id, term = key
        D.loc[doc_id][term] = tfidf

    return D


def cosine_similarity(D, Q, doc_norm):
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

    cosine_sim_dict = {}
    q_size = np.linalg.norm(Q)
    for doc_id, row in D.iterrows():
        top = np.dot(row, Q)
        di_size = doc_norm[doc_id]
        cosine_sim_dict[doc_id] = (top / (float(di_size) * float(q_size)))
    return cosine_sim_dict


def get_top_n(sim_dict, N=3):
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

    return sorted([(doc_id, round(score, 5)) for doc_id, score in sim_dict.items()], key=lambda x: x[1], reverse=True)[
           :N]


def get_topN_score_for_queries(query, body_index, DL, doc_norm, N=3):
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

    w_pls_dict = get_posting_gen(body_index, query)
    words = tuple(w_pls_dict.keys())
    pls = tuple(w_pls_dict.values())
    D = generate_document_tfidf_matrix(query, body_index, words, pls, DL)
    Q = generate_query_tfidf_vector(query, body_index, DL)
    cosine_sim_dict = cosine_similarity(D, Q, doc_norm)
    topN_list = get_top_n(cosine_sim_dict, N)
    return topN_list

def getBinaryListResult(index, query, index_type, N):
    newDict = {}
    w_pls_dict = get_posting_gen(index, query, index_type)
    for term in query:
        if term in w_pls_dict.keys():
            for doc_id, tf in w_pls_dict[term]:
                newDict[doc_id] = newDict.get(doc_id, 0) + 1

    resDict = dict(map(lambda item: (item[0], item[1]), newDict.items()))

    sorted_docs = sorted(resDict.items(), key=lambda x: x[1], reverse=True)[:N]
    return sorted_docs

def combineListTuplesIntoDict(bm25List=None, cosineList=None, binaryList=None, pageRank=None):

    maxPage = pageRank["3434750"]

    args = [bm25List, cosineList, binaryList]
    merged_dict = defaultdict(float)
    mergeL = []
    for arg in args:
        if arg is not None:
            mergeL += arg
    for doc_id, score in mergeL:
        merged_dict[doc_id] += score

    merged_list = [(key, value + pageRank.get(key, 0) / maxPage) for key, value in merged_dict.items()]

    return merged_list

def getDocListResultWithPageRank(index, query, index_type, N, pageRank):
    newDict = {}
    w_pls_dict = get_posting_gen(index, query, index_type)
    for term in query:
        if term in w_pls_dict.keys():
            for doc_id, tf in w_pls_dict[term]:
                newDict[doc_id] = newDict.get(doc_id, 0) + 1

    maxPage = pageRank["3434750"]
    for k, v in newDict.items():
        newDict[k] = v / len(query)

    sorted_docs = sorted(newDict.items(), key=lambda x: x[1], reverse=True)[:N]
    return sorted_docs
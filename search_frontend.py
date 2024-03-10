"""
Imports:
"""
import pickle
import threading

from BM25 import BM25, merge_results_
from Tokenizer import Tokenizer
from CosineSim import CosineSim
from PageRankSim import PageRankSim
from flask import Flask, request, jsonify
from google.cloud import storage
from time import time
import asyncio

BUCKET_NAME = "inverted_indexes_bucket"
client = storage.Client()


class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):

        self.my_bucket = client.bucket(bucket_name=BUCKET_NAME)
        self.tokenizer = Tokenizer()
        for blob in client.list_blobs(BUCKET_NAME):
            if blob.name == "body_index_final.pkl":
                with blob.open('rb') as openfile:
                    self.body_stem_index = pickle.load(openfile)

            elif blob.name == "title_index_final.pkl":
                with blob.open('rb') as openfile:
                    self.title_stem_index = pickle.load(openfile)

            elif blob.name == "body_dl_.pkl":
                with blob.open('rb') as openfile:
                    self.DL_body = pickle.load(openfile)

            elif blob.name == "title_dl_.pkl":
                with blob.open('rb') as openfile:
                    self.DL_title = pickle.load(openfile)

            elif blob.name == "doc_l2_norm.pkl":
                with blob.open('rb') as openfile:
                    self.doc_norm = pickle.load(openfile)

            elif blob.name == "title_dictionary.pkl":
                with blob.open('rb') as openfile:
                    self.title_dic = pickle.load(openfile)

            elif blob.name == "pageRank.pkl":
                with blob.open('rb') as openfile:
                    self.page_rank = pickle.load(openfile)
            elif blob.name == "pageviews-202108-user.pkl":
                with blob.open('rb') as openfile:
                    self.pageviews = pickle.load(openfile)

        self.cosine_stem_body = CosineSim(self.body_stem_index,
                                          self.DL_body,
                                          "body_stem_index_directory",
                                          self.doc_norm)
        self.page_rank_sim = PageRankSim(self.title_stem_index,
                                         "title_index_dictionary_final",
                                         self.page_rank)

        self.BM25_body = BM25(self.body_stem_index, self.DL_body, "body_index_directory_final", "body", self.page_rank,
                              k1=1.5, b=0.65)
        self.BM25_title = BM25(self.title_stem_index, self.DL_title, "title_index_directory_final", "title",
                               self.page_rank, k1=2.2, b=0.85)

        self.cosine_stem_body = CosineSim(self.body_stem_index, self.DL_body, "body_index_directory_final",
                                          self.doc_norm)
        self.cosine_stem_title = CosineSim(self.title_stem_index, self.DL_body, "title_index_directory_final",
                                           self.doc_norm)

        self.page_rank_sim = PageRankSim(self.title_stem_index, "title_index_dictionary_final", self.page_rank)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


async def title_search(query_stemmed, N):
    title_res = app.BM25_title.search_(query_stemmed, N)
    return title_res

async def body_search(query_stemmed, N):
    body_res = app.BM25_body.search_(query_stemmed, N)
    return body_res

@app.route("/search")
async def search():
    #global title_res,body_res
    ''' Returns up to a 100 of your best search results for the query. This is
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use
        PageRank, query expansion, etc.

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of up to 100 search results, ordered from best to worst where each
        element is a tuple (wiki_id, title).
    '''
    res = []
    title_res=[]
    body_res=[]
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    question = ["where", "how", "what", "why", "when", "who"]
    query_stemmed = list(set(app.tokenizer.tokenize(query)))
    query = query.lower()
    is_question = any(substring in query for substring in question) or query.endswith("?")
    if is_question:
        title_weight = 0.3
        body_weight = 0.7

    else:
        title_weight = 0.7
        body_weight = 0.3

    ttttime = time()
    #title_res = app.BM25_title.search_(query_stemmed, 50)
    #body_res = app.BM25_body.search_(query_stemmed, 50)
    title_task = asyncio.create_task(title_search(query_stemmed, 50))
    body_task = asyncio.create_task(body_search(query_stemmed, 50))

    title_res = await title_task
    body_res = await body_task
    ttime = time()

    merged_res = merge_results_(title_res, body_res, title_weight=title_weight, text_weight=body_weight,
                                page_rank=app.page_rank, page_views=app.pageviews, N=30)
    print(f'merged time -> {(time() - ttime)}')

    res = [(str(doc_id), app.title_dic.get(doc_id, "not found")) for doc_id, score in merged_res][:30]
    print(f'total time -> {(time() - ttttime)}')

    return jsonify(res)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)



################################################################################
# def merged_results_(body_list, title_list, body_weight, title_weight) -> list[tuple[any, any]]:
#     body_dict_sorted_by_id = sorted(body_list, key=lambda x: x[0], reverse=True)
#     title_dict_sorted_by_id = sorted(title_list, key=lambda x: x[0], reverse=True)
#     result_dict = {}
#
#     for (doc_id_body, score_body) in body_dict_sorted_by_id:
#         if doc_id_body in [doc_id for doc_id, _ in title_dict_sorted_by_id]:
#             score_title = next(score for doc_id, score in title_dict_sorted_by_id if doc_id == doc_id_body)
#             merged_score = (score_body * body_weight) + (score_title * title_weight)
#         else:
#             merged_score = score_body
#
#         # Add merged score to result dictionary
#         result_dict[doc_id_body] = merged_score
#
#     return sorted([(doc_id, score) for doc_id, score in result_dict.items()], key=lambda x: x[1], reverse=True)
#
#
# ################################################################################
# def merged_results_and_sort(body_list, title_list, body_weight, title_weight) -> list[tuple[any, any]]:
#     body_dict_sorted_by_id = sorted(body_list, key=lambda x: x[0], reverse=True)
#     title_dict_sorted_by_id = sorted(title_list, key=lambda x: x[0], reverse=True)
#     result_dict = {}
#
#     for (doc_id_title, score_title) in title_dict_sorted_by_id:
#         if doc_id_title in [doc_id for doc_id, _ in body_dict_sorted_by_id]:
#             score_body = next(score for doc_id, score in body_dict_sorted_by_id if doc_id == doc_id_title)
#             merged_score = (score_body * body_weight) + (score_title * title_weight)
#         else:
#             merged_score = score_title
#
#         # Add merged score to result dictionary
#         result_dict[doc_id_title] = merged_score
#
#     return sorted([(doc_id, score) for doc_id, score in result_dict.items()], key=lambda x: x[1], reverse=True)
#

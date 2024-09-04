"""
Imports for our app:
"""
import pickle
from typing import List, Tuple, Any
from BM25 import BM25, merge_results_
from Tokenizer import Tokenizer
from CosineSim import CosineSim
from PageRankSim import PageRankSim
from flask import Flask, request, jsonify
from google.cloud import storage
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
                    self.page_views = pickle.load(openfile)

        self.BM25_body = BM25(self.body_stem_index, self.DL_body, "body_index_directory_final", "body", self.page_rank, self.page_views, k1=1.75, b=0.65)
        self.BM25_title = BM25(self.title_stem_index, self.DL_title,  "title_index_directory_final", "title", self.page_rank, self.page_views, k1=2.2, b=0.85)

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
    
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
        return jsonify(res)
    # BEGIN SOLUTION
    question = ["where", "how", "what", "why", "when", "who"]
    query_stemmed = list(set(app.tokenizer.tokenize(query)))
    query = query.lower()
    is_question = any(substring in query for substring in question) or query.endswith("?")

    if is_question:
        title_weight = 0.425
        body_weight = 0.575

    else:
        title_weight = 0.725
        body_weight = 0.275

    title_task = asyncio.create_task(title_search(query_stemmed, 50))
    body_task = asyncio.create_task(body_search(query_stemmed, 50))

    title_res = await title_task
    body_res = await body_task

    merged_res = merge_results_(title_res, body_res, app.page_views, title_weight=title_weight, text_weight=body_weight,
                                page_rank=app.page_rank, N=30)

    res = [(str(doc_id), app.title_dic.get(doc_id, "not found")) for doc_id, score in merged_res][:30]
    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=False, use_reloader=False)

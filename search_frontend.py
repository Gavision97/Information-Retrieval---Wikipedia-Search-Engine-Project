from inverted_index_gcp import *
import math
import pickle
import numpy as np
import pandas as pd
from collections import Counter

from CosinSimilarity import CosinSimilarity
from flask import Flask, request, jsonify
from Tokenizer import Tokenizer
from google.cloud import storage

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):
        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)
        bucket_name = "inverted_indexes_bucket"

        client = storage.Client()

        self.tokenizer = Tokenizer()
        self.my_bucket = client.bucket(bucket_name=bucket_name)
        self.page_rank = {}
        self.tempDict = {}

        for blob in client.list_blobs(bucket_name):
            if blob.name == "body_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.body_stem_index = pickle.load(openfile)

            elif blob.name == "title_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.title_stem_index = pickle.load(openfile)

            elif blob.name == "body_dictionary_length.pkl":
                with blob.open('rb') as openfile:
                    self.DL_body = pickle.load(openfile)

            elif blob.name == "title_dictionary_length.pkl":
                with blob.open('rb') as openfile:
                    self.DL_title = pickle.load(openfile)

            elif blob.name == "doc2vec.pkl":
                with blob.open('rb') as openfile:
                    self.doc_title_dict = pickle.load(openfile)

            elif blob.name == "pageRank.pkl":
                with blob.open('rb') as openfile:
                    self.tempDict = pickle.load(openfile)
                    self.page_rank = {str(k): v for k, v in self.tempDict.items()}


        self.cosine_stem_body = CosinSimilarity(self.body_stem_index, self.DL_body, "_body_stem")

        #self.cosine_title = CosinSimilarity(self.title_stem_index, self.DL_title, "_title_stem", self.doc_norm)

app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
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
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION

    query_stemmed = list(set(app.tokenizer.tokenize(query, True)))

    body_res = app.cosine_stem_body.get_topN_score_for_queries(query_stemmed, 100)
    res = [(int(doc_id), app.doc_title_dict.get(doc_id, "not found")) for doc_id, score in body_res]

    # END SOLUTION
    return jsonify(["hiiiii"])


    # TODO : after verifying that the app is working, try to add also title.
    #title_res = getDocListResultWithPageRank(app.title_stem_index, query_stemmed, "_title_stem", 30, app.page_rank)
    #title_weight = 0.3
    #body_weight = 0.7
    #גmerged_res = merge_results(title_res, body_res, title_weight=title_weight, text_weight=body_weight,page_rank=app.page_rank , N=10)
    #res = [(int(doc_id), app.doc_title_dict.get(doc_id, "not found")) for doc_id, score in merged_res]


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

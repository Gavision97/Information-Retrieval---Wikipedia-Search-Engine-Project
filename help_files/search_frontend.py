import concurrent.futures
from flask import Flask, request, jsonify
import pickle
from google.cloud import storage
from nltk.corpus import stopwords
from Tokenizer import Tokenizer
from search_backend import *
from BM25 import *
import csv
import pandas as pd
import re
from CosineSim import *

class MyFlaskApp(Flask):

    def run(self, host=None, port=None, debug=None, **options):
        bucket_name = "##########"
        self.tokenizer = Tokenizer()
        client = storage.Client()
        self.my_bucket = client.bucket(bucket_name=bucket_name)
        self.page_rank = {}
        self.tempDict = {}
        for blob in client.list_blobs(bucket_name):
            if blob.name == "body_index.pkl":
                with blob.open('rb') as openfile:
                    self.index_body = pickle.load(openfile)

            elif blob.name == "title_index.pkl":
                with blob.open('rb') as openfile:
                    self.index_title = pickle.load(openfile)

            elif blob.name == "anchor_index.pkl":
                with blob.open('rb') as openfile:
                    self.index_anchor = pickle.load(openfile)

            elif blob.name == "body_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.body_stem_index = pickle.load(openfile)

            elif blob.name == "title_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.title_stem_index = pickle.load(openfile)

            elif blob.name == "body_DL.pkl":
                with blob.open('rb') as openfile:
                    self.DL_body = pickle.load(openfile)

            elif blob.name == "title_DL.pkl":
                with blob.open('rb') as openfile:
                    self.DL_title = pickle.load(openfile)

            elif blob.name == "doc2vec.pkl":
                with blob.open('rb') as openfile:
                    self.doc_title_dict = pickle.load(openfile)

            elif blob.name == "doc_norm_size.pkl":
                with blob.open('rb') as openfile:
                    self.doc_norm = pickle.load(openfile)


            elif blob.name == "pageRank.pkl":
                with blob.open('rb') as openfile:
                    self.tempDict = pickle.load(openfile)
                    self.page_rank = {str(k): v for k, v in self.tempDict.items()}

            elif blob.name == "pageViews.pkl":
                with blob.open('rb') as openfile:
                    self.page_views = pickle.load(openfile)

            # to use the word2vec model:
            # word2vec.most_similar(model.get_entity('Ritalin'), 1)

        self.BM25_body = BM25(self.body_stem_index, self.DL_body, "_body_stem", self.page_rank, k1=1.5, b=0.65)
        self.BM25_title = BM25(self.title_stem_index, self.DL_title,  "_title_stem", self.page_rank, k1=2.2, b=0.85)
        self.cosine_stem_body = CosineSim(self.body_stem_index, self.DL_body, "_body_stem", self.doc_norm)
        self.cosine_body = CosineSim(self.index_body, self.DL_body, "", self.doc_norm)
        self.cosine_title = CosineSim(self.title_stem_index, self.DL_title, "_title_stem", self.doc_norm)

        super(MyFlaskApp, self).run(host=host, port=port, debug=debug, **options)


app = MyFlaskApp(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route("/search")
def search():
    ''' Returns up to a 100 of your best search results for the query. This is 
        the place to put forward your best search engine, and you are free to
        implement the retrieval whoever you'd like within the bound of the 
        project requirements (efficiency, quality, etc.). That means it is up to
        you to decide on whether to use stemming, remove stopwords, use 
        PageRank.py, query expansion, etc.

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
    question = ["where", "how", "what", "why"]
    query_stemmed = list(set(app.tokenizer.tokenize(query, True)))
    query = query.lower()
    if any(substring in query for substring in question) or query.endswith("?"):
        body_res = app.cosine_stem_body.calcCosineSim(query_stemmed, app.body_stem_index, 30)
        title_res = getDocListResultWithPageRank(app.title_stem_index, query_stemmed, "_title_stem", 30, app.page_rank)
        title_weight = 0.3
        body_weight = 0.7

    else:
        #body_res = app.cosine_stem_body.calcCosineSim(query_stemmed, app.body_stem_index, 30)
        title_res = app.BM25_title.search(query_stemmed, 30)
        body_res = app.BM25_body.search(query_stemmed, 30)
        title_weight = 0.7
        body_weight = 0.3

    merged_res = merge_results(title_res, body_res, title_weight=title_weight, text_weight=body_weight,page_rank=app.page_rank , N=10)

    res = [(int(doc_id), app.doc_title_dict.get(doc_id, "not found")) for doc_id, score in merged_res]

    # END SOLUTION
    return jsonify(res)

@app.route("/search_body")
def search_body():
    ''' Returns up to a 100 search results for the query using TFIDF AND COSINE
        SIMILARITY OF THE BODY OF ARTICLES ONLY. DO NOT use stemming. DO USE the 
        staff-provided tokenizer from Assignment 3 (GCP part) to do the 
        tokenization and remove stopwords. 

        To issue a query navigate to a URL like:
         http://YOUR_SERVER_DOMAIN/search_body?query=hello+world
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
    res = []
    res_list = app.cosine_body.calcCosineSim(app.tokenizer.tokenize(query, False), app.index_body, N=100)
    res = [(doc_id, app.doc_title_dict[doc_id]) for doc_id, score in res_list]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_title")
def search_title():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE TITLE of articles, ordered in descending order of the NUMBER OF 
        QUERY WORDS that appear in the title. For example, a document with a 
        title that matches two of the query words will be ranked before a 
        document with a title that matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_title?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res_list = getBinaryListResult(app.index_title, app.tokenizer.tokenize(query, False), "_title", 100)
    res = [(doc_id, app.doc_title_dict[doc_id]) for doc_id, score in res_list]
    # END SOLUTION
    return jsonify(res)

@app.route("/search_anchor")
def search_anchor():
    ''' Returns ALL (not just top 100) search results that contain A QUERY WORD 
        IN THE ANCHOR TEXT of articles, ordered in descending order of the 
        NUMBER OF QUERY WORDS that appear in anchor text linking to the page. 
        For example, a document with a anchor text that matches two of the 
        query words will be ranked before a document with anchor text that 
        matches only one query term. 

        Test this by navigating to the a URL like:
         http://YOUR_SERVER_DOMAIN/search_anchor?query=hello+world
        where YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ALL (not just top 100) search results, ordered from best to 
        worst where each element is a tuple (wiki_id, title).
    '''
    res = []
    query = request.args.get('query', '')
    if len(query) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res_list = getBinaryListResult(app.index_anchor, app.tokenizer.tokenize(query, False), "_anchor", 100)
    res = [(doc_id, app.doc_title_dict[doc_id]) for doc_id, score in res_list]

    # END SOLUTION
    return jsonify(res)

@app.route("/get_pagerank", methods=['POST'])
def get_pagerank():
    ''' Returns PageRank.py values for a list of provided wiki article IDs.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pagerank
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pagerank', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of floats:
          list of PageRank.py scores that correrspond to the provided article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [app.page_rank[str(wiki_id)] for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

@app.route("/get_pageview", methods=['POST'])
def get_pageview():
    ''' Returns the number of page views that each of the provide wiki articles
        had in August 2021.

        Test this by issuing a POST request to a URL like:
          http://YOUR_SERVER_DOMAIN/get_pageview
        with a json payload of the list of article ids. In python do:
          import requests
          requests.post('http://YOUR_SERVER_DOMAIN/get_pageview', json=[1,5,8])
        As before YOUR_SERVER_DOMAIN is something like XXXX-XX-XX-XX-XX.ngrok.io
        if you're using ngrok on Colab or your external IP on GCP.
    Returns:
    --------
        list of ints:
          list of page view numbers from August 2021 that correrspond to the 
          provided list article IDs.
    '''
    res = []
    wiki_ids = request.get_json()
    if len(wiki_ids) == 0:
      return jsonify(res)
    # BEGIN SOLUTION
    res = [app.page_views[wiki_id] for wiki_id in wiki_ids]
    # END SOLUTION
    return jsonify(res)

if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)
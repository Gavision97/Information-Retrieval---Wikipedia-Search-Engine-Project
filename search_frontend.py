"""
Imports:
"""
import pickle
from Tokenizer import Tokenizer
from CosineSim import CosineSim
from PageRankSim import PageRankSim
from flask import Flask, request, jsonify
from google.cloud import storage

BUCKET_NAME = "inverted_indexes_bucket"
client = storage.Client()

class MyFlaskApp(Flask):
    def run(self, host=None, port=None, debug=None, **options):

        self.my_bucket = client.bucket(bucket_name=BUCKET_NAME)
        self.tokenizer = Tokenizer()
        self.page_rank = {}

        for blob in client.list_blobs(BUCKET_NAME):
            if blob.name == "body_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.body_stem_index = pickle.load(openfile)

            elif blob.name == "title_stem_index.pkl":
                with blob.open('rb') as openfile:
                    self.title_stem_index = pickle.load(openfile)

            elif blob.name == "body_dictionary_length.pkl":
                with blob.open('rb') as openfile:
                    self.DL_body = pickle.load(openfile)

            elif blob.name == "doc_l2_norm.pkl":
                with blob.open('rb') as openfile:
                    self.doc_norm = pickle.load(openfile)

            elif blob.name == "title_dictionary.pkl":
                with blob.open('rb') as openfile:
                    self.title_dic = pickle.load(openfile)

            elif blob.name == "pageRank.pkl":
                with blob.open('rb') as openfile:
                    self.page_rank = pickle.load(openfile)

        self.cosine_stem_body = CosineSim(self.body_stem_index,
                                          self.DL_body,
                                          "body_stem_index_directory",
                                          self.doc_norm)

        self.page_rank_sim = PageRankSim(self.title_stem_index,
                                         "title_stem_index_dictionary",
                                         self.page_rank)

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

    cosine_sim_body_weight = 0.8
    title_page_rank_weight = 0.2

    '''
    First, tokenize the query.
    Convert the text to lowercase, remove stopwords, and apply stemming.
    '''
    query_stemmed = list(set(app.tokenizer.tokenize(query, True)))

    '''
    The next step is to return N number of documents along with their cosine similarity with the query. 
    For more documentation, refer to 'CosineSimOriginal.py'.
    '''
    body_res_ls, body_res_dict = app.cosine_stem_body.get_top_N_docs_by_cosine_similarity(query_stemmed, N=150)
    title_res = app.page_rank_sim.get_top_N_docs_by_title_and_page_rank(query_stemmed, N=150, cosine_sim_on_body_index_dict=body_res_dict)

    merged_res_dict = merged_results_and_sort(body_res_ls, title_res, cosine_sim_body_weight, title_page_rank_weight)


    # Extract each document title by its 'doc_id' value
    N = 30
    res = [(int(doc_id), app.title_dic.get(doc_id, "not found")) for doc_id, score in merged_res_dict][:N]

    # END SOLUTION
    return jsonify(res)


def merged_results_and_sort(body_list, title_list, body_weight, title_weight) -> None:
    body_dict_sorted_by_id = sorted(body_list, key=lambda x: x[0], reverse=True)
    title_dict_sorted_by_id = sorted(title_list, key=lambda x: x[0], reverse=True)

    result_dict = {}

    for (doc_id_body, score_body) in body_dict_sorted_by_id:

        if doc_id_body in [doc_id for doc_id, _ in title_dict_sorted_by_id]:
            score_title = next(score for doc_id, score in title_dict_sorted_by_id if doc_id == doc_id_body)
            merged_score = (score_body * body_weight) + (score_title * title_weight)
        else:
            merged_score = score_body

        # Add merged score to result dictionary
        result_dict[doc_id_body] = merged_score

    return sorted([(doc_id, score) for doc_id, score in result_dict.items()], key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    # run the Flask RESTful API, make the server publicly available (host='0.0.0.0') on port 8080
    app.run(host='0.0.0.0', port=8080, debug=True)

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *


class Tokenizer:

    def __init__(self):
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "history", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became"]

        self.all_stopwords = english_stopwords.union(corpus_stopwords)
        self.RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
        self.stemmer = PorterStemmer()

    def tokenize(self, text, to_stem):
        clean_text = []
        text = text.lower()
        tokens = [token.group() for token in self.RE_WORD.finditer(text)]
        for token in tokens:
            if token not in self.all_stopwords:
                if to_stem:
                    token = self.stemmer.stem(token)
                clean_text.append(token)
        return clean_text
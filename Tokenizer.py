import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import *
nltk.download('stopwords')


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

    def tokenize(self, text, stem=False):
        cleaned_text = []
        tokens = [token.group() for token in self.RE_WORD.finditer(text.lower())]
        for token in tokens:
            if token not in self.all_stopwords:
                if stem:
                    token = self.stemmer.stem(token)
                cleaned_text.append(token)
        return cleaned_text

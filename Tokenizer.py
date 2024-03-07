from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import *
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


class Tokenizer:

    def __init__(self):
        english_stopwords = frozenset(stopwords.words('english'))
        corpus_stopwords = ["category", "references", "also", "external", "links",
                            "may", "first", "see", "people", "one", "two",
                            "part", "thumb", "including", "second", "following",
                            "many", "however", "would", "became", "yet", "oh", "even",
                            "within", "beyond", "hey", "since", "without", "ugh", "wow",
                            "ah", "already", "oops", "really", "still", "hmm", "among"]


        self.all_stopwords = english_stopwords.union(corpus_stopwords)
        self.stemmer = PorterStemmer()
        self.special_words = ['3d', '4k', 'ip', 'js', 'ai', 'vr', 'ar', 'dl', 'ml', '09', '11', '9']
        
    def get_word_pattern(self):
        word_pattern = r"(?:(?<=^)|(?<=\s))(\w+[-']?\w+([-']\w+)*)[,.']?(?<![,.!])"
        return word_pattern


    def tokenize(self, text):
      RE_TOKENIZE = re.compile(rf"""
      (
          # Words
          (?P<WORD>{self.get_word_pattern()})
          # space
          |(?P<SPACE>[\s\t\n]+)
          # everything else
          |(?P<OTHER>\w+))""",  re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)

      return [self.stemmer.stem(v) for match in RE_TOKENIZE.finditer(text) for k, v in match.groupdict().items() if v is not None and k != 'SPACE' and bool(re.match(r'^[a-zA-Z0-9]+$', v)) and (len(v) > 2 or v.lower() in self.special_words) and v.lower() not in self.all_stopwords and len(v) <= 24] 

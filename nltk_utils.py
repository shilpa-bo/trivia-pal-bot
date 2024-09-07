import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

# nltk.download('punkt_tab')

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenized_sentence, all_words):
    """
    If a word in the tokenized sentence is avaialable in all_words, then the position is 1
    otherwise the position is 0
    """
    # stem tokenized sentence
    tokenized_sentence = [stem(w) for w in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, w, in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx] = 1.0
    return bag

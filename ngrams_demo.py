import numpy as np
import gensim
from gensim.models import FastText
from gensim.models.utils_any2vec import _save_word2vec_format, _load_word2vec_format, _compute_ngrams, _ft_hash

def compute_ngrams(word, min_n, max_n):
    BOW, EOW = ('<', '>')  # Used by FastText to attach to all words as prefix and suffix
    extended_word = BOW + word + EOW
    ngrams = []
    for ngram_length in range(min_n, min(len(extended_word), max_n) + 1):
        for i in range(0, len(extended_word) - ngram_length + 1):
            ngrams.append(extended_word[i:i + ngram_length])
    return ngrams

def FastTextNgramsVector(fasttext_model):
    fasttext_word_list = fasttext_model.wv.vocab.keys()
    NgramsVector = {}
    ngram_weights = fasttext_model.wv.vectors_ngrams # (10, 4)
    for word in fasttext_word_list:
        ngrams = _compute_ngrams(word,min_n = fasttext_model.wv.min_n,max_n = fasttext_model.wv.max_n)
        for ngram in ngrams:
            ngram_hash = _ft_hash(ngram) % fasttext_model.wv.bucket
            if ngram_hash in fasttext_model.wv.hash2index:
                NgramsVector[ngram] = ngram_weights[fasttext_model.wv.hash2index[ngram_hash]]
    return NgramsVector


def word_vec(self, word, use_norm=False):
    if word in self.vocab:
        return super(gensim.models.keyedvectors.FastTextKeyedVectors, self).word_vec(word, use_norm)
    else:
        # from gensim.models.fasttext import compute_ngrams
        word_vec = np.zeros(self.vectors_ngrams.shape[1], dtype=np.float32)
        ngrams = _compute_ngrams(word, self.min_n, self.max_n)
        if use_norm:
            ngram_weights = self.vectors_ngrams_norm
        else:
            ngram_weights = self.vectors_ngrams
        ngrams_found = 0
        for ngram in ngrams:
            ngram_hash = _ft_hash(ngram) % self.bucket
            if ngram_hash in self.hash2index:
                word_vec += ngram_weights[self.hash2index[ngram_hash]]
                ngrams_found += 1
        if word_vec.any():
            return word_vec / max(1, ngrams_found)
        else:  # No ngrams of the word are present in self.ngrams
            raise KeyError('all ngrams for word %s absent from model' % word)

sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]

model = FastText(sentences,  size=4, window=3, min_count=1, iter=10, min_n = 3 , max_n = 6,word_ngrams = 0)
print(FastTextNgramsVector(model))
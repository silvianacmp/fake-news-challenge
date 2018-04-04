import numpy as np
from feature_generators.feature_generator import FeatureGenerator

OOV = 'OOV'


class BagOfWordsFeatureGenerator(FeatureGenerator):

    def __init__(self):
        super().__init__()
        self.vocab = None

    @staticmethod
    def get_vocab(articles, stances):
        # get vocab
        vocab = {OOV: 0}
        count = 1
        for a in articles.values():
            for w in a['article_tokens']:
                if w not in vocab:
                    vocab[w] = count
                    count += 1
        for s in stances:
            for w in s['headline_tokens']:
                if w not in vocab:
                    vocab[w] = count
                    count += 1
        print('Vocab size: {}'.format(len(vocab)))
        return vocab

    @staticmethod
    def get_bow(vocab, tokens):
        bow = np.zeros(len(vocab))
        for t in tokens:
            if t in vocab:
                bow[vocab[t]] += 1.0
            else:
                bow[OOV] += 1.0

        # normalize by dividing with tokens length
        bow /= len(tokens)
        return bow

    def process(self, articles, stances):
        if self.vocab is None:
            self.vocab = self.get_vocab(articles, stances)
        print('Extracting bag of words features')
        for a in articles.values():
            a['article_bow'] = self.get_bow(self.vocab, a['article_tokens'])

        for s in stances:
            s['headline_bow'] = self.get_bow(self.vocab, s['headline_tokens'])
        print('Done extracting bag of words features')
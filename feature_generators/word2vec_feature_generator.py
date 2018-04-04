from feature_generators.feature_generator import FeatureGenerator
import numpy as np
import gensim


class Word2VecFeatureGenerator(FeatureGenerator):

    def __init__(self):
        super().__init__()
        self.model = gensim.models.KeyedVectors.load_word2vec_format('../word2vec/GoogleNews-vectors-negative300.bin', binary=True)
        print('Loaded word2vec model')

    def process(self, articles, stances):
        print('Extracting word2vec features')
        for a in articles.values():
            accum = np.zeros(300)
            count = 0

            for w in a['article_tokens']:
                if w in self.model.wv:
                    accum += self.model.wv[w]
                    count += 1
            a['article_word2vec_mean'] = accum / count

        for s in stances:
            accum = np.zeros(300)
            count = 0

            for w in s['headline_tokens']:
                if w in self.model.wv:
                    accum += self.model.wv[w]
                    count += 1

            s['headline_word2vec_mean'] = accum / count
        print('Done extracting word2vec features')
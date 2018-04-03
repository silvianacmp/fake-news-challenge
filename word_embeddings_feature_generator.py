from feature_generator import FeatureGenerator
import numpy as np


class WordEmbeddingsFeatureGenerator(FeatureGenerator):

    def __init__(self, path):
        super().__init__()
        vocab, embd = self.load_word_embd(path)
        self.vocab = vocab
        self.embd = embd

    @staticmethod
    def load_word_embd(path):
        vocab = {}
        embd = []
        file = open(path, 'r')
        index = 2

        for line in file.readlines():
            row = line.strip().split(' ')
            vocab[row[0]] = index
            index += 1
            vect = np.array([float(n) for n in row[1:]])
            embd.append(vect)
        print('Loaded Embeddings!')
        file.close()
        embd = np.asanyarray(embd)
        return vocab, embd

    def process(self, articles, stances):
        print('Extracting word embeddings features')
        for a in articles.values():
            accum = np.zeros(self.embd.shape[1])
            count = 0

            for w in a['article_tokens']:
                if w in self.vocab:
                    accum += self.embd[self.vocab[w]]
                    count += 1

            a['article_embd_mean'] = accum / count

        for s in stances:
            accum = np.zeros(self.embd.shape[1])
            count = 0

            for w in s['headline_tokens']:
                if w in self.vocab:
                    accum += self.embd[self.vocab[w]]
                    count += 1

            s['headline_embd_mean'] = accum / count
        print('Done extracting word embeddings features')

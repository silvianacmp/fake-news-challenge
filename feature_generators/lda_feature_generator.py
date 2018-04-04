from feature_generators.feature_generator import FeatureGenerator
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


class LdaFeatureGenerator(FeatureGenerator):
    def __init__(self, article_vect_col, stance_vect_col, n_components):
        super().__init__()
        self.article_vect_col = article_vect_col
        self.stance_vect_col = stance_vect_col
        self.n_components = n_components
        self.lda = None

    def process(self, articles, stances):
        article_vect = np.array([a[self.article_vect_col] for a in articles.values()])
        stance_vect = np.array([s[self.stance_vect_col] for s in stances])
        all = np.vstack([article_vect, stance_vect])
        all = all.reshape((all.shape[0], all.shape[2]))
        print('Extracting lda features')
        if self.lda is None:
            self.lda = LatentDirichletAllocation(n_components=self.n_components)
            y = self.lda.fit_transform(all)
        else:
            y = self.lda.transform(all)

        for i, k in enumerate(articles.keys()):
            articles[k]['article_lda'] = y[i]

        for i, stance in enumerate(stances):
            stance['headline_lda'] = y[i + len(articles)]
        print('Done extracting lda features')

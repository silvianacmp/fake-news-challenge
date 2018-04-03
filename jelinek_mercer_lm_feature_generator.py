from feature_generator import FeatureGenerator
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from tqdm import tqdm

class JelinekMercerLmFeatureGenerator(FeatureGenerator):

    def __init__(self, collection_prop=0.5):
        super().__init__()
        self.collection_prop = collection_prop
        self.count_vectorizer = None
        self.collection_counts = None

    def process(self, articles, stances):
        stance_vect = [s['Headline'] for s in stances]
        article_vect = [a['articleBody'] for a in articles.values()]
        all = stance_vect + article_vect

        if self.count_vectorizer is None:
            self.count_vectorizer = CountVectorizer(stop_words='english')
            counts = self.count_vectorizer.fit_transform(all)
        else:
            counts = self.count_vectorizer.transform(all)

        normalized_counts = counts / np.sum(counts, axis=1)

        if self.collection_counts is None:
            self.collection_counts = np.sum(counts, axis=0) / np.sum(counts)

        final_counts = ((1 - self.collection_prop) * normalized_counts
                        + self.collection_prop * np.broadcast_to(self.collection_counts, normalized_counts.shape))

        tmp_article_dict = {}
        for i, k in enumerate(articles.keys()):
            tmp_article_dict[k] = i + len(stances)

        for i, stance in tqdm(enumerate(stances)):
            headline_prob = np.array(final_counts[i]).flatten()
            article_prob = np.array(final_counts[tmp_article_dict[stance['Body ID']]]).flatten()
            kl = -np.sum(headline_prob * np.log(article_prob / headline_prob))
            stance['kl_jelinek_mercer'] = kl
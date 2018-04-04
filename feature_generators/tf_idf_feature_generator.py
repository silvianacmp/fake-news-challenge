from feature_generators.feature_generator import FeatureGenerator
from sklearn.feature_extraction.text import TfidfVectorizer


class TfIdfFeatureGenerator(FeatureGenerator):
    def __init__(self, stop_words):
        super().__init__()
        self.stop_words = stop_words
        self.tf_idf_vectorizer = None

    def process(self, articles, stances):
        documents = [a['articleBody'] for a in articles.values()] + [s['Headline'] for s in stances]
        print('Extracting tf-idf features')
        if self.tf_idf_vectorizer is None:
            self.tf_idf_vectorizer = TfidfVectorizer(stop_words=self.stop_words)
            tf_idf = self.tf_idf_vectorizer.fit_transform(documents)
        else:
            tf_idf = self.tf_idf_vectorizer.transform(documents)

        for i, a in enumerate(articles.values()):
            a['article_tf_idf'] = tf_idf[i].todense()

        for i, s in enumerate(stances):
            s['headline_tf_idf'] = tf_idf[len(articles) + i].todense()
        print('Done extracting tf-idf features')


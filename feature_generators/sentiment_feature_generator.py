from nltk.sentiment.vader import SentimentIntensityAnalyzer
from feature_generators.feature_generator import FeatureGenerator
from nltk.tokenize import sent_tokenize
import pandas as pd


class SentimentFeatureGenerator(FeatureGenerator):

    def __init__(self):
        super().__init__()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def _sentiment_mean(self, doc):
        acc = []
        sentences = sent_tokenize(doc)
        for s in sentences:
            acc.append(self.sentiment_analyzer.polarity_scores(s))
        return pd.DataFrame(acc).mean()

    def process(self, articles, stances):
        print('Extracting sentiment polarity features')
        article_tmp = {}
        for k, v in articles.items():
            article_sentiment = self._sentiment_mean(v['articleBody'])
            article_tmp[k] = article_sentiment

        for s in stances:
            headline_sentiment = self._sentiment_mean(s['Headline'])
            s['headline_compound'] = headline_sentiment['compound']
            s['headline_neg'] = headline_sentiment['neg']
            s['headline_neu'] = headline_sentiment['neu']
            s['headline_pos'] = headline_sentiment['pos']

            a = article_tmp[s['Body ID']]
            s['article_compound'] = a['compound']
            s['article_neg'] = a['neg']
            s['article_neu'] = a['neu']
            s['article_pos'] = a['pos']
        print('Done extracting sentiment polarity features')

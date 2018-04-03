import pandas as pd
from baseline.utils.dataset import DataSet
from dataset_splitter import get_splits
from preprocesser import preprocess_dataset, lower
from utils import *
from bag_of_words_feature_generator import BagOfWordsFeatureGenerator
from word2vec_feature_generator import Word2VecFeatureGenerator
from tf_idf_feature_generator import TfIdfFeatureGenerator
from svd_feature_generator import SvdFeatureGenerator
from lda_feature_generator import LdaFeatureGenerator
from word_embeddings_feature_generator import WordEmbeddingsFeatureGenerator
from jelinek_mercer_lm_feature_generator import JelinekMercerLmFeatureGenerator
from sentiment_feature_generator import SentimentFeatureGenerator
from dirichlet_lm_feature_generator import DirichletLmFeatureGenerator
from nltk.corpus import stopwords

if __name__ == "__main__":
    stopwords = set(stopwords.words('english'))

    dataset = DataSet(name='train', path='data')
    articles, train_stances, test_stances = get_splits(dataset, train_split=0.9)

    preprocess_dataset(articles,
                       text_col='articleBody',
                       tokens_col='article_tokens',
                       ops=[lower],
                       stop_words=stopwords)
    preprocess_dataset(train_stances,
                       text_col='Headline',
                       tokens_col='headline_tokens',
                       ops=[lower],
                       stop_words=stopwords)
    preprocess_dataset(test_stances,
                       text_col='Headline',
                       tokens_col='headline_tokens',
                       ops=[lower],
                       stop_words=stopwords)

    # bag_of_words = BagOfWordsFeatureGenerator()
    # bag_of_words.process(articles, train_stances)
    # bag_of_words.process(articles, test_stances)
    #
    tf_idf = TfIdfFeatureGenerator(stop_words=list(stopwords))
    tf_idf.process(articles, train_stances)
    tf_idf.process(articles, test_stances)

    word2vec = Word2VecFeatureGenerator()
    word2vec.process(articles, train_stances)
    word2vec.process(articles, test_stances)

    word_embeddings = WordEmbeddingsFeatureGenerator(path='./glove/glove.6B.100d.txt')
    word_embeddings.process(articles, train_stances)
    word_embeddings.process(articles, test_stances)

    svd = SvdFeatureGenerator(article_vect_col='article_tf_idf', stance_vect_col='headline_tf_idf', n_components=50)
    svd.process(articles, train_stances)
    svd.process(articles, test_stances)

    lda = LdaFeatureGenerator(article_vect_col='article_tf_idf', stance_vect_col='headline_tf_idf', n_components=10)
    lda.process(articles, train_stances)
    lda.process(articles, test_stances)

    jelinek_mercer_lm = JelinekMercerLmFeatureGenerator(collection_prop=0.5)
    jelinek_mercer_lm.process(articles, train_stances)
    jelinek_mercer_lm.process(articles, test_stances)

    dirichlet_lm = DirichletLmFeatureGenerator()
    dirichlet_lm.process(articles, train_stances)
    dirichlet_lm.process(articles, test_stances)

    sentiment = SentimentFeatureGenerator()
    sentiment.process(articles, train_stances)
    sentiment.process(articles, test_stances)

    cos_similarity(articles, train_stances, article_col='article_word2vec_mean',
                   stance_col='headline_word2vec_mean', similarity_col='cosine_word2vec_mean')
    cos_similarity(articles, test_stances, article_col='article_word2vec_mean',
                   stance_col='headline_word2vec_mean', similarity_col='cosine_word2vec_mean')

    cos_similarity(articles, train_stances, article_col='article_bow',
                   stance_col='headline_bow', similarity_col='cosine_bow')
    cos_similarity(articles, test_stances, article_col='article_bow',
                   stance_col='headline_bow', similarity_col='cosine_bow')

    cos_similarity(articles, train_stances, article_col='article_embd_mean',
                   stance_col='headline_embd_mean', similarity_col='cosine_embd_mean')
    cos_similarity(articles, test_stances, article_col='article_embd_mean',
                   stance_col='headline_embd_mean', similarity_col='cosine_embd_mean')

    cos_similarity(articles, train_stances, article_col='article_tf_idf',
                   stance_col='headline_tf_idf', similarity_col='cosine_tf_idf')
    cos_similarity(articles, test_stances, article_col='article_tf_idf',
                   stance_col='headline_tf_idf', similarity_col='cosine_tf_idf')

    cos_similarity(articles, train_stances, article_col='article_svd',
                   stance_col='headline_svd', similarity_col='cosine_svd')
    cos_similarity(articles, test_stances, article_col='article_svd',
                   stance_col='headline_svd', similarity_col='cosine_svd')

    cos_similarity(articles, train_stances, article_col='article_lda',
                   stance_col='headline_lda', similarity_col='cosine_lda')
    cos_similarity(articles, test_stances, article_col='article_lda',
                   stance_col='headline_lda', similarity_col='cosine_lda')

    features = ['Body ID', 'cosine_bow', 'cosine_tf_idf', 'cosine_embd_mean', 'cosine_word2vec_mean', 'cosine_lda',
                'cosine_svd', 'kl_jelinek_mercer', 'kl_dirichlet', 'article_compound', 'article_neg', 'article_pos',
                'article_neu', 'headline_compound', 'headline_neg', 'headline_pos', 'headline_neu', 'Stance']

    train_stances_df = pd.DataFrame(train_stances)
    test_stances_df = pd.DataFrame(test_stances)

    train_stances_df[features].to_csv('train_stances_feats.df')
    test_stances_df[features].to_csv('test_stances_feats.df')

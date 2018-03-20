import pandas as pd
from baseline.utils.dataset import DataSet
from dataset_splitter import get_splits
from preprocesser import preprocess_dataset
from embedder import bag_of_words, cos_similarity, word_embeddings_mean, tf_idf
from language_model import jelinek_mercer_lm
from nltk.corpus import stopwords

if __name__ == "__main__":
    stopwords = set(stopwords.words('english'))

    dataset = DataSet(name='train', path='data')
    preprocess_dataset(dataset.articles, text_col='articleBody', tokens_col='article_tokens')
    preprocess_dataset(dataset.stances, text_col='Headline', tokens_col='headline_tokens')
    # bag_of_words(dataset.articles, dataset.stances)
    # word_embeddings_mean(dataset.articles, dataset.stances, path='./glove/glove.6B.100d.txt')
    # tf_idf(dataset.articles, dataset.stances, stop_words=list(stopwords))
    #
    # cos_similarity(dataset.articles, dataset.stances, article_col='article_bow',
    #                stance_col='headline_bow', similarity_col='cosine_bow')
    # cos_similarity(dataset.articles, dataset.stances, article_col='article_embd_mean',
    #                stance_col='headline_embd_mean', similarity_col='cosine_embd_mean')
    # cos_similarity(dataset.articles, dataset.stances, article_col='article_tf_idf',
    #                stance_col='headline_tf_idf', similarity_col='cosine_tf_idf')
    #
    jelinek_mercer_lm(dataset.articles, dataset.stances)

    stances_df = pd.DataFrame(dataset.stances)
    # print('Cosine similarity for bag of words:')
    # print(stances_df.groupby(by='Stance')['cosine_bow'].mean())
    #
    # print('Cosine similarity for word embeddings mean:')
    # print(stances_df.groupby(by='Stance')['cosine_embd_mean'].mean())
    #
    # print('Cosine similarity for tf-idf:')
    # print(stances_df.groupby(by='Stance')['cosine_tf_idf'].mean())

    print('KL for Jelinek Mercer:')
    print(stances_df.groupby(by='Stance')['kl'].mean())
    articles, train_dataset, test_dataset = get_splits(dataset, train_split=0.9)

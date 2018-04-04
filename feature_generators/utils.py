from sklearn.metrics.pairwise import cosine_similarity


def cos_similarity(articles, stances, article_col, stance_col, similarity_col):
    for s in stances:
        similarity = cosine_similarity(articles[s['Body ID']][article_col].reshape(1, -1), s[stance_col].reshape(1, -1))
        s[similarity_col] = similarity[0, 0]

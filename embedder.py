import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

OOV = 'OOV'


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


def bag_of_words(articles, stances, vocab=None):
    if vocab is None:
        vocab = get_vocab(articles, stances)

    for a in articles.values():
        a['article_bow'] = get_bow(vocab, a['article_tokens'])

    for s in stances:
        s['headline_bow'] = get_bow(vocab, s['headline_tokens'])

    return vocab


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


def word_embeddings_mean(articles, stances, path):
    vocab, embd = load_word_embd(path)
    for a in articles.values():
        accum = np.zeros(embd.shape[1])
        count = 0

        for w in a['article_tokens']:
            if w in vocab:
                accum += embd[vocab[w]]
                count += 1

        a['article_embd_mean'] = accum / count

    for s in stances:
        accum = np.zeros(embd.shape[1])
        count = 0

        for w in s['headline_tokens']:
            if w in vocab:
                accum += embd[vocab[w]]
                count += 1

        s['headline_embd_mean'] = accum / count
    return vocab, embd


def word_embeddings_tanh(articles, stances, path):
    vocab, embd = load_word_embd(path)
    for a in articles.values():
        accum = np.zeros(embd.shape[1])
        count = 0

        for w1, w2 in zip(a['article_tokens'][:-1], a['article_tokens'][1:]):
            if w1 in vocab and w2 in vocab:
                accum += np.tanh(embd[vocab[w1]] + embd[vocab[w1]])
                count += 1

        a['article_embd_tanh'] = accum / count

    for s in stances:
        accum = np.zeros(embd.shape[1])
        count = 0

        for w1, w2 in zip(s['headline_tokens'][:-1], s['headline_tokens'][1:]):
            if w1 in vocab and w2 in vocab:
                accum += np.tanh(embd[vocab[w1]] + embd[vocab[w1]])
                count += 1

        s['headline_embd_tanh'] = accum / count
    return vocab, embd


def tf_idf(articles, stances, stop_words, tf_idf_vectorizer=None):
    if tf_idf_vectorizer is None:
        documents = [a['articleBody'] for a in articles.values()] + [s['Headline'] for s in stances]

        tf_idf_vectorizer = TfidfVectorizer(stop_words=stop_words)
        tf_idf_vectorizer.fit(documents)

    for a in articles.values():
        a['article_tf_idf'] = tf_idf_vectorizer.transform([a['articleBody']]).todense()

    for s in stances:
        s['headline_tf_idf'] = tf_idf_vectorizer.transform([s['Headline']]).todense()


def cos_similarity(articles, stances, article_col, stance_col, similarity_col):
    for s in stances:
        similarity = cosine_similarity(articles[s['Body ID']][article_col].reshape(1, -1), s[stance_col].reshape(1, -1))
        s[similarity_col] = similarity[0, 0]

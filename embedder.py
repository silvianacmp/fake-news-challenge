import numpy as np

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


def bag_of_words(articles, stances):
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
    print('Bag-of-words vocab size: {}'.format(len(vocab)))
    for a in articles.values():
        a['article_bow'] = get_bow(vocab, a['article_tokens'])

    for s in stances:
        s['headline_bow'] = get_bow(vocab, s['headline_tokens'])

    return vocab

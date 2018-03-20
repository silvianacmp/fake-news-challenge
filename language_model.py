from tqdm import tqdm
import collections
import numpy as np


def jelinek_mercer_lm(articles, stances, collection_prop=0.2, collection_counts=None, collection_n=None):
    update_collection = False
    if collection_counts is None:
        # get collection counts
        collection_counts = collections.defaultdict(float)
        collection_n = 0
        update_collection = True

    for a in articles.values():
        collection_n += len(a['article_tokens'])

        article_counts = collections.defaultdict(float)
        for w in a['article_tokens']:
            article_counts[w] += 1

            if update_collection:
                collection_counts[w] += 1
        a['article_counts'] = article_counts

    for s in stances:
        collection_n += len(s['headline_tokens'])

        headline_counts = collections.defaultdict(float)
        for w in s['headline_tokens']:
            headline_counts[w] += 1

            if update_collection:
                collection_counts[w] += 1
        s['headline_counts'] = headline_counts

    # Get KL for each headline-article pair
    print('Computing KL')
    for s in tqdm(stances):
        acc = 0
        headline_counts = s['headline_counts']
        headline_n = len(s['headline_tokens'])

        article_counts = articles[s['Body ID']]['article_counts']
        article_n = len(articles[s['Body ID']]['article_counts'])

        for w, c in collection_counts.items():
            headline_prob = (1 - collection_prop) * (headline_counts[w] / headline_n) + collection_prop * (
                        c / collection_n)
            article_prob = (1 - collection_prop) * (article_counts[w] / article_n) + collection_prop * (
                        c / collection_n)
            acc += headline_prob * np.log(article_prob / headline_prob)

        s['kl'] = (-1) * acc

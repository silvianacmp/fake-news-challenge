from functools import reduce
import nltk

lemmatizer = nltk.WordNetLemmatizer()


def lemmatize(w):
    return lemmatizer.lemmatize(w)


def lower(w):
    return w.lower()


def preprocess(sentence, ops):
    return [reduce(lambda x, y: y(x), [w] + ops) for w in nltk.word_tokenize(sentence)]


def preprocess_dataset(dataset, text_col, tokens_col, ops=None):
    if ops is None:
        ops = [lemmatize, lower]
    if isinstance(dataset, dict):
        iter = dataset.items()
    elif isinstance(dataset, list):
        iter = enumerate(dataset)

    for _, d in iter:
        d[tokens_col] = preprocess(d[text_col], ops=ops)



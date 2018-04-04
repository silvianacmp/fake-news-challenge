"""
Scoring:
  +0.25 for each correct unrelated
  +0.25 for each correct related (label is any of agree, disagree, discuss)
  +0.75 for each correct agree, disagree, discuss
"""
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
import itertools

LABELS = ['agree', 'discuss', 'disagree', 'unrelated']
RELATED = [0, 1, 2]
UNRELATED = 3


def official_score(y, pred):
    score = 0
    for i in range(len(y)):
        y_, pred_ = y[i], pred[i]
        if y_ == pred_:
            score += 0.25
            if y_ != UNRELATED:
                score += 0.50
        if y_ in RELATED and pred_ in RELATED:
            score += 0.25
    return score


def max_official_score(y):
    max_score = 0.0
    for i in range(len(y)):
        if y[i] == UNRELATED:
            max_score += 0.25
        else:
            max_score += 0.75
    return max_score


def baseline_official_score(y):
    return len(y) * 0.25


def official_score_relative(y, pred):
    max_score = max_official_score(y)
    score = official_score(y, pred)
    return score / max_score


def accuracy(y, pred):
    return np.mean(np.equal(y, pred))


def class_wise_f1(y, pred):
    return f1_score(y, pred, average=None)


def mean_f1(y, pred):
    return f1_score(y, pred, average='macro')


def get_confusion_matrix(y, pred):
    cnf = confusion_matrix(y, pred)
    return cnf


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    Credits: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def score_report(y, pred):
    off_sc = official_score(y, pred)
    max_sc = max_official_score(y)
    base_sc = baseline_official_score(y)
    off_rel_sc = official_score_relative(y, pred)
    acc = accuracy(y, pred)
    class_f1 = class_wise_f1(y, pred)
    avg_f1 = mean_f1(y, pred)
    cnf = get_confusion_matrix(y, pred)
    print('Maximum score: {} \nBaseline (all unrelated) score: {} \nPredictions score: {}'.format(max_sc, base_sc,
                                                                                                  off_sc))
    print('\nPredictions score r'
          'elative to max score: {}%'.format(off_rel_sc))
    print('\nAccuracy: {}'.format(acc))
    print('\nClass-wise F1: \nAgree: {}\nDiscuss: {}\nDisagree: {}\nUnrelated: {}'.format(class_f1[0],
                                                                                          class_f1[1],
                                                                                          class_f1[2],
                                                                                          class_f1[3]))
    print('\nMean F1: {}'.format(avg_f1))
    print('\nConfusion Matrix:')
    print(cnf)

    plot_confusion_matrix(cnf, classes=LABELS)


if __name__ == '__main__':
    y = np.random.randint(0, 4, size=(100,))
    pred = np.full((100,), UNRELATED)
    score_report(y, pred)

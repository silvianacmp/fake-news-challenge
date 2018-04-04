from sklearn.feature_selection import SelectKBest
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
import numpy as np


def mutual_information_ranker(X, y, col_labels):
    scorer = SelectKBest(mutual_info_classif, k=1)
    scorer.fit(X, y)
    sorted_col_idxs = np.argsort(scorer.scores_)
    sorted_cols = [(col_labels[idx], scorer.scores_[idx]) for idx in sorted_col_idxs]
    return sorted_cols


def tree_based_ranker(X, y, col_labels):
    scorer = GradientBoostingClassifier()
    scorer.fit(X, y)
    sorted_col_idxs = np.argsort(scorer.feature_importances_)
    sorted_cols = [(col_labels[idx], scorer.feature_importances_[idx]) for idx in sorted_col_idxs]
    return sorted_cols

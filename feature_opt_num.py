"""
===================================================
Recursive feature elimination with cross-validation
===================================================
"""
# print(__doc__)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import make_classification


def opt_features(X, y, model, score):
    rfecv = RFECV(estimator=model, step=1, cv=StratifiedKFold(2), scoring=score)
    rfecv.fit(X, y)
    print("Optimal number of features : %d" % rfecv.n_features_)

    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation {} score".format(score))
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()


if __name__ == '__main__':
    # Load data
    X = None
    y = None

    model = SVC(kernel="linear") # other models? 
    opt_features(X, y, model, score='accuracy')

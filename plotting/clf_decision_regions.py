"""
==================================================
Plot the decision boundaries of a Classifier
==================================================
"""
# print(__doc__)

from itertools import product
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier


def train_clf(X,y):
    '''Train classifiers'''
    clf1 = DecisionTreeClassifier(max_depth=4)
    clf2 = KNeighborsClassifier(n_neighbors=5)
    clf3 = SVC(kernel='rbf', probability=True)
    clf4 = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
                            voting='soft', weights=[2, 1, 2])
    clf1.fit(X, y)
    clf2.fit(X, y)
    clf3.fit(X, y)
    clf4.fit(X, y)
    return clf1, clf2, clf3, clf4

def plot_dec_reg(X, y, clf1, clf2, clf3, clf4):
    '''Plot decision regions'''
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

    for idx, clf, tt in zip(product([0, 1], [0, 1]), # ((x,y) for x in A for y in B)
                            [clf1, clf2, clf3, clf4],
                            [clf1.__class__.__name__, clf2.__class__.__name__, clf3.__class__.__name__, clf4.__class__.__name__]):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        # Z clf.predict_proba()[:,1]
        Z = Z.reshape(xx.shape)
        axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
        axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k')
        axarr[idx[0], idx[1]].set_title(tt)
    plt.show()


if __name__ == '__main__':
    # Load data
    X = None
    y = None

    clf1, clf2, clf3, clf4 = train_clf(X,y)
    plot_dec_reg(X, y, clf1, clf2, clf3, clf4)

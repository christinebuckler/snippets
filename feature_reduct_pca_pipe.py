"""
=========================================================
Pipelining: chaining a PCA and a logistic regression
=========================================================
Use a GridSearchCV to set the dimensionality of the PCA.
The PCA does an unsupervised dimensionality reduction, while the logistic
regression does the prediction.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model, decomposition
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV


def pca_pipe(X, y, model):
    pca = decomposition.PCA()
    pipe = Pipeline(steps=[('pca', pca), ('model', model)])
    pca.fit(X)

    n_components = np.arange(1,X.shape[1])
    Cs = np.logspace(-4, 4, 10)
    # Parameters of pipelines can be set using ‘__’ separated parameter names:
    estimator = GridSearchCV(pipe, dict(pca__n_components=n_components, model__C=Cs))
    estimator.fit(X, y)
    return pca, estimator, n_components

def plot_pca_range(pca, estimator, n_components):
    '''Plot PCA spectrum'''
    plt.figure(1, figsize=(4, 3))
    plt.clf()
    plt.axes([.2, .2, .7, .7])
    plt.plot(pca.explained_variance_, linewidth=2)
    plt.axis('tight')
    plt.xlabel('n_components')
    plt.ylabel('explained variance')
    plt.axvline(estimator.best_estimator_.named_steps['pca'].n_components,
                linestyle=':', label='{} components chosen'.format(estimator.best_estimator_.named_steps['pca'].n_components))
    plt.legend(prop=dict(size=12))
    plt.show()


if __name__ == '__main__':
    X = None
    y = None

    logistic = linear_model.LogisticRegression()
    pca, estimator, n_components = pca_pipe(X, y, model=logistic)
    plot_pca_range(pca, estimator, n_components)

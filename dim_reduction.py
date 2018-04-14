# import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV
from sklearn.decomposition import PCA
from sklearn.lda import LDA

mpl.rcParams.update({
    'font.size'           : 12.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'small',})

def regularization(X_train, y_train, X_test, y_test):
    lasso = LassoCV(cv=5, n_jobs=-1, selection='random')
    ridge = RidgeCV(cv=5, n_jobs=-1, selection='random')
    enet = ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, .99, 1], cv=5, n_jobs=-1, selection='random')
    models = [lasso, ridge, enet]
    for model in models:
        model.fit(X_train, y_train)
        print('R2:',model.score(X_test, y_test)
        print('alpha:',model.alpha_)
        print('parameters:', model.get_params())

def pca(X_train, X_test, perc=.99):
    # linear relationships only
    pca = PCA(n_components=perc) # minimum percentage of variance explained
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)
    # percentages = pca.explained_variance_ratio_ # (features ordered from highest to lowest)
    return X_train, X_test

def lda(X_train, y_train, X_test, y_test):
    # Linear Discriminant Analysis (LDA) additionally maximizes the spread between classes
    lda = LDA()
    X_train = lda.fit(X_train, y_train).transform(X_train)
    X_test = lda.transform(X_test)
    y_pred = lda.predict(X_test)
    y_prob = lda.predict_proba(X_test)
    print('Mean Accuracy:', lda.score(X_test, y_test))
    print('Parameters:', lda.get_params())
    # # Plot 2 class data with top 3 components
    # class1 = X_train[:,:3][y_train == 1]
    # class2 = X_train[:,:3][y_train == 0]
    # from mpl_toolkits.mplot3d import Axes3D
    # from mpl_toolkits.mplot3d import proj3d
    # fig = plt.figure(figsize=(8,8))
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(class1[0,:], class1[1,:], class1[2,:], 'o', markersize=8, color='blue', alpha=0.5, label='class1')
    # ax.plot(class2[0,:], class2[1,:], class2[2,:], '^', markersize=8, color='red', alpha=0.5, label='class2')
    # plt.title('Class distribution with top 3 components')
    # ax.legend(loc='upper right')
    # plt.show()


if __name__ == '__main__':

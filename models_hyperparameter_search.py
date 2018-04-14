"""
============================================================
Parameter estimation using grid search with cross-validation
============================================================
A classifier is optimized by cross-validation, which is done
using the GridSearchCV objecton a development set that
comprises only half of the available labeled data.

The performance of the selected hyper-parameters and trained
model is then measured on a dedicated evaluation set that
was not used during the model selection step.
"""

from __future__ import print_function
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, auc, roc_curve
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, SGDClassifier
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, AdaBoostRegressor, AdaBoostClassifier, BaggingRegressor, BaggingClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier

mpl.rcParams.update({
    'font.size'           : 12.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'small',})

def load_data():
    df = pd.read_csv('data/AnalyticsChallenge1-Train.csv')
    del df['StandardHours'] # every value is 80
    del df['Over18'] # every value is 'Y'
    del df['EmployeeCount']
    travel = ['Non-Travel', 'Travel_Rarely', 'Travel_Frequently']
    df['BusinessTravel'] = df.BusinessTravel.astype("category", ordered=True, categories=travel).cat.codes
    df = pd.get_dummies(df, columns=['Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime'],\
            drop_first=True, prefix=['Dept', 'Edu', 'Gender', 'Role', 'Status', 'OT'])
    del df ['EmployeeNumber']
    df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    y = df.pop('Attrition').values
    X = df.values
    cols = df.columns.values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.5, random_state=0, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, cols

def get_models():
    models = [LinearRegression(), SGDRegressor(), SVR(), KNeighborsRegressor(), DecisionTreeRegressor(), RandomForestRegressor(), GradientBoostingRegressor(), AdaBoostRegressor(), BaggingRegressor(), MLPRegressor()]
    reg = [{}]
    sgd = [{}]
    svr = [{}]
    knn = [{}]
    dt = [{}]
    rf = [{}]
    gb = [{}]
    ab = [{}]
    bag = [{}]
    mlp = [{}]
    params = [reg, sgd, svr, knn, dt, rf, gb, ab, bag, mlp]
    return models, params

def get_class_models():
    models = [LogisticRegression(), SGDClassifier(), SVC(), KNeighborsClassifier(), DecisionTreeClassifier(), RandomForestClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), BaggingClassifier(), MLPClassifier()]
    reg = [{'penalty': ['l1', 'l2'], 'C': [1, 10, 100, 1000], 'fit_intercept': [True, False], 'class_weight': [None, 'balanced']}]
    sgd = [{}]
    svc = [{'kernel': ['linear'], 'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None], 'probability': [True]},
                        {'kernel': ['poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None], 'probability': [True]},
                        {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None], 'probability': [True]},
                        {'kernel': ['sigmoid'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000], 'class_weight': ['balanced', None], 'probability': [True]}]
    knn = [{}]
    dt = [{}]
    rf = [{}]
    gb = [{'learning_rate': np.logspace(-2, 0, num=3), 'max_depth': [1, 3, 10], 'min_samples_leaf': [1, 3, 10], 'subsample': [1.0, 0.5], 'max_features': [None, 'sqrt'], 'n_estimators': [100]]
    ab = [{}]
    bag = [{}]
    mlp = [{'hidden_layer_sizes': [(10, 10), (50,50), (100,100), (200,200)], 'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'], 'alpha': [.0001, .0005, .001, .005, .01, .05, .1], 'batch_size': [50, 100, 200, 500, 1000], 'learning_rate': ['constant', 'invscaling', 'adaptive']}]
    params = [reg, sgd, svc, knn, dt, rf, gb, ab, bag, mlp]
    return models, params

def find_important_features(name, score, clf, cols):
    if 'Regression' in name or 'SVC' in name:
        feature_importances = np.abs(clf.best_estimator_.coef_[0])
    else:
        feature_importances = clf.best_estimator_.feature_importances_
    n=5
    top_colidx = np.argsort(feature_importances)[::-1][0:n]
    feature_importances = feature_importances[top_colidx]
    feature_importances = feature_importances / float(feature_importances.max()) #normalize
    feature_names = [cols[idx] for idx in top_colidx]
    print("Top {} features and relative importances:".format(str(n)))
    for fn, fi in zip(feature_names, feature_importances):
        print("     {0:<30s} | {1:6.3f}".format(fn, fi))

    y_ind = np.arange(n-1, -1, -1)
    fig = plt.figure(figsize=(8, 8))
    plt.barh(y_ind, feature_importances, height = 0.3, align='center')
    plt.yticks(y_ind, feature_names)
    plt.xlabel('Relative feature importances')
    plt.ylabel('Features')
    plt.title("{} ({} score)".format(name, score))
    plt.tight_layout()
    plt.savefig('plots/feature_importance_'+name+'_'+score, dpi = 300)
    plt.close()

def classifier_report(name, score, clf, X_test, y_test):
    print("Detailed classification report:")
    print("(The model is trained on the full development set.")
    print("The scores are computed on the full evaluation set.)")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred))

    # confusion matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred) = [[TN FP], [FN TP]]
    # cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalized

    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test,y_pred)
    y_prob = clf.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, 1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=1, label='Model: {}\nScore: {}\nAUC: {:.2f}\nAccuracy ={:.2f}\nRecall ={:.2f}\nPrecision ={:.2f}'.format(name, score, roc_auc, accuracy,recall, precision))
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6))
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic curve')
    plt.legend(loc="best")

def param_search(c, models, params, X_train, X_test, y_train, y_test, cols):
    scores = ['precision', 'recall']
    best_models = []
    for model,param in zip(models,params):
        name = model.__class__.__name__
        for score in scores:
            print("\n########## {} model ##########".format(name))
            print("Tuning hyper-parameters for %s" % score, '...')

            clf = GridSearchCV(model, param, cv=5, scoring='%s_macro' % score, n_jobs=-1)
            clf.fit(X_train, y_train)
            print("Best parameters set found on development set:")
            print(clf.best_params_)
            print('Best {} score: {:.4f}'.format(score, clf.best_score_))
            # print("Grid scores on development set:")
            # means = clf.cv_results_['mean_test_score']
            # stds = clf.cv_results_['std_test_score']
            # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            #     print("%0.3f (+/-%0.03f) for %r"
            #           % (mean, std * 2, params))
            best_models.append(clf.best_estimator_)

            find_important_features(name, score, clf, cols)

            if c:
                classifier_report(name, score, clf, X_test, y_test)
            else:
                plt.close()
            return best_models


if __name__ == '__main__':
    X_train, X_test, y_train, y_test, cols = load_data()

    # c = input("Classifier? [T]rue/[F]alse: ")
    c = 'T'
    if c == 'F':
        c = False
        models, params = get_models()
    else:
        c = True
        models, params = get_class_models()

    plt.figure(figsize=(6,6))
    # add pipeline with StandardScaler
    best_models = param_search(c, models, params, X_train, X_test, y_train, y_test, cols)
    # function to write result output to csv...
    # df.to_csv(filepath, columns=, index=False)
    plt.savefig('plots/roc_curve')
    # plt.show()
    plt.close()

    # http://scikit-learn.org/stable/modules/model_evaluation.html
    # http://scikit-learn.org/stable/auto_examples/plot_compare_reduction.html

    # Hyperparameter search and Cross Validation
    # grid search, random search, smart hyperparameter tuning, Bayesian
    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # http://scikit-learn.org/stable/modules/grid_search.html
    # http://www.ritchieng.com/machine-learning-efficiently-search-tuning-param/
    # http://betatim.github.io/posts/bayesian-hyperparameter-search/
    # https://thuijskens.github.io/2016/12/29/bayesian-optimisation/


    # Support Vector Machines: http://scikit-learn.org/stable/modules/svm.html#svm
    # from sklearn.svm import SVR, SVC

    # Unsupervised Learning/Clustering: KMeans
    # Neural Networks with keras
    # from keras.models import Sequential
    # from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
    # from sklearn.cluster import KMeans

"""
======================
Classification Toolbox
======================
"""

import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier

import warnings
warnings.filterwarnings("ignore")

mpl.rcParams.update({
    'font.size'           : 12.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'small',})

def vif(df, cols, X):
    # vif = []
    # linear = LinearRegression()
    # for col in range(X.shape[1]):
    #     linear.fit(np.delete(X, col, axis=1), X[:,col])
    #     r2 = linear.score(np.delete(X, col, axis=1), X[:,col])
    #     vif.append(1./(1-r2))
    # return np.array(vif)
    vif = {col : variance_inflation_factor(X, i) for i,col in enumerate(cols)}
    print('\nFeatures with VIF > 5 indicate collinearity with at least one other feature.')
    return {k:round(v) for k,v in vif.items() if v >= 5}

def ols_sm(X_train, y_train):
    X_train = sm.add_constant(X_train) # adds col of ones for intercept coefficient in OLS model
    ols = sm.Logit(y_train,X_train).fit()
    with open('ols_model_summary.txt', 'w') as f:
        f.write(ols.summary().as_text())

def simple_class(X_train, y_train, X_test, y_test, models):
    for name, model in models.items():
        print('\n',name)
        y_pred = model.predict(X_test)
        # print('\nLinear Regression Summary')
        # print('R2:', model.score(X_test, y_test))
        # print('Intercept:', model.intercept_, '\nCoefficients:', model.coef_)
        # print('Regression Model Parameters:', model.get_params())

        '''Predict how well model will perfom on test data'''
        # http://scikit-learn.org/stable/modules/model_evaluation.html
        # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
        # scoring='wrong' to see valid scoring options
        # Common regression metrics: R2, RMSE, quantiles/MAPE, precision accuracy
        scoring='recall'
        # ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision', 'completeness_score', 'explained_variance', 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score', 'homogeneity_score', 'mutual_info_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'normalized_mutual_info_score', 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted', 'r2', 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted', 'roc_auc', 'v_measure_score']
        score = cross_val_score(estimator=model, X=X_train, y=y_train, fit_params=None, scoring=scoring, cv=5, n_jobs=-1)
        print(score)
        print('Mean Cross Validation {} Score:'.format(scoring), score.mean())


        cm = confusion_matrix(y_test, y_pred)
        print('Confusion Matrix:\n[[TN, FP]\n[FN, TP]]\n', cm)
        report = classification_report(y_test, y_pred)
        # print('Classification Report:\n', report)
        acc = accuracy_score(y_test, y_pred)
        pre = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        print('Scores for positive Fraud class')
        print('Accuracy:', acc, 'Precision:', pre, 'Recall:', rec)
        probabilities = model.predict_proba(X_test)[:, 1]
        tpr, fpr, thresholds = roc_curve(probabilities, y_test)
        plt.plot(fpr, tpr, label=name)
        plt.xlabel("False Positive Rate (1 - Specificity)")
        plt.ylabel("True Positive Rate (Sensitivity, Recall)")
        plt.title("ROC plot")
    plt.legend(loc='best')
    plt.savefig('plots/roc_curve')
    plt.close()

def roc_curve(probabilities, labels):
    thresholds = np.sort(probabilities)
    tprs = []
    fprs = []
    num_positive_cases = sum(labels)
    num_negative_cases = len(labels) - num_positive_cases
    for threshold in thresholds:
        # With this threshold, give the prediction of each instance
        predicted_positive = probabilities >= threshold
        # Calculate the number of correctly predicted positive cases
        true_positives = np.sum(predicted_positive * labels)
        # Calculate the number of incorrectly predicted positive cases
        false_positives = np.sum(predicted_positive) - true_positives
        # Calculate the True Positive Rate
        tpr = true_positives / float(num_positive_cases)
        # Calculate the False Positive Rate
        fpr = false_positives / float(num_negative_cases)
        fprs.append(fpr)
        tprs.append(tpr)
    return tprs, fprs, thresholds.tolist()


def get_model_profits(model, cost_benefit, X_train, X_test, y_train, y_test):
    """Fits passed model on training data and calculates profit from cost-benefit
    matrix at each probability threshold."""
    model.fit(X_train, y_train)
    predicted_probs = model.predict_proba(X_test)[:, 1]
    profits, thresholds = profit_curve(cost_benefit, predicted_probs, y_test)
    return profits, thresholds

def profit_curve(cost_benefit, predicted_probs, labels):
    """Function to calculate list of profits based on supplied cost-benefit
    matrix and prediced probabilities of data points and thier true labels."""
    n_obs = float(len(labels))
    # Make sure that 1 is going to be one of our thresholds
    maybe_one = [] if 1 in predicted_probs else [1]
    thresholds = maybe_one + sorted(predicted_probs, reverse=True)
    profits = []
    for threshold in thresholds:
        y_predict = predicted_probs >= threshold
        confusion_matrix = standard_confusion_matrix(labels, y_predict)
        threshold_profit = np.sum(confusion_matrix * cost_benefit) / n_obs
        profits.append(threshold_profit)
    return np.array(profits), np.array(thresholds)

def standard_confusion_matrix(y_true, y_pred):
    """Make confusion matrix with format:
                  -----------
                  | TP | FP |
                  -----------
                  | FN | TN |
                  -----------"""
    [[tn, fp], [fn, tp]] = confusion_matrix(y_true, y_pred)
    return np.array([[tp, fp], [fn, tn]])

def plot_model_profits(model_profits):
    """Plotting function to compare profit curves of different models."""
    for model, profits, _, name in model_profits:
        percentages = np.linspace(0, 100, profits.shape[0])
        plt.plot(percentages, profits, label=name)

    plt.title("Profit Curves")
    plt.xlabel("Percentage of test instances (decreasing by score)")
    plt.ylabel("Profit")
    plt.legend(loc='best')
    plt.savefig('plots/profit_curve')

def find_best_threshold(model_profits):
    """Find model-threshold combo that yields highest profit."""
    max_model = None
    max_threshold = None
    max_profit = None
    for model, profits, thresholds, name in model_profits:
        max_index = np.argmax(profits)
        if not max_model or profits[max_index] > max_profit:
            max_model = model
            max_threshold = thresholds[max_index]
            max_profit = profits[max_index]
            max_name = name
    return max_model, max_threshold, max_profit, max_name

def profit_curve_main(cost_benefit, X_train, X_test, y_train, y_test, models):
    """Main function to test profit curve code.

    Parameters
    ----------
    filepath     : str - path to find churn.csv
    cost_benefit  : ndarray - 2D, with profit values corresponding to:
                                          -----------
                                          | TP | FP |
                                          -----------
                                          | FN | TN |
                                          -----------
    """
    model_profits = []
    for name, model in models.items():
        profits, thresholds = get_model_profits(model, cost_benefit,
                                                X_train, X_test,
                                                y_train, y_test)
        model_profits.append((model, profits, thresholds, name))
    plot_model_profits(model_profits)
    max_model, max_thresh, max_profit, name = find_best_threshold(model_profits)
    max_labeled_positives = max_model.predict_proba(X_test) >= max_thresh
    proportion_positives = max_labeled_positives.mean()
    reporting_string = ('Best model:\t\t{}\n'
                        'Best threshold:\t\t{:.2f}\n'
                        'Resulting profit:\t{}\n'
                        'Proportion positives:\t{:.2f}')
    print((reporting_string.format(name, max_thresh, max_profit, proportion_positives)))

if __name__ == '__main__':
    filename = '../data/poc.csv'
    data = pd.read_csv(filename, index_col=0)
    y = data.pop('fraud').values
    X = data.values
    cols = data.columns
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=.33,random_state=42,stratify=y)

    # print(vif(data, cols, X_train)) # Identify multicollinearity
    # ols_sm(X, y)

    models = [RF(), LR(), GBC(), SVC(probability=True)]
    trained_models = {}
    for model in models:
        pipe = Pipeline([('sm', SMOTE(random_state=42)),
                         ('ss', StandardScaler()),
                        #  ('pca', PCA(n_components=5)),
                         ('model', model)])
        pipe.fit(X_train, y_train)
        trained_models[model.__class__.__name__] = pipe

    simple_class(X_train, y_train, X_test, y_test, trained_models)

    cost_benefit = np.array([[-1, -1], [-5, 0]])
    profit_curve_main(cost_benefit, X_train, X_test, y_train, y_test, trained_models)















    # Classification (white box): LogisticRegression, KNeighborsClassifier, DecisionTreeClassifier, SGDClassifier
    # Classification (black box): RandomForestClassifier, BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier, MLPClassifier
    # VotingClassifier(estimators=[])
    # Classification metrics: accuracy, confusion matrix, AUC, precision-recall
        # log-loss = -1/N * SUM i=1 to N... [ yi*log(pi) + (1-yi)*log(1-pi) ]
            # requires both class label and probability
        # F1 score = 2 * precision*recall / precision+recall

    # '''scatter plot with decision boundary line'''
    # x_ = np.linspace(start,stop,num_pts).reshape(-1,1)
    # sigmoid = model_sk.predict_proba(x)[:,1] # "h" hypothesis function
    # plt.plot(x_,sigmoid)

    # NLP with naive_bayes, document similarity?
    # classifier only?
    # gnb = GaussianNB()
    # gnb.fit(X_train, y_train)
    # gnb.predict(X_test)

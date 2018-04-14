"""
==================
Regression Toolbox
==================
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from data_preprocess import preprocess_data, split, standardize

mpl.rcParams.update({
    'font.size'           : 12.0,
    'axes.titlesize'      : 'large',
    'axes.labelsize'      : 'medium',
    'xtick.labelsize'     : 'small',
    'ytick.labelsize'     : 'small',
    'legend.fontsize'     : 'small',})

def vif(X, cols):
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

def ols_sm(X_train, y_train, X_test):
    X_train = sm.add_constant(X_train) # adds col of ones for intercept coefficient in OLS model
    ols = OLS(y_train,X_train).fit()
    # with open('ols_model_summary.csv', 'w') as f:
    #     f.write(ols.summary().as_csv())
    with open('ols_model_summary.txt', 'w') as f:
        f.write(ols.summary().as_text())

    # Plot True vs Predicted values to examine if linear model is a good fit
    fig = plt.figure(figsize=(12,8))
    X_test = sm.add_constant(X_test)
    plt.scatter(y_test, ols.predict(X_test))
    plt.xlabel('True values')
    plt.ylabel('Predicted values')
    plt.title('True vs Predicted values')
    plt.show()
    plt.close()
    # Add quadratic term to X or take log of y to improve

    # Discern if a linear relationship exists with partial regression plots
    fig = plt.figure(figsize=(12,8))
    fig = sm.graphics.plot_partregress_grid(ols, fig=fig)
    plt.title('Partial Regression Plots')
    plt.show()
    plt.close()

    # Identify outliers and high leverage points
    # a. Identify outliers (typically, those data points with studentized residuals outside of +/- 3 stdev).
    # Temporarily remove these from your data set and re-run your model.
    # Do your model metrics improve considerably? Does this give you cause for more confidence in your model?
    # b. Identify those outliers that are also high-leverage points (high residual and high leverage --> high influence).
    fig, ax = plt.subplots(figsize=(12,8))
    fig = sm.graphics.influence_plot(ols, ax=ax, criterion="cooks")
    plt.show()
    fig, ax = plt.subplots(figsize=(8,6))
    fig = sm.graphics.plot_leverage_resid2(ols, ax=ax)
    plt.show()
    plt.close()

    # Confirm homoscedasticity (i.e., constant variance of residual terms)
    # If residuals exhibit a “funnel shaped” effect, consider transforming your data into logarithmic space.
    studentized_residuals = ols.outlier_test()[:, 0]
    y_pred = ols.fittedvalues
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(y_pred, studentized_residuals)
    ax.axhline(y=0.0, color='k', ls='--')
    ax.set_xlabel('Predicted y')
    ax.set_ylabel('Studentized Residuals')
    plt.show()
    plt.close()

    # Test if residuals are normally distributed in QQ plot
    # plots quantile of the normal distribution against studentized residuals
    # if sample quantiles are normally distributed, the dots will align with 45 deg line
    fig, ax = plt.subplots()
    sm.graphics.qqplot(studentized_residuals, fit=True, line='45', ax=ax)
    plt.show()
    plt.close()

    # Find influencial points in data
    # DFBETAS - standardized measure of how much each coefficient changes when that observation is left out
    threshold = 2./len(X_train)**.5
    infl = ols.get_influence()
    df = pd.DataFrame(infl.summary_frame().filter(regex="dfb"))
    inf = df[df > threshold].dropna(axis=0, how='all')
    print('Influencial points:\n', inf)


def simple_linear(X_train, y_train, X_test, y_test):
    linear = LinearRegression()
    linear.fit(X_train,y_train)
    y_pred = linear.predict(X_test)
    print('\nLinear Regression Summary')
    print('R2:', linear.score(X_test, y_test))
    print('Intercept:', linear.intercept_, '\nCoefficients:', linear.coef_)
    print('Parameters:', linear.get_params())

    '''Predict how well model will perfom on test data'''
    # http://scikit-learn.org/stable/modules/model_evaluation.html
    # http://scikit-learn.org/stable/modules/classes.html#module-sklearn.metrics
    # scoring='wrong' to see valid scoring options
    # Common regression metrics: R2, RMSE, quantiles/MAPE, precision accuracy
    score = cross_val_score(estimator=linear, X=X_train, y=y_train, fit_params=None, scoring='r2', cv=5, n_jobs=-1)
    print('Mean Cross Validation Score:', score.mean())


if __name__ == '__main__':
    filepath = 'data/continous_target_data.csv'
    df = pd.read_csv(filepath)
    df = preprocess_data(df)
    X_train, X_test, y_train, y_test, cols = split(df, target='?', smote=False)
    X_train, X_test = standardize(X_train, X_test)

    # print(vif(X_train, cols))
    # ols_sm(X_train, y_train, X_test)
    simple_linear(X_train, y_train, X_test, y_test)

    '''Simplify with subset of features and re-evaluate model'''
    '''Improve your model by adding complexity incrementally'''
    # a. Consider adding (synthesized) interaction terms to model synergistic effects between multiple regressors.
    # b. Consider adding (synthesized) higher-order terms to model non-linear effects.

    '''Feature importance'''

    # Regression (white box): LinearRegression, KNeighborsRegressor, DecisionTreeRegressor, SGDRegressor
    # Regression (black box): RandomForestRegressor, BaggingRegressor, GradientBoostingRegressor, AdaBoostRegressor, MLPRegressor

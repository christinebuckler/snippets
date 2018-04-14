"""
========================
Load and format raw data
========================
"""
import pickle,os
import pandas as pd
import numpy as np
from sklearn.datasets import make_regression, make_classification
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import warnings


warnings.filterwarnings("ignore")

def preprocess_data(df):
    df = df.sample(frac=1, random_state=123)

    df["Int'l Plan"] = (df["Int\'l Plan"] == 'yes').astype(int)
    df["VMail Plan"] = (df["VMail Plan"] == 'yes').astype(int)
    df['Churn?'] = (df['Churn?'] == 'True.').astype(int)
    del df['State']
    del df['Area Code']
    del df['Phone']

    # df[col] = (df[col] == 'string').astype(int)
    # df[col].get_dummies(df[col], columns=cols, drop_first=True)

    # size_dict = {'XL': 3, 'L': 2, 'M': 1} # ordinal features
    # inv_size_dict = {v: k for k, v in size_dict.items()}
    # class_dict = {label:idx for idx,label in enumerate(set(df['class label']))}
    # df['class label'] = df['class label'].map(class_dict)

    # del df[col]

    # http://scikit-learn.org/stable/modules/preprocessing.html

    # df_dict = df.transpose().to_dict().values()
    # dv = DictVectorizer()
    # V = dv.fit_transform(df_dict)
    # df = pd.DataFrame(V, columns=dv.get_feature_names())
    # vocab_dict = dv.vocabulary_

    # df[col] = LabelEncoder().fit_transform(df[col])
    # le = LabelEncoder()
    # df = df.apply(le.fit_transform)
    # le.classes_
    # orig_form = le.inverse_transform([df[col]])

    # df[col] = OneHotEncoder(categorical_features=array([col_idx]), sparse=False).fit_transform(df[col])
    # enc = OneHotEncoder(sparse=False)
    # df = df.apply(enc.fit_transform)

    return df

def split(df, target, smote=False):
    y = df.pop(target).values
    X = df.values
    cols = df.columns.values

    if smote==True:
        print('Original dataset class breakout {}'.format(Counter(y)))
        sm = SMOTE(ratio='auto', random_state=123, k_neighbors=5, m_neighbors=10, kind='regular', n_jobs=-1)
        # available kinds: 'regular', 'borderline1', 'borderline2', 'svm'
        # extra params for svm: out_step=.5, svm_estimator=sklearn.svm.SVC
        X, y = sm.fit_sample(X, y)
        print('Resampled dataset class breakout {}'.format(Counter(y)))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
    return X_train, X_test, y_train, y_test, cols

def standardize(X_train, X_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test

def generate_data(dtype='regression'):
    if dtype=='regression':
        X, y = make_regression(n_samples=100, n_features=100, n_informative=10, n_targets=1, bias=0.0, effective_rank=None, tail_strength=0.5, noise=0.0, shuffle=True, coef=False, random_state=123)
    elif dtype=='classification':
        X, y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2, n_clusters_per_class=2, weights=None, flip_y=0.01, class_sep=1.0, hypercube=True, shift=0.0, scale=1.0, shuffle=True, random_state=123)
    elif dtype=='clusters':
        X, y = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=1.0, center_box=(-10.0, 10.0), shuffle=True, random_state=123)
    else:
        print('Choose an available data type.')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123, stratify=y)
    return X_train, X_test, y_train, y_test


if __name__ == '__main__':

    # File with continous target data...

    filepath = 'data.csv'
    df = pd.read_csv(filepath)
    df = preprocess_data(df)

    # pickled = 'df.pkl'
    # if os.path.exists(pickled):
    #     print('Loading pickle...')
    #     df = pd.read_pickle(pickled)
    # else:
    #     print('Saving pickle...')
    #     df = pd.read_csv(filepath)
    #     df = preprocess_data(df)
    #     df.to_pickle(pickled)
    # os.system("rm %s"%pickled)

    X_train, X_test, y_train, y_test, cols = split(df, target, smote=False)
    X_train, X_test = standardize(X_train, X_test)

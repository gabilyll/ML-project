import numpy as np
import pandas as pd

def entropy(Y):
    unique, count = np.unique(Y, return_counts=True, axis=0)
    prob = count / len(Y)
    en = np.sum((-1) * prob * np.log2(prob))
    return en


# Joint Entropy
def jEntropy(Y, X):
    YX = np.c_[Y, X]
    return entropy(YX)


# conditional entropy
def cEntropy(Y, X):
    return jEntropy(Y, X) - entropy(X)


# gain
def gain(Y, X):
    return entropy(Y) - cEntropy(Y, X)


# SU
def symmetricalUncertain(Y, X):
    n = float(Y.shape[0])
    vals = np.unique(Y)
    # Computing Entropy for the feature x.
    Hx = entropy(X)
    # Computing Entropy for the feature y.
    Hy = entropy(Y)
    # Computing Joint entropy between x and y.
    Hxy = jEntropy(Y, X)
    IG = Hx - Hxy
    return 2 * IG / (Hx + Hy)


def fit(X, y, k):
    top_n1 = []
    for i in X.columns.values:
        X[i] = X[i].astype('float64')
        y = y.astype('float64')
        top_n1.append(symmetricalUncertain(X[i], y))

    col_name1 = np.array(X.columns)
    a1 = pd.DataFrame(top_n1)
    b1 = pd.DataFrame(col_name1)
    info3 = pd.concat([a1, b1], axis=1)
    info3.columns = ['Score', 'Features']
    top3 = info3.nlargest(k, 'Score')
    SelectedFeatures3 = np.array(top3['Features'])
    return SelectedFeatures3.tolist()

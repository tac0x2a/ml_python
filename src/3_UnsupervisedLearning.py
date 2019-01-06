#%%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import mglearn as mg

#%%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target)

min = X_train.min(axis=0)
max = X_train.max(axis=0)
print(min)
print(max)


#%% データ変換(スケーリング)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
# X_train_new = scaler.fit(X_train).transform(X_train)
X_train_new = scaler.fit_transform(X_train)  # まとめて呼べる

min = X_train_new.min(axis=0)
max = X_train_new.max(axis=0)
print(min)
print(max)

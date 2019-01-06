#%%
import numpy as np
import matplotlib.pyplot as plt
import mglearn as mg

#%%
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

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

#%% 主成分分析 ----------------------------------------------------------
# cancerデータの特徴量ごとのヒストグラムを表示してみる(写経)
# 特徴量ごとのヒストグラムはいろんなところで使えそう・・・

feature_cnt = cancer.data.shape[1]
display_columns = 2
fig, axes = plt.subplots(int(feature_cnt / display_columns),
                         display_columns, figsize=(10, 20))

neg = cancer.data[cancer.target == 0]
pos = cancer.data[cancer.target == 1]

ax = axes.ravel()

for i in range(feature_cnt):
    feature = cancer.data[:, i]
    _, bins = np.histogram(feature, bins=50)

    pos_f = pos[:, i]
    neg_f = neg[:, i]
    ax[i].hist(pos_f, bins=bins, alpha=.8)
    ax[i].hist(neg_f, bins=bins, alpha=.5)

    feature_name = cancer.feature_names[i]
    ax[i].set_title(feature_name)

    ax[i].set_yticks(())

ax[0].legend(["neg", "pos"])
fig.tight_layout()

#%% 前処理としてStandardScalerを適用してみる
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = cancer.data
X_scaled = scaler.fit_transform(X)

y = cancer.target


#%% 主成分分析で主成分2つだけ残す
print("X_scaled.shape", X_scaled.shape)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)
print("X_pca.shape", X_pca.shape)

plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y)
# plt.savefig('doc/img/pca_cancer.png')

#%% 主成分ごとに，元の特徴量の係数を確認する
print(pca.components_)
print(pca.components_.shape)  # (2, 30), 30は元でーたの特徴量数

#%% ヒートマップでも確認してみる
plt.matshow(abs(pca.components_))
plt.colorbar()
plt.xticks(range(feature_cnt), cancer.feature_names, rotation=70)
plt.yticks(range(2), ["First", "Second"])

# plt.savefig("doc/img/pca_heatmap.png")


#%%


#%%

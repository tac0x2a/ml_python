
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


#%% 固有顔による特徴量抽出 ----------------------------------------------------------
from sklearn.datasets import fetch_lfw_people
people = fetch_lfw_people(min_faces_per_person=20, resize=.7)
image_shape = people.images[0].shape  # (87, 65)

print(image_shape)

fig, axes = plt.subplots(2, 5, figsize=(15, 8))
for name, image, ax in zip(people.target_names, people.images, axes.ravel()):
    ax.imshow(image)
    ax.set_title(name)

#%% データカウント
for name, cnt in zip(people.target_names, np.bincount(people.target)):
    print("{}:{}".format(name, cnt))

#%% 同じ人ばっかり出てこないように50枚づつピックアップする
mask = np.zeros(people.target.shape, dtype=np.bool)

for id in np.unique(people.target):
    target_indexes = np.where(people.target == id)
    mask[np.where(people.target == id)[0][:50]] = 1

X_people = people.data[mask]
y_people = people.target[mask]

X_people /= 255.0  # 0から1の実数に変換．数値的に安定(?)するらしい

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_people, y_people)

#%% 元データで訓練・評価
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train, y_train)
knn.score(X_test, y_test)  # 0.238


#%% PCAの主成分で訓練・評価
pca = PCA(n_components=100, whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1).fit(X_train_pca, y_train)
knn.score(X_test_pca, y_test)  # 0.329

#%% 主成分を可視化してみる
image_shape = people.images[0].shape

fig, axies = plt.subplots(2, 5, figsize=(15, 8))
for idx, (data, ax) in enumerate(zip(pca.components_, axies.ravel())):
    ax.imshow(data.reshape(image_shape))
    ax.set_title(idx)

fig.savefig("doc/img/face_pca.png")

#%%

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import mglearn as mg

# -----------------------------------------------------------
# クラス分類のサンプルデータ
X, y = mg.datasets.make_forge()
X.shape
y.shape
X
# 散布図をプロット
mg.discrete_scatter(X[:, 0], X[:, 1], y ) #
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")

# mg.discrete_scatterを使わずにやろうと思うと・・・
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')

# -----------------------------------------------------------
# 回帰のサンプルデータ(waveデータセット)
X, y = mg.datasets.make_wave(n_samples=128)
X.shape
y.shape

plt.plot(X, y, 'o')
plt.xlabel("Feature")
plt.ylabel("Target")


# -----------------------------------------------------------
# 特徴量の多いデータのサンプルとして 乳がんデータセットを見てみる
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
# print(cancer.DESCR)

# データの確認
cancer.data.shape #(569,30)
cancer.target.shape # (569,)
# np.bincount(cancer.target) # 値ごとに集計したndarrayを返す
{n: c for n,c in zip(cancer.target_names, np.bincount(cancer.target))} #良性/悪性のカウント

# さらに特徴量を交互作用(interaction)で拡張してみる
#


# -----------------------------------------------------------
# forgeのデータセットに対して k-最近傍法でクラス分類してみる
X, y = mg.datasets.make_forge()

mg.plots.plot_knn_classification(n_neighbors=3) # 3近傍の例

# 実際にscikit-learnでやってみる．
X, y = mg.datasets.make_forge()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# X_train.shape
# y_train.shape
# X_test.shape
# y_test.shape

from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)

clf.predict([[10, 1]]) # 0
clf.predict([[10, 5]]) # 1

clf.score(X_test, y_test)


# -----------------------------------------------------------
# cancerのデータセットに対して k-最近傍法でクラス分類してみる
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# X_train.shape
# X_test.shape

# n=1から10を試して，精度を比較する
r = range(1,100)
training_accuracy = []
test_accuracy     = []

for n in r:
    clf = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.ylabel("Accuracy") # 精度
plt.xlabel("n")
plt.plot(r, training_accuracy, label="training_accuracy")
plt.plot(r, test_accuracy, label="test_accuracy")
plt.legend() # plotで設定したlabelを表示する

# nが小さいほど，出来上がるモデルは複雑になる．
# nがサンプル数と等しい場合が最も単純であると考える．

# -----------------------------------------------------------
# waveデータでn-近傍回帰してみる
mg.plots.plot_knn_regression(n_neighbors=3)

# 実際にやってみる
X, y = mg.datasets.make_wave(n_samples=1024)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

plt.plot(X_train, y_train, '^', label="train", markersize=8)
plt.plot(X_test, y_test, 'v', label="test", markersize=8)
plt.xlabel("Feature")
plt.ylabel("Target")

from sklearn.neighbors.regression import KNeighborsRegressor
reg = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
reg.score(X_test, y_test) # 回帰問題なので R^2スコアを返す

# n_samples を増やすとデータの傾向が見えてくる．
# どうやら，sinカーブを線形にバイアスしたものにランダム加算したものっぽい．


# knnのnを変えながら評価してみる

# plt.subplots(行数, 列数, figsize=(5,10))
fig, axis = plt.subplots(5, 1, figsize=(8,16))

# np.reshape(newshape)
# Gives a new shape to an array without changing its data.
# reshape(n,m) => n行 m列にする．-1を指定すると要素数を元によしなにやってくれる
line = np.linspace(-3, 3, 1000).reshape(-1, 1) # 1000行1列
for n, ax in zip([1,3,9,15,30], axis): # axは描画先
    reg = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train)
    ax.plot(line, reg.predict(line)) # -3 から 3 までのデータを評価
    ax.plot(X_train, y_train, '^', c=mg.cm2(0), markersize=2)
    ax.plot(X_test,  y_test,  'v', c=mg.cm2(1), markersize=2)

    ax.set_title(
        "n={}, train_score={:.2f}, test_score={:.2f}".format(n, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

axis[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")

# nを増やすと予測(Model predictions)はスムーズになり，
# 訓練データに対する精度が下がる．
# 一方で# テストデータに対するスコアも下がっている．(n_samples=40の場合)
# 線形バイアスがかかってるデータなので，直感的にはスコアは上がりそうだが・・・なんで？
# どうやらサンプル数が少ないことが原因っぽい．
# n_samples=1024にすると，nが大きいほどtest_scoreも大きくなった．

# nを30まで増やしてみたが，n=9でtest_score=0.72でサチってるっぽい．
# グラフを見ると，nが増えるほどモデルはなめらかな正弦波に近づくが，
# ランダム変動を吸収できないので，これくらいの精度に落ち着いてしまうのでは．

# --------------------------

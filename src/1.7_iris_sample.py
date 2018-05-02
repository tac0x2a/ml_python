import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# サンプルのデータセットを読み込む
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print(iris_dataset.keys())

print(iris_dataset['DESCR']) # データセットの説明

# ラベル名(種類)
print(iris_dataset['target_names'])

# ラベル(種類を0から2の整数としてエンコードしたもの)
print(iris_dataset['target'])

# 特徴量名(dataに対応するにインデックス)
print(iris_dataset['feature_names'])

iris_dataset['data'].shape

# 最初の5サンプル
iris_dataset['data'][:5]

# ラベル名に置換してみる(普通のlist)
[ iris_dataset['target_names'][i] for i in iris_dataset['target']]

# ndarrayならこんな書き方ができる(numpy.ndarray)
iris_dataset['target_names'][iris_dataset['target']]

# ---------------------------------------------------------------------
# 新たに計測したIrisの種類を予測する機械学習モデルを構築する

from sklearn.model_selection import train_test_split

# 訓練セットとテストセットに分割する
# X: 入力(特徴量)
# y: 出力(ラベル)
# random_stateは乱数のシード．明示的に指定してあげることで結果が決定的になる．
X_train, X_test, y_train, y_test = train_test_split(iris_dataset['data'], iris_dataset['target'], random_state=0)

 # (112,4): 75%
X_train.shape

# (38,4): 25%
X_test.shape

y_train.shape
y_test.shape


# プロットするためにDataFrameに起こす．
iris_df = pd.DataFrame(X_train, columns=iris_dataset.feature_names)
iris_df

# ペアプロットしてデータを観察する
# c:class,color?
# s: 点の大きさ
grr = pd.scatter_matrix(iris_df, c=y_train, figsize=(10,10), hist_kwds={'bins':20}, s=60, alpha=.8)

# どうやら分類できそうな気がする．

# ---------------------------------------------------------------------
# k-最近傍法(k-Nearest Neighbors)での分類を試みる
from sklearn.neighbors import KNeighborsClassifier
# n_neighbors:近傍点の数
knn = KNeighborsClassifier(n_neighbors=1)

# 訓練セットを使ってモデル構築する
knn.fit(X_train, y_train)

# テストデータを使って評価してみる
y_pred = knn.predict(X_test)

# True(正解)の割合を計算する
# np.mean([True, False, False, False]) #=> 0.25
np.mean(y_pred == y_test)

# 上記 knn.predictとnp.meanを一気にやっちゃう
knn.score(X_test, y_test)


# 適当なデータでモデルを使って分類してみる．
# 野生のアイリスを見つけたとしよう。ガクの長さが5cm、ガクの幅が2.9cm、花弁の長さが1cm、花弁の幅が0.2cmだったとする。
# このアイリスの品種は何だろうか？
X_new = np.array([[5.0, 2.9, 1.0, 0.2]])
prediction = knn.predict(X_new)
print(iris_dataset['target_names'][prediction])

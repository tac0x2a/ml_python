#%%
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import mglearn as mg

#%% -----------------------------------------------------------
# クラス分類のサンプルデータ
X, y = mg.datasets.make_forge()
X.shape
y.shape
X
#%% 散布図をプロット
mg.discrete_scatter(X[:, 0], X[:, 1], y)
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")

#%% mg.discrete_scatterを使わずにやろうと思うと・・・
plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
plt.legend(["Class 0", "Class 1"], loc=4)
plt.xlabel("First feature")
plt.ylabel("Second feature")


#%% -----------------------------------------------------------
# 回帰のサンプルデータ(waveデータセット)
X, y = mg.datasets.make_wave(n_samples=128)
X.shape
y.shape

plt.plot(X, y, 'o')
plt.xlabel("Feature")
plt.ylabel("Target")


#%% -----------------------------------------------------------
# 特徴量の多いデータのサンプルとして 乳がんデータセットを見てみる
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
cancer.keys()
print(cancer.DESCR)

#%% データの確認
cancer.data.shape  # (569,30)
cancer.target.shape  # (569,)
# np.bincount(cancer.target) # 値ごとに集計したndarrayを返す

cancer.feature_names

# 良性/悪性のカウント
{n: c for n, c in zip(cancer.target_names, np.bincount(cancer.target))}

# さらに特徴量を交互作用(interaction)で拡張してみる

#%% -----------------------------------------------------------
# forgeのデータセットに対して k-最近傍法でクラス分類してみる
X, y = mg.datasets.make_forge()

# 3近傍の例
mg.plots.plot_knn_classification(n_neighbors=3)

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

clf.predict([[10, 1]])  # 0
clf.predict([[10, 5]])  # 1

clf.score(X_test, y_test)

# これをプロットすると
X_p = np.array([[10, 1], [10, 5]])
y_p = clf.predict(X_p)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o')
plt.scatter(X_p[:, 0], X_p[:, 1], c=y_p, s=80, marker='*')

# 決定境界を表示してみる
X, y = mg.datasets.make_forge()
fig, axis = plt.subplots(1, 4, figsize=(10, 2))
neighbors = [1, 3, 9, len(X)]
for n, ax in zip(neighbors, axis):
    clf = KNeighborsClassifier(n_neighbors=n).fit(X, y)
    mg.plots.plot_2d_separator(clf, X, fill=True, eps=0.5, ax=ax, alpha=0.3)
    mg.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
    ax.set_title("{} neighbors".format(n))
    ax.set_xlabel("feature 0")
    ax.set_ylabel("feature 1")
axis[0].legend(loc=3)

# 近傍点が少ないほどモデルは複雑で，多いほど単純になる
# 近傍点を全てのテストデータポイント(len(X[:])=26)にすると，
# 全ての点について最も多いクラスのになる．

#%% -----------------------------------------------------------
# cancerのデータセットに対して k-最近傍法でクラス分類してみる
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
X_train.shape
X_test.shape

plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o')

import pandas as pd
cancer_df = pd.DataFrame(X_train, columns=cancer.feature_names)
pd.scatter_matrix(cancer_df, c=y_train, figsize=(16, 16),
                  hist_kwds={'bins': 40}, s=60, alpha=.8)

# n=1から100を試して，精度を比較する
r = range(1, 100)
training_accuracy = []
test_accuracy = []

for n in r:
    clf = KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train)
    training_accuracy.append(clf.score(X_train, y_train))
    test_accuracy.append(clf.score(X_test, y_test))

plt.ylabel("Accuracy")  # 精度
plt.xlabel("n")
plt.plot(r, training_accuracy, label="training_accuracy")
plt.plot(r, test_accuracy, label="test_accuracy")
plt.legend()  # plotで設定したlabelを表示する

# nが小さいほど，出来上がるモデルは複雑になる．
# nがサンプル数と等しい場合が最も単純であると考える．

#%% -----------------------------------------------------------
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
X, y = mg.datasets.make_wave(n_samples=1024)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
reg = KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train)
reg.score(X_test, y_test)  # 回帰問題なので R^2スコアを返す

# n_samples を増やすとデータの傾向が見えてくる．
# どうやら，sinカーブを線形にバイアスしたものにランダム加算したものっぽい．


# knnのnを変えながら評価してみる
X, y = mg.datasets.make_wave(n_samples=1024)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# plt.subplots(行数, 列数, figsize=(5,10))
fig, axis = plt.subplots(5, 1, figsize=(8, 16))

# ついでにスコアの推移も見る
training_accuracy = []
test_accuracy = []
# np.reshape(newshape)
# Gives a new shape to an array without changing its data.
# reshape(n,m) => n行 m列にする．-1を指定すると要素数を元によしなにやってくれる
line = np.linspace(-3, 3, 1000).reshape(-1, 1)  # 1000行1列
for n, ax in zip([1, 3, 9, 15, 30], axis):  # axは描画先
    reg = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train)
    training_accuracy.append(reg.score(X_train, y_train))
    test_accuracy.append(reg.score(X_test, y_test))
    ax.plot(line, reg.predict(line))  # -3 から 3 までのデータを評価
    ax.plot(X_train, y_train, '^', c=mg.cm2(0), markersize=2)
    ax.plot(X_test,  y_test,  'v', c=mg.cm2(1), markersize=2)

    ax.set_title(
        "n={}, train_score={:.2f}, test_score={:.2f}".format(n, reg.score(X_train, y_train), reg.score(X_test, y_test)))
    ax.set_xlabel("Feature")
    ax.set_ylabel("Target")

axis[0].legend(["Model predictions", "Training data/target",
                "Test data/target"], loc="best")

# nを増やすと予測(Model predictions)はスムーズになり，
# 訓練データに対する精度が下がる．
# 一方で# テストデータに対するスコアも下がっている．(n_samples=40の場合)
# 線形バイアスがかかってるデータなので，直感的にはスコアは上がりそうだが・・・なんで？
# どうやらサンプル数が少ないことが原因っぽい．
# n_samples=1024にすると，nが大きいほどtest_scoreも大きくなった．

# nを30まで増やしてみたが，n=9でtest_score=0.72でサチってるっぽい．
# グラフを見ると，nが増えるほどモデルはなめらかな正弦波に近づくが，
# ランダム変動を吸収できないので，これくらいの精度に落ち着いてしまうのでは．

plt.ylabel("Accuracy")  # 精度
plt.xlabel("n")
plt.plot([1, 3, 9, 15, 30], training_accuracy, label="training_accuracy")
plt.plot([1, 3, 9, 15, 30], test_accuracy, label="test_accuracy")
plt.legend()  # plotで設定したlabelを表示する

#%% -----------------------------------------------------------
# 線形モデル

mg.plots.plot_linear_regression_wave()


from sklearn.linear_model import LinearRegression

# 訓練データセットに対して特徴量が少ない場合
X, y = mg.datasets.make_wave(n_samples=100)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

lr = LinearRegression().fit(X_train, y_train)

lr.score(X_train, y_train)
lr.score(X_test, y_test)

lr.coef_
lr.intercept_

# 訓練データセットに対して特徴量が多い場合
X, y = mg.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

X_train.shape
X_test.shape
y_train.shape
y_test.shape

lr = LinearRegression().fit(X_train, y_train)

lr.score(X_train, y_train)  # 0.96(過剰適合)
lr.score(X_test, y_test)  # 0.65

lr.coef_
lr.intercept_


# リッジ回帰ではWが正則化されるので過剰適合を防げる
from sklearn.linear_model import Ridge
ridge = Ridge().fit(X_train, y_train)
ridge.score(X_train, y_train)  # 0.88
ridge.score(X_test, y_test)  # 0.75

ridge10 = Ridge(alpha=10.0).fit(X_train, y_train)
ridge10.score(X_train, y_train)  # 0.78
ridge10.score(X_test, y_test)  # 0.63 simpleになりすぎて，かえって性能低下した

ridge01 = Ridge(alpha=.1).fit(X_train, y_train)
ridge01.score(X_train, y_train)  # 0.92 (過剰適合っぽいが)
ridge01.score(X_test, y_test)  # 0.77 (汎化性能も上がっている)

plt.plot(ridge.coef_, "s", label="Ridge=1.0")
plt.plot(ridge10.coef_, "^", label="Ridge=10.0")
plt.plot(ridge01.coef_, "v", label="Ridge=0.10")
plt.plot(lr.coef_, "o", label="LinearRegression")

plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lr.coef_))
plt.ylim(-25, 25)
plt.legend()

# 線形回帰やalpha0.1のリッジ回帰では重みがばらついているのに対し，
# alphaが大きくなると，重みが0に近い値に正則化され，モデルがシンプルになっていることがわかる．

# 今度は訓練データのサイズを変化させたとき，線形回帰とRidge(alpha=1)で学習曲線を比較する
mg.plots.plot_ridge_n_samples()

# 線形回帰ではテストデータが少ないとほとんどなにも学習できない
# 十分な訓練データがある場合は，正則化はあまり重要ではない．


# Lasso
X, y = mg.datasets.load_extended_boston()
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

from sklearn.linear_model import Lasso
lasso = Lasso().fit(X_train, y_train)
lasso.score(X_train, y_train)  # 0.29
lasso.score(X_test, y_test)  # 0.20
np.sum(lasso.coef_ == 0)  # 100 の特徴量が捨てられ
np.sum(lasso.coef_ != 0)  # 4 つの特徴量だけが使われる

plt.plot(lasso.coef_, "o", label="Lasso")
plt.xlabel("Coefficient index")
plt.ylabel("Coefficient magnitude")
plt.hlines(0, 0, len(lasso.coef_))
plt.legend()


#%% ----------------------------------------------------
# 線形クラス分類
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
X, y = mg.datasets.make_forge()

fig, axis = plt.subplots(4, 2, figsize=(10, 14))
for c, axi in zip([0.01, 1.0, 10.0, 100.0], axis):
    for model, ax in zip([LogisticRegression, LinearSVC], axi):
        clf = model(C=c).fit(X, y)
        mg.plots.plot_2d_separator(
            clf, X, fill=False, eps=0.5, ax=ax, alpha=.7)
        mg.discrete_scatter(X[:, 0], X[:, 1], y, ax=ax)
        ax.set_title("{} C={}, s={}".format(
            clf.__class__.__name__, c, clf.score(X, y)))
        ax.set_xlabel("Feature 0")
        ax.set_ylabel("Feature 1")
        ax.legend()

# 高次元データ(cancer)だとどうか・・
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

c = 0.100
LogisticRegression(C=c).fit(X_train, y_train).score(X_test, y_test)
LogisticRegression(C=c).fit(X_train, y_train).score(X_train, y_train)
LinearSVC(C=c).fit(X_train, y_train).score(X_test, y_test)
LinearSVC(C=c).fit(X_train, y_train).score(X_train, y_train)


#%% ----------------------------------------------------
# 多クラス分類
from sklearn.datasets import make_blobs
X, y = make_blobs(random_state=40)
plt.scatter(X[:, 0], X[:, 1], c=y)
# 2次元3クラス

# LinearSVCで分類してみる
from sklearn.svm import LinearSVC
linear_svc = LinearSVC(C=10).fit(X, y)
print("X shape: ", X.shape)  # 100,2
print("Coefficient shape: ", linear_svc.coef_.shape)  # 係数配列 3,2
print("Intercept shape: ", linear_svc.intercept_.shape)  # 切片配列 3,

# 分類結果を見てみる
# plt.scatter(X[:,0], X[:,1], c=y)
mg.plots.plot_2d_classification(linear_svc, X, fill=True, alpha=.7)
mg.discrete_scatter(X[:, 0], X[:, 1], y)
line = np.linspace(-15, 15, 2)
for coef, intercept, color in zip(linear_svc.coef_, linear_svc.intercept_, ['b', 'r', 'g']):
    plt.plot(line, -(line * coef[0] + intercept) / coef[1], c=color)
# plt.ylim(-10, 15)
# plt.xlim(-10, 8)
plt.xlabel("Feature 0")
plt.ylabel("Feature 1")
plt.legend(['Class 0', 'Class 1', 'Class 2', 'Line class 0',
            'Line class 1', 'Line class 2'], loc=(1.01, 0.3))


#%% ----------------------------------------------------
# 決定木

from sklearn.tree import DecisionTreeClassifier

from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

plt.scatter(X[:, 0], X[:, 1], c=y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, random_state=0)
tree = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)
tree.score(X_test, y_test)


#%% graphvizで書き出す
tree_file = "tree.dot"
from sklearn.tree import export_graphviz
export_graphviz(tree, out_file=tree_file, class_names=[
                "malignant", "benign"], feature_names=cancer.feature_names, impurity=False, filled=True)

# pip install grahviz
# brew install graphviz
import graphviz
with open(tree_file) as f:
    dot_graph = f.read()
graphviz.Source(dot_graph)

#%% 特徴量の重要度(feature importance)
tree.feature_importances_


def plot_features_importances(model, dataset):
    importances = model.feature_importances_
    n_features = dataset.data.shape[1]
    plt.barh(range(n_features), importances, align='center')
    plt.yticks(np.arange(n_features), dataset.feature_names)
    plt.xlabel("model.feature_importances_")


plot_features_importances(tree, cancer)


#%% ランダムフォレスト
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

plt.scatter(X[:, 0], X[:, 1], c=y)

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(
    n_estimators=32, random_state=2).fit(X_train, y_train)
random_forest.score(X_test, y_test)

#%% visualize trees in RandomForest
fig, axes = plt.subplots(2, 3, figsize=(14, 10))
for i, (ax, tree) in enumerate(zip(axes.ravel(), random_forest.estimators_)):
    ax.set_title("Tree {}".format(i))
    mg.plots.plot_tree_partition(X_train, y_train, tree, ax=ax)
mg.plots.plot_2d_separator(random_forest, X_train,
                           fill=True, ax=axes[-1, -1], alpha=.4)
axes[-1, -1].set_title("Random Forest")
mg.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)


#%% 勾配ブースティング回帰木
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.25, random_state=3)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

plt.scatter(X[:, 0], X[:, 1], c=y)

from sklearn.ensemble import GradientBoostingClassifier
tree = GradientBoostingClassifier(learning_rate=0.05).fit(X_train, y_train)
tree.score(X_test, y_test)

#%% Cancerデータでやってみる
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

from sklearn.ensemble import GradientBoostingClassifier
tree = GradientBoostingClassifier(learning_rate=0.05).fit(X_train, y_train)
print("score:{}".format(tree.score(X_test, y_test)))

plot_features_importances(tree, cancer)

#%% カーネル法を用いたサポートベクタマシン ----------------------------------------------------
# 元データ
from sklearn.datasets import make_blobs
X, y = make_blobs(centers=4, random_state=8)
y %= 2
mg.discrete_scatter(X[:, 0], X[:, 1], y)

#%% LinearSVC(直線での分割)だとうまくいかない
from sklearn.svm import LinearSVC
model = LinearSVC().fit(X, y)
mg.plots.plot_2d_separator(model, X)
mg.discrete_scatter(X[:, 0], X[:, 1], y)

#%% 特徴量を加えて3次元で確認してみる
new_column = X[:, 1:]**2
X_new = np.hstack([X, new_column])

from mpl_toolkits.mplot3d import Axes3D, axes3d
figure = plt.figure()
ax = Axes3D(figure)

Xp = X_new[y == 1]
Xn = X_new[y == 0]
ax.scatter(Xn[:, 0], Xn[:, 1], Xn[:, 2], c='b')
ax.scatter(Xp[:, 0], Xp[:, 1], Xp[:, 2], c='r')

ax.set_xlabel("f0")
ax.set_ylabel("f1")
ax.set_zlabel("f1**2")

#%% ガウシアンカーネル
from sklearn.svm import SVC
X, y = mg.tools.make_handcrafted_dataset()

svm = SVC(kernel='rbf', C=10, gamma=.1).fit(X, y)  # Radial Basis Function =~ ガウシアンカーネル
svm.score(X, y)

#%% プロットする
mg.plots.plot_2d_separator(svm, X)
mg.discrete_scatter(X[:, 0], X[:, 1], y)

sv = svm.support_vectors_ # サポートベクタ: クラス境界付近の，境界を決定するデータポイント
sv_labels = svm.dual_coef_.ravel() > 0

mg.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)


#%% 各特徴量を正規化しない場合
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target)

svc = SVC().fit(X_train, y_train)
print('OriginalData Score: {}'.format(svc.score(X_test, y_test))) # 0.629

#%% MinMaxScalerの代わりに手動で正規化した場合
min = X_train.min(axis=0)
range = X_train.max(axis=0) - min

X_train_scaled = (X_train - min) / range
X_test_scaled = (X_test - min) / range
svc_scaled = SVC().fit(X_train_scaled, y_train)
print('Normalized Score: {}'.format(svc_scaled.score(X_test_scaled, y_test))) # 0.937

#%%

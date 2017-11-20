# 「Pythonで始める機械学習」のメモ

書籍が提供しているデータセット等便利パッケージ
```
pip install mglearn
```

## 1章 はじめに
### `1.7_iris_sample.py`
アイリスという花の分類をk-近傍法で行うサンプル．

+ データセットの読み込み(ndarray)
+ DataFrameにして`pd.scatter_matrix`でペアプロットして生データ確認
+ 訓練セットの分割: `from sklearn.model_selection import train_test_split`．訓練データ75%, テストデータ25%がデフォルト
+ k-近傍法: `from sklearn.neighbors import KNeighborsClassifier`
+ `scikit-learn`の教師あり学習では共通して，モデル構築:`fit`, 分類:`predict`, 評価:`score`が提供されている

## 2章 教師あり学習
+ クラス分類: ラベルの予測．選択肢の中からクラスラベルを予測すること
+ 回帰: 連続値の予測．量を予測する．
+ 過剰適合(overfitting): 情報量に比べて過度に複雑なモデルを作ってしまうこと
+ 適合不足(underfitting): 単純すぎるモデルを選択してしまうこと
+ 特徴量エンジニアリング(feature engineering): 測定結果としての特徴量間の積(交互作用)から導出した新たな特徴量を含めること．
+ 学習曲線(learning curve): モデルの性能をデータセットサイズの関数として示したもの
+ 訓練セットのスコアが高く，テストセットのスコアが低い場合，過剰適合の可能性が高い．
+ 訓練セットとテストセットの精度がとても近い場合，適合不足の可能性が高い．
+ 適合不足の場合は複雑なモデルになるよう調整する
+ 過剰適合の場合はシンプルなモデルになるよう調整する

### k-最近傍法(k-NN) : `クラス分類`
テストデータに，最も近い訓練データn点で多数決する．
nが小さいほど，出来上がるモデルは複雑であり，
nがサンプル数と等しい場合が最も単純であると考える．
一般に，モデルが複雑すぎても単純すぎても性能が低下する．

### k-近傍回帰 : `回帰`
テストデータに，最も近い訓練データn点の平均値を予測値とする．


### 線形モデル
入力特徴量の重み付き線形和が予測されるレスポンスyになる．各重みWとバイアスbがモデルを構成する．

訓練データのサンプル数より特徴量のほうが多い場合，どのようなyであっても完全にデータセットの線形関数としてモデル化できる．(それらが線形独立であると仮定？)
これによって過剰適合の危険がある．

特徴量が少ない場合，線形モデルはあまり適さない．(線形にしか分離できないので，どうやっても分離できないケースが多くなる)
特徴量が多く高次元となる場合は線形モデルによる分類は非常に強力．
そのため，特徴量が多い場合での過剰適合を回避する方法が重要になってくる．


*線形モデルの学習アルゴリズムごとの違い*
+ 重みとバイアスの特定の組み合わせと，訓練データの適合度を計る尺度(ロス関数)
+ 正則化するorしない，するならどの方法？(L1,L2 or else)

#### 線形回帰(最小二乗法) : `回帰`
訓練データに対してターゲットとの平均二乗誤差(mean squared error, 誤差の二乗の平均)が最小になるようW(重み，係数)とb(切片)を求めたもの．

線形回帰ではテストデータが少ないとほとんどなにも学習できない．

#### リッジ回帰(Ridge) : `回帰`
平均二乗誤差を最小化するのは線形回帰と同様．
Wを正則化(regularization)して，ここの特徴量が出力に与える影響をなるべく小さくすることで，モデルが単純となり，過剰適合を防ぐ．
汎化性能面では，*一般に線形回帰よりリッジ回帰を使ったほうが良い* ．

モデルの簡潔さ(0に近い係数の数)は，パラメータ`alpha`で制御できる．(デフォルトは1.0)
`alpha`が大きいほど，モデルは簡潔となるため，汎化にはそちらのほうがよい場合がある．
これらはデータに依存するため，パラメータを適切に設定する必要がある．
なお，十分な訓練データがある場合は，正則化はあまり重要ではなく，線形回帰と同じ性能を示す．

リッジ回帰の正則化はL2ノルム(各次元の重みを2乗した和=Wのユークリッド長)に対してペナルティを与える．(L2正則化)

#### Lasso : `回帰`
Lassoの正則化はL1ノルム(重みの絶対値の和)にペナルティを与える．(L1正則化)
Ridgeとの違いは，いくつかの重みは完全に0になるため，特徴量が選択されていると考えても良い．
(例えば，特徴量が104あって，そのうち33個だけを使うなど)


同様に`alpha`を減らすと複雑なモデルを構築して適合度合いを上げることができる．
`alpha`を調整することで，使う特徴量を制限しつつ，Ridgeと同程度の性能を出すことができる．

#### クラス分類のための線形モデル : `クラス分類`
重み付き線形和が0より大きいかどうかで分類する．

+ ロジスティック回帰(logistic regression) : `クラス分類`
+ 線形サポートベクタマシン(linear SVM) : `クラス分類`

回帰という名前がついているが，クラス分類のアルゴリズムである．
これらはデフォルトでL2正則化を行う(特徴量削減をしない)が，L1正則化を行うこともできる．
正則化の度合いはパラメータ`C`で制御され，Cが大きいほど正則化が弱くなり，複雑なモデルになる．RidgeやLassoの`alpha`とは逆．

正則化が弱くなると訓練データに対するスコアは上がるが，過剰適合しやすくなる．



### `2_SupervisedLearning.py`
+ `plt.scatter` で散布図を表示
+ `np.bincount(cancer.target)` で，値(ラベル)ごとに集計したndarrayを返す
+ 複数の線を同じ表に表示したければ，`plt.plot` を複数回コールする
+ k-最近傍法を回帰に対応させた`k-近傍回帰`がある．
+ k-近傍回帰: `from sklearn.neighbors import KNeighborsRegressor`
+ `plt.subplots` でトレリス表示できる
+ `reshape(n,m)` でデータを n行 m列にする．mかnに-1を指定すると要素数を元によしなにやってくれる
+ `np.linspace(-3, 3, 1000).reshape(-1, 1)` とかすると，1000行1列データにできる

+ `lr.coef_` : 重み W
+ `lr.intercept_` : バイアス,切片 b
+ `scikit-learn`では，訓練データから得られた属性には全て，末尾にアンダースコアをつける習慣がある．(`coef_`, `intercept_` など)
+ `LogisticRegression(penalty="l1")`として，どのようなペナルティ(ルール)で正則化するか指定できる


### その他
+ Jupyter Notebook を Atom で動かす Hydrogen がとても便利．
+ [matplotlibの基本的な使い方](https://qiita.com/Morio/items/d75159bac916174e7654)

### matplotlib.pyplotのメモ
+ `plt.ylim(-25, 25)` : y軸の表示範囲を制限する
+ `plt.hlines(8, 10, 42)` : y=8の傾き0の水平線を, x=10からx=42まで引く

+ 散布図
  ```py
  X.shape # (100,2)
  y.shape # (100,)

  plt.legend(["Class 0", "Class 1"], loc=4)
  plt.xlabel("First feature")
  plt.ylabel("Second feature")
  plt.scatter(X[:, 0], X[:, 1], c=y, marker='o')
  ```

  特徴量が1次元の場合はplotでもOK
  ```py
  X.shape
  y.shape

  plt.xlabel("Feature")
  plt.ylabel("Target")
  plt.plot(X, y, 'o')
  ```

+ トレリス(分割する)
  ```py
  # plt.subplots(行数, 列数, figsize=(5,10))
  fig, axis = plt.subplots(3, 1, figsize=(8,16)) # 3行1列に分割
  # このあと，axis 内の各 axに対して描画処理を行う

  # np.reshape(newshape)
  # Gives a new shape to an array without changing its data.
  # reshape(n,m) => n行 m列にする．-1を指定すると要素数を元によしなにやってくれる
  line = np.linspace(-3, 3, 1000).reshape(-1, 1) # 1000行1列
  for n, ax in zip([1,3,9], axis): # axは描画先
      reg = KNeighborsRegressor(n_neighbors=n).fit(X_train, y_train)
      ax.plot(line, reg.predict(line)) # -3 から 3 までのデータを評価
      ax.plot(X_train, y_train, '^', c=mg.cm2(0), markersize=8)
      ax.plot(X_test,  y_test,  'v', c=mg.cm2(1), markersize=8)

      ax.set_title(
          "n={}, train_score={:.2f}, test_score={:.2f}".format(n, reg.score(X_train, y_train), reg.score(X_test, y_test)))
      ax.set_xlabel("Feature")
      ax.set_ylabel("Target")

  # 最初のグラフにだけレジェンドを表示する．plotした後じゃないとだめなのでここで．
  axis[0].legend(["Model predictions", "Training data/target", "Test data/target"], loc="best")
  ```

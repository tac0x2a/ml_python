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

### k-最近傍法(k-NN)
テストデータに，最も近い訓練データn点で多数決する．
nが小さいほど，出来上がるモデルは複雑であり，
nがサンプル数と等しい場合が最も単純であると考える．
一般に，モデルが複雑すぎても単純すぎても性能が低下する．

### k-近傍回帰
テストデータに，最も近い訓練データn点の平均値を予測値とする．


### `2_SupervisedLearning.py`
+ `plt.scatter` で散布図を表示
+ `np.bincount(cancer.target)` で，値(ラベル)ごとに集計したndarrayを返す
+ 複数の線を同じ表に表示したければ，`plt.plot` を複数回コールする
+ k-最近傍法を回帰に対応させた`k-近傍回帰`がある．
+ k-近傍回帰: `from sklearn.neighbors import KNeighborsRegressor`
+ `plt.subplots` でトレリス表示できる
+ `reshape(n,m)` でデータを n行 m列にする．mかnに-1を指定すると要素数を元によしなにやってくれる
+ `np.linspace(-3, 3, 1000).reshape(-1, 1)` とかすると，1000行1列データにできる


### その他
+ Jupyter Notebook を Atom で動かす Hydrogen がとても便利．
+ [matplotlibの基本的な使い方](https://qiita.com/Morio/items/d75159bac916174e7654)

### matplotlib.pyplotのメモ
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

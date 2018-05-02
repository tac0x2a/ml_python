#「Pythonで始める機械学習」のメモ

書籍が提供しているデータセット等便利パッケージ
```
pip install mglearn
```

------------------------------------------------------------------------------
## 1章 はじめに
### 使用されるライブラリ等
+ `scikit-learn` : 機械学習アルゴリズムライブラリ．[ユーザガイド](http://scikit-learn.org/stable/user_guide.html)がある．
+ `Jupyter Notebook`: ブラウザでコードをインタラクティブに実行する環境．
+ `NumPy`: 科学技術計算ライブラリ．線形代数，擬似乱数生成器など．`scikit-learn`の基本データ構造であるNumPy配列(`ndarray`)を含む．
+ `SciPy`: 科学技術計算の関数群．`scikit-learn`でこれらの関数群を使用している．疎行列を表現する`scipi.sparse`を含む．
+ `matplotlib`: グラフ描画ライブラリ．
+ `pandas`: テーブル形式のデータ構造(DataFrame)のライブラリ．このテーブルにはクエリもできる．

### `1.7_iris_sample.py`
アイリスという花の分類をk-近傍法で行うサンプル．

+ データセットの読み込み(ndarray)
+ DataFrameにして`pd.scatter_matrix`でペアプロットして生データ確認
+ 訓練セットの分割: `from sklearn.model_selection import train_test_split`．訓練データ75%, テストデータ25%がデフォルト
  ```py
  from sklearn.model_selection import train_test_split
  X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  ```
+ k-近傍法: `from sklearn.neighbors import KNeighborsClassifier`
+ `scikit-learn`の教師あり学習では共通して，モデル構築:`fit`, 分類:`predict`, 評価:`score`が提供されている


------------------------------------------------------------------------------
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

------------------------------------------------------------------------------
### k-最近傍法(k-NN) : `クラス分類`
テストデータに，最も近い訓練データn点で多数決する．
nが小さいほど，出来上がるモデルは複雑であり，
nがサンプル数と等しい場合が最も単純であると考える．
一般に，モデルが複雑すぎても単純すぎても性能が低下する．

```py
from sklearn.neighbors import KNeighborsClassifier
KNeighborsClassifier(n_neighbors=n).fit(X_train, y_train).score(X_test, y_test)
```

### k-近傍回帰 : `回帰`
テストデータに，最も近い訓練データn点の平均値を予測値とする．

```py
from sklearn.neighbors.regression import KNeighborsRegressor
KNeighborsRegressor(n_neighbors=3).fit(X_train, y_train).score(X_test, y_test)
# 回帰問題なので R^2スコアを返す
```

### KNeighborsの特徴
+ 主なパラメータ
  + 近傍点数: 通常は3から5程度でOK
  + 距離尺度: 通常はユークリッド距離

+ 点数が増えた場合，モデル構築は高速(点を追加するだけなので)だが予測(点の走査)は遅くなる．
+ 数百を超える特徴量を持つデータではうまく機能せず，疎なデータセット(ほとんどの特徴量が0)では特に性能が悪いため，実際はほとんど使われてない．

------------------------------------------------------------------------------
### 線形モデル
入力特徴量の重み付き線形和が予測されるレスポンスyになる．各重みWとバイアスbがモデルを構成する．

訓練データのサンプル数より特徴量のほうが多い場合，どのような`y`であっても完全にデータセットの線形関数としてモデル化できる．(それらが線形独立であると仮定？)
これによって過剰適合の危険がある．

特徴量が少ない場合，線形モデルはあまり適さない．(線形にしか分離できないので，どうやっても分離できないケースが多くなる)
特徴量が多く高次元となる場合は線形モデルによる分類は非常に強力．
そのため，特徴量が多い場合での過剰適合を回避する方法が重要になってくる．

線形モデルによる回帰では，まずはRidge回帰を試してみて，特徴量が沢山あるが一部のみが重要であることがわかっている場合はLasso回帰を検討するのが良い．

回帰およびクラス分類のいずれにおいても，モデルの係数の重みは予測や分類の結果への影響度を示している．
しかしながら，`C`や`alpha`によってこれら係数の正負が入れ替わったりする場合もあるため，眉唾で解釈しなければならない．


##### サンプルが多い(10万,100万)場合
+ LogisticRegressionやRidgeの`solver='sag'`オプションを使うとデフォルトより高速になる場合がある．
+  SGDClassifier: `クラス分類` や SGDRegressor:`回帰` の仕様を検討する．

##### 主なパラメータと特徴
+ 正則化(ペナルティ)
  + `L1`: 一部の特徴量のみを使う．使用しない特徴量の重みは完全に0になる．一部の特徴量が重要であることがわかっている場合に使うと良い．
  + `L2`: 一部の特徴量を主に使う．重みは0にはならない．とくに理由がなければこちらを使うのが良い．
+ 正則化パラメータ: `alpha`が大きい場合や`C`が小さい場合，モデルが単純になる．
+ ロス関数: 訓練データの適合度を計る尺度．基本はユークリッド距離の最小二乗誤差

訓練が非常に高速で予測も高速．大きなデータセットで使われることが多いが，これは他のモデルでは学習できないため．
予測手法を理解しやすいが，特徴量間の相関がわかりにくく，係数の意味を理解しにくい．
特徴量の数がサンプル個数より多い場合に良い性能が出やすい．


#### 線形回帰(最小二乗法) : `回帰`
訓練データに対してターゲットとの平均二乗誤差(mean squared error, 誤差の二乗の平均)が最小になるようW(重み，係数)とb(切片)を求めたもの．

線形回帰ではテストデータが少ないとほとんどなにも学習できない．
一般的にRidge回帰やLasso回帰のほうが汎化性能が良いため，線形回帰を使用する機会はほぼ無い．

```py
from sklearn.linear_model import LinearRegression
LinearRegression().fit(X_train, y_train).score(X_test, y_test)
```

線形回帰はパラメータを必用とせずお手軽だが，逆にモデルの複雑さをコントロール出来ないことを意味する．
つまり，過剰適合を抑制する手段がない．(そのためにRidge回帰やLasso回帰を使う)
特に線形モデルでは，訓練データの特徴量が多く十分な訓練データ数を用意できない場合，過剰適合が起こりやすいため注意が必要．

#### リッジ回帰(Ridge) : `回帰`
平均二乗誤差を最小化するのは線形回帰と同様．
Wを正則化(regularization)して，ここの特徴量が出力に与える影響をなるべく小さくすることで，モデルが単純となり，過剰適合を防ぐ．
汎化性能面では，*一般に線形回帰よりリッジ回帰を使ったほうが良い* ．

```py
from sklearn.linear_model import Ridge
Ridge(alpha=1.0).fit(X_train, y_train).score(X_test, y_test)
```

モデルの簡潔さ(0に近い係数の数)は，パラメータ`alpha`で制御できる．(デフォルトは1.0)
`alpha`が大きいほど，モデルは簡潔となるため，汎化にはそちらのほうがよい場合がある．
これらはデータに依存するため，パラメータを適切に設定する必要がある．
なお，十分な訓練データがある場合は，正則化はあまり重要ではなく，線形回帰と同じ性能を示す．

リッジ回帰の正則化はL2ノルム(各次元の重みを2乗した和=Wのユークリッド長)に対してペナルティを与える．(L2正則化)

#### Lasso : `回帰`
Lassoの正則化はL1ノルム(重みの絶対値の和)にペナルティを与える．(L1正則化)
Ridgeとの違いは，いくつかの重みは完全に0になるため，特徴量が選択されていると考えても良い．
(例えば，特徴量が104あって，そのうち33個だけを使うなど)

```py
from sklearn.linear_model import Lasso
Lasso().fit(X_train, y_train).score(X_test, y_test)
```

同様に`alpha`を減らすと複雑なモデルを構築して適合度合いを上げることができる．
`alpha`を調整することで，使う特徴量を制限しつつ，Ridgeと同程度の性能を出すことができる．


------------------------------------------------------------------------------
#### クラス分類のための線形モデル : `クラス分類`
重み付き線形和が0より大きいかどうかで分類する．

+ ロジスティック回帰(logistic regression) : `クラス分類`
  ```py
  from sklearn.linear_model import LogisticRegression
  LogisticRegression(C=1.0, penalty='l2').fit(X_train, y_train).score(X_test, y_test)
  ```

+ 線形サポートベクタマシン(linear SVM) : `クラス分類`
  ```py
  from sklearn.svm import LinearSVC
  LinearSVC(C=1.0, penalty='l2').fit(X_train, y_train).score(X_test, y_test)
  ```

回帰という名前がついているが，クラス分類のアルゴリズムである．
これらはデフォルトでL2正則化を行う(特徴量削減をしない)が，L1正則化を行うこともできる．(`penalty`)
正則化の度合いはパラメータ`C`で制御され，Cが大きいほど正則化が弱くなり，複雑なモデルになる．RidgeやLassoの`alpha`とは逆．

正則化が弱くなると訓練データに対するスコアは上がるが，過剰適合しやすくなる．

#### 線形モデルによる多クラス分類
1対その他(one-vs.-rest)アプローチ: クラスごとに2クラス分類器(ベクトルと切片)を用意する．

どの分類器でもそれ以外に分類される領域については，クラス分類式の値が一番大きいクラス(つまりその点に最も近い線を持つクラス)に分類される．


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


------------------------------------------------------------------------------
## その他
+ Jupyter Notebook を Atom で動かす Hydrogen がとても便利．
+ [matplotlibの基本的な使い方](https://qiita.com/Morio/items/d75159bac916174e7654)
+ pandasのscatter_matrixがいい感じ
  ```python
  dataframe = pd.DataFrame(X, columns=data.feature_names)
  pd.scatter_matrix(dataframe, c=y, figsize=(16,16), hist_kwds={'bins':40}, s=10, alpha=.8)
  ```

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

### 用語とか
+ `COO(Coodinate)-format` : 0成分を省略したフォーマット．メモリ効率が良い．

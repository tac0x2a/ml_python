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

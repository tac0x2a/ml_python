# 「Pythonで始める機械学習」のメモ

### `1.7_iris_sample.py`
アイリスという花の分類をk-近傍法で行うサンプル．

+ データセットの読み込み(ndarray)
+ DataFrameにして`pd.scatter_matrix`でペアプロットして生データ確認
+ 訓練セットの分割: `from sklearn.model_selection import train_test_split`．訓練データ75%, テストデータ25%がデフォルト
+ k-近傍法: `from sklearn.neighbors import KNeighborsClassifier`
+ `scikit-learn`の教師あり学習では共通して，モデル構築:`fit`, 分類:`predict`, 評価:`score`が提供されている


### その他
+ Jupyter Notebook を Atom で動かす Hydrogen がとても便利．

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
# %matplotlib notebook では EvalError: Refused to evalute a string as JavaScript 'unsafe-value'

# import mglearn # 書籍が提供するサンプルデータ

# (ndarray: N-Dimension Array)
a = np.array([[1, 2, 3], [4, 5, 6]])


# 対角成分が1でそれ以外が0の、2次元NumPy配列を作る
eye = np.eye(4)
print("NumPy array:\n{}".format(eye))

# NumPy配列をSciPyのCSR形式の疎行列に変換する
# 非ゼロ要素だけが格納される
from scipy import sparse
sparse_matrix = sparse.csr_matrix(eye)
print("SciPy sparse CSR matrix:\n{}".format(sparse_matrix))


# -10から10の範囲を100要素に区切ったリストを生成
x = np.linspace(-10, 10, 100)
y = np.sin(x)

plt.plot(x, y, marker='.')


# pandas
# R の DataFrameを模したライブラリ．
# CSVの他にもSQLやエクセルファイルからデータを取り込める

import pandas as pd
data = {
    'Name': ["John", "Anna", "Peter", "Linda"],
    'Location': ["New York", "Paris", "Berlin", "London"],
    'Age': [24, 13, 53, 33]
}
data_pandas = pd.DataFrame(data)

data_pandas

# 条件に一致する行を抜き出す
data_pandas[data_pandas.Age > 30]

data_pandas.Age # 指定したカラムを抜き出す

data_pandas.Age > 30









































# -------------------------------------------------------

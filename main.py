import pandas as pd 
from sklearn.preprocessing import LabelEncoder


# データセットを読み込む
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

"""
print(train.dtypes)
print(train['MSZoning'].unique())
"""

# ラベルエンコーダー
for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))

"""
print(train.dtypes)
print(train['MSZoning'].unique())
"""
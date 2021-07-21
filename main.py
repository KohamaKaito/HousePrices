import pandas as pd 
from sklearn.preprocessing import LabelEncoder
import missingno as msno

import numpy as np # linear algebra
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from sklearn import linear_model


# データセットを読み込む
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

#print(train.info())


# ラベルエンコーダー
for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))


# 欠損値の対処
train = train.fillna(train.median())
test = test.fillna(test.median())


# keep ID for submission
train_ID = train['Id']
test_ID = test['Id']


# split data for training
y_train = train['SalePrice']
X_train = train.drop(['Id','SalePrice'], axis=1)
X_test = test.drop('Id', axis=1)

"""
# どの特徴量が影響力があるか
rf = RandomForestRegressor(n_estimators=80, max_features='auto')
rf.fit(X_train, y_train)
print('Training done using Random Forest')

ranking = np.argsort(-rf.feature_importances_)
f, ax = plt.subplots(figsize=(11, 9))
sns.barplot(x=rf.feature_importances_[ranking], y=X_train.columns.values[ranking], orient='h')
ax.set_xlabel("feature importance")
plt.tight_layout()
plt.show()
"""

X_train01 = X_train['OverallQual']
X_train02 = X_train['GrLivArea']
X_train = pd.concat([X_train01, X_train02],axis=1)

X_test01 = X_test['OverallQual']
X_test02 = X_test['GrLivArea']
X_test = pd.concat([X_test01, X_test02],axis=1)


reg = linear_model.LinearRegression()
reg.fit(X_train, y_train)
print (reg.predict(X_test))

y_pred = reg.predict(X_test)

# submission
submission = pd.DataFrame({
    "Id": test_ID,
    "SalePrice": y_pred
})
submission.to_csv('houseprice.csv', index=False)
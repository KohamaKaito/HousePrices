import pandas as pd 


# import data
train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

print(train.dtypes)

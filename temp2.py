# load csv file
import pandas as pd

df = pd.read_csv('Chen_jianxi.csv')

# print the first 5 rows
print(df.head())
# print row count
print(df.shape[0])
# print column count
print(df.shape[1])
# print column names
print(df.columns)

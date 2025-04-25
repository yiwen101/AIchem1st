# load csv file
import pandas as pd


df = pd.read_parquet("test-00000-of-00001.parquet")
# split df into 3 parts, write as part1, part2, part3 parquet files
assert len(df) == 1500
df1 = df.iloc[:500]
df2 = df.iloc[500:1000]
df3 = df.iloc[1000:]
df1.to_parquet("part1.parquet")
df2.to_parquet("part2.parquet")
df3.to_parquet("part3.parquet")


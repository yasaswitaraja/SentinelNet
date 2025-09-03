import pandas as pd
df = pd.read_csv("/workspaces/SentinelNet/nsl-kdd/KDDTest+.txt",header=None)
columns = ["duration","protocol_type","service","flag","src_bytes","dst_bytes","label"]
df.columns = columns +[f"feature_{i}" for i in range(len(df.columns)-len(columns))]
print(df.head())
print("\nData Types:\n",df.dtypes)
print("\nDetailed Info:\n")
df.info()
print("\nMissing values per column:\n",df.isnull().sum())
duplicates = df.duplicated().sum()
print(f"\nNumber of duplicate rows:{duplicates}")

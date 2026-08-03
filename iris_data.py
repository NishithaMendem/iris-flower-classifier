import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Iris.csv")

df = df.drop("Id", axis=1)

df.dropna(inplace=True)
print(df.isnull().sum())

print(df.drop_duplicates())

print(df["Species"].unique())
print(df["Species"].value_counts())

x = df.drop("Species", axis=1)
y = df["Species"]

print(x.head())
print(y.head())
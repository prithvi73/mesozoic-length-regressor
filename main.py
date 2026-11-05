import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Reading the data into a dataframe
df = pd.read_csv("Data/dinosaurs.csv")
df = df.dropna()
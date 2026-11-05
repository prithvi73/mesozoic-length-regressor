import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Reading the data into a dataframe
df = pd.read_csv("Data/dinosaurs.csv")
df = df.dropna()

# Dropping unnecessary columns
df = df.drop(columns=["name", "occurrence_no", "region", "lng", "lat"])

# Isolating the feature and target matrix
y = df["length_m"] # Target Matrix
df = df.drop(["length_m"], axis=1)

# Encoding the non-int cols as well
X = pd.get_dummies(df, dtype=int).to_numpy # Feature Matrix

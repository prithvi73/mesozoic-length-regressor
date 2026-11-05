import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Reading the data into a dataframe
df = pd.read_csv("Data/dinosaurs.csv")
df = df.dropna()

# Dropping unnecessary columns
df = df.drop(columns=["name", "occurrence_no", "region", "lng", "lat"])

# Isolating the feature and target matrix
y = df["length_m"] # Target Matrix
df = df.drop(["length_m"], axis=1)

# Encoding the non-int cols as well
X = pd.get_dummies(df, dtype=int).to_numpy() # Features Matrix

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Training the model
reg = RandomForestRegressor(random_state=42)
reg.fit(X_train_scaled, y_train)
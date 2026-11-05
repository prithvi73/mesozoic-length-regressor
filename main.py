import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

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

# Testing the model
y_pred = reg.predict(X_test_scaled)
mae = mean_absolute_error(y_test, y_pred)
print(mae)

# Seeing the models output on an individual piece of data
test_subject = X_test_scaled[0:1]
actual_length = y_test.to_numpy()[0]
predicted_length = reg.predict(test_subject)[0]

print(f"Actual length : {actual_length}")
print(f"Predicted length : {predicted_length}")
print(f"Difference (Error) : {abs(actual_length - predicted_length):.2f} meters")
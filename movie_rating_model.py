import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load Dataset
df = pd.read_csv("IMDb Movies India.csv")

# Step 2: Print actual column names
print("Columns in dataset:\n", df.columns.tolist())

# Step 3: Select and rename relevant columns
df = df[["Genre", "Director", "Actor 1", "Rating"]]
df.rename(columns={"Actor 1": "Star1", "Rating": "IMDB Rating"}, inplace=True)

# Step 4: Drop missing values
df.dropna(inplace=True)

# Step 5: Split features and target
X = df[["Genre", "Director", "Star1"]]
y = df["IMDB Rating"]

# Step 6: Encode categorical variables
label_encoders = {}
for col in X.columns:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Save encoder

# Step 7: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 8: Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))

# Step 10: Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred)
plt.xlabel("Actual IMDb Ratings")
plt.ylabel("Predicted IMDb Ratings")
plt.title("Actual vs Predicted IMDb Ratings")
plt.grid(True)
plt.show()

# Step 11: Save model and encoders
joblib.dump(model, "movie_rating_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("Model and encoders saved successfully.")

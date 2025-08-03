import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score
import xgboost as xgb
import os
import seaborn as sns
import joblib

# Load dataset
df = sns.load_dataset("penguins").dropna()

# Clean formatting
df["sex"] = df["sex"].str.lower()
df["island"] = df["island"].str.capitalize()

# Drop 'year' if it exists
df = df.drop(columns=["year"], errors="ignore")

# Encode target
le = LabelEncoder()
df["species"] = le.fit_transform(df["species"])

# One-hot encode features
df = pd.get_dummies(df, columns=["sex", "island"])

X = df.drop("species", axis=1)
y = df["species"]

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
model = xgb.XGBClassifier(n_estimators=100, max_depth=3, use_label_encoder=False, eval_metric="mlogloss")
model.fit(X_train, y_train)

print("Train F1 Score is:", f1_score(y_train, model.predict(X_train), average='weighted'))
print("Test F1 Score is:", f1_score(y_test, model.predict(X_test), average='weighted'))

# Save model + encoder + columns
os.makedirs("app/data", exist_ok=True)
model.save_model("app/data/model.json")
joblib.dump(le, "app/data/label_encoder.pkl")
joblib.dump(X.columns.tolist(), "app/data/columns.pkl")

print("Model, label encoder, and columns saved to app/data/")
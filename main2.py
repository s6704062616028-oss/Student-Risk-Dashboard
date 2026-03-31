import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

from imblearn.over_sampling import SMOTE
import joblib


# 1. LOAD DATA

df = pd.read_csv("Depression Student Dataset.csv")


# 2. CLEAN

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# 3. ENCODE

encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# 4. TARGET

X = df.drop("Depression", axis=1)
y = df["Depression"]

# 5. SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# 6. SMOTE

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# 7. SCALE

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 8. 🔥 NEURAL NETWORK (MLP)

model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation='relu',
    max_iter=300,
    random_state=42
)

print("Training...")
model.fit(X_train, y_train)
print("Done")

# 9. EVAL

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# 10. SAVE

joblib.dump(model,    "depression_nn.pkl")
joblib.dump(encoders, "depression_encoders.pkl")  
joblib.dump(scaler,   "depression_scaler.pkl")     
joblib.dump(X.columns.tolist(), "depression_features.pkl")

print("✅ Saved!")

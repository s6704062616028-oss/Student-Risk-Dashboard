import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from imblearn.over_sampling import SMOTE
import joblib

df = pd.read_excel("student_mental_health_burnout_1M.xlsx").sample(50000, random_state=42)

df["dropout_risk"] = df["dropout_risk"].apply(lambda x: 1 if x > 5 else 0)

print("STEP 1: Loaded")

# =========================
# 2. CLEAN
# =========================
num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

cat_cols = df.select_dtypes(include=['object', 'string']).columns
for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

print("STEP 2: Cleaned")

# =========================
# 3. ENCODE
# =========================
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

print("STEP 3: Encoded")

# =========================
# 4. KMEANS (Unsupervised)
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df["cluster"] = kmeans.fit_predict(df.drop("dropout_risk", axis=1))

# =========================
# 5. SCALE + PCA
# =========================
X_temp = df.drop("dropout_risk", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_temp)

pca = PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

# =========================
# 6. TARGET
# =========================
X = pd.DataFrame(X_pca)
y = df["dropout_risk"]

# =========================
# 7. SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# 8. SMOTE
# =========================
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# =========================
# 9. MODEL
# =========================
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42
)

print("Training...")
model.fit(X_train, y_train)

# =========================
# 10. EVAL
# =========================
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# =========================
# 11. SAVE
# =========================
joblib.dump(model, "model.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X_temp.columns.tolist(), "dropout_features.pkl")
joblib.dump(pca, "pca.pkl")
joblib.dump(df.drop("dropout_risk", axis=1).columns.tolist(), "all_features.pkl")

print("✅ Saved!")
# =========================
# 1. DATA LOADING & UNDERSTANDING
# =========================
import pandas as pd

df = pd.read_csv("data_ecommerce_customer_churn.csv")

print(df.head())
print(df.info())
print(df.describe())
# =========================

# =========================
# 2. DATA CLEANING
# =========================
print(df.isnull().sum())

# Fill specific important columns
df['DaySinceLastOrder'].fillna(df['DaySinceLastOrder'].median(), inplace=True)
df['WarehouseToHome'].fillna(df['WarehouseToHome'].median(), inplace=True)

# Fill remaining numeric NaNs
df.fillna(df.median(numeric_only=True), inplace=True)

# Final check
print("\nMissing values after cleaning:\n", df.isnull().sum())
# =========================

# =========================
# 3. ENCODING CATEGORICAL DATA
# =========================
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in ['PreferedOrderCat', 'MaritalStatus']:
    df[col] = le.fit_transform(df[col])
# =========================

# =========================
# 4. FEATURE ENGINEERING (RFM)
# =========================

df['Recency'] = df['DaySinceLastOrder']
df['Frequency'] = df['Tenure']
df['Monetary'] = df['CashbackAmount']

rfm = df[['Recency', 'Frequency', 'Monetary']]
# =========================

# =========================
# 5. DATA SCALING
# =========================
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm)
# =========================

# =========================
# 6. ELBOW METHOD (FIND CLUSTERS)
# =========================
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []

for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(rfm_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 10), inertia, marker='o')
plt.title("Elbow Method")
plt.xlabel("Clusters")
plt.ylabel("Inertia")
plt.show()

# =========================
# 7. K-MEANS CLUSTERING
# =========================
kmeans = KMeans(n_clusters=4, random_state=42)
df['Cluster'] = kmeans.fit_predict(rfm_scaled)

print(df['Cluster'].value_counts())
# =========================

# =========================
#8. CHURN PREDICTION MODEL
# =========================
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = df.drop(['Churn'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# =========================
# 9. FEATURE IMPORTANCE
# =========================
import matplotlib.pyplot as plt

importances = model.feature_importances_
features = X.columns

pd.Series(importances, index=features).sort_values().plot(kind='barh')
plt.title("Feature Importance")
plt.show()
# ========================= 

# =========================
# 10. AUC - ROC CURVE
# =========================
from sklearn.metrics import roc_curve, auc

# Get probability scores
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot
import matplotlib.pyplot as plt

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--')  # random line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# =========================
# 11.CORRELATION HEATMAP
# =========================
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# =========================
# 12. SAVE FINAL OUTPUT
# =========================
df.to_csv("final_output.csv", index=False)
# =========================

# SAVE MODELS 
# =========================
import joblib

joblib.dump(model, "churn_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(kmeans, "kmeans_model.pkl")

print("\n✅ Models saved successfully!")
# =========================

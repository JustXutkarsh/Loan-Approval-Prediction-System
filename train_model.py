# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
data = pd.read_csv("data/loan.csv")
 # Replace with your dataset path

# -------------------------------
# 2️⃣ Clean and preprocess columns
# -------------------------------
# Drop Loan_ID if exists
if 'Loan_ID' in data.columns:
    data = data.drop('Loan_ID', axis=1)

# Fill missing values
for col in data.columns:
    if data[col].dtype == 'object':
        data[col] = data[col].fillna(data[col].mode()[0])
    else:
        data[col] = data[col].fillna(data[col].median())

# Encode categorical variables
label_encoders = {}
for col in data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Save encoders for Streamlit use
joblib.dump(label_encoders, "label_encoders.pkl")

# -------------------------------
# 3️⃣ Balance dataset (fix imbalance)
# -------------------------------
if 'Loan_Status' in data.columns:
    df_majority = data[data.Loan_Status == 1]
    df_minority = data[data.Loan_Status == 0]
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )
    data = pd.concat([df_majority, df_minority_upsampled])

# -------------------------------
# 4️⃣ Split data
# -------------------------------
X = data.drop('Loan_Status', axis=1)
y = data['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 5️⃣ Train Random Forest model
# -------------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)
model.fit(X_train, y_train)

# -------------------------------
# 6️⃣ Evaluate
# -------------------------------
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# -------------------------------
# 7️⃣ Save model
# -------------------------------
joblib.dump(model, "loan_model.pkl")
print("\n✅ Model and encoders saved successfully!")



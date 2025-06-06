import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report
import numpy as np
import pickle
import os

# Load dataset
df = pd.read_excel("my_datasets.xlsx", sheet_name=0, header=0)

# Correct subject groups with consistent spelling
science_subjects = ['MATHEMATICS', 'BASIC TECH', 'COMPUTER', 'AGRIC SCIENCE', 'PHE']
commercial_subjects = ['MATHEMATICS', 'BUSINESS STUDIES', 'COMPUTER', 'SOCIAL STUDIES', 'ENGLISH']
arts_subjects = ['ENGLISH', 'YORUBA', 'SOCIAL STUDIES', 'CCA', 'IRS/CRS']

# Fixed feature order matching your Flask app input fields
feature_order = [
    "ENGLISH",         # english
    "MATHEMATICS",     # math
    "SOCIAL STUDIES",  # social
    "AGRIC SCIENCE",   # agric
    "PHE",             # phe
    "BASIC TECH",      # btech
    "COMPUTER",        # computer
    "BUSIESS STUDIES",# business
    "IRS/CRS",         # religious_studies
    "CCA",             # cca
    "YORUBA"           # yoruba
]

# Filter available features
available_subjects = [subj for subj in feature_order if subj in df.columns]

# Convert to numeric, fill NaNs
df[available_subjects] = df[available_subjects].apply(pd.to_numeric, errors='coerce').fillna(0)

# Define rule-based target for supervised learning
def assign_career_path(row):
    def passed_all(subjects):
        return all(row.get(subj, 0) >= 60 for subj in subjects)
    if passed_all(science_subjects):
        return "Science"
    elif passed_all(commercial_subjects):
        return "Commercial"
    elif passed_all(arts_subjects):
        return "Arts"
    else:
        return "General Studies"

df['CareerPath'] = df.apply(assign_career_path, axis=1)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df['CareerPath'])

# Prepare features matrix
X = df[available_subjects].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Compute class weights to handle imbalance
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
weight_map = {cls: weight for cls, weight in zip(np.unique(y_train), class_weights)}

# Create sample weights array
sample_weights = np.array([weight_map[label] for label in y_train])

# Train XGBoost classifier with sample weights
model = xgb.XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
model.fit(X_train, y_train, sample_weight=sample_weights)

# Predict on test set
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save model, label encoder, and config
os.makedirs("models", exist_ok=True)

with open("models/career_path_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

career_config = {
    "feature_order": available_subjects
}

with open("models/career_config.pkl", "wb") as f:
    pickle.dump(career_config, f)

print("\n✅ ML model training complete and saved with consistent feature order and class balancing.")

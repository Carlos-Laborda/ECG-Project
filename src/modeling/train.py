import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------------------------------------------------------------------
# Load and Prepare the Data
# ---------------------------------------------------------------------------------------------------------------------
# Load features.csv
data_path = "../../data/processed/features.parquet"
features_df = pd.read_parquet(data_path)

# Filter the dataset for the desired categories
features_df = features_df[
    features_df["category"].isin(["high_physical_activity", "baseline"])
]

# Map the target categories to binary labels
features_df["label"] = features_df["category"].map(
    {"baseline": 0, "high_physical_activity": 1}
)

# Drop non-feature columns
X = features_df[
    [
        "mean",
        "std",
        "min",
        "max",
        "rms",
        "iqr",
        "psd_mean",
        "psd_max",
        "dominant_freq",
        "shannon_entropy",
        "sample_entropy",
    ]
]
y = features_df["label"]

# Split the data by participant
participants = features_df["participant_id"].unique()
train_participants = participants[:8]  # First 8 participants for training
test_participants = participants[8:]  # Last 2 participants for testing

X_train = X[features_df["participant_id"].isin(train_participants)]
y_train = y[features_df["participant_id"].isin(train_participants)]
X_test = X[features_df["participant_id"].isin(test_participants)]
y_test = y[features_df["participant_id"].isin(test_participants)]

# ---------------------------------------------------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------------------------------------------------
# Standardize the feature values (mean=0, std=1)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------------------------------------------------
# Train the Model
# ---------------------------------------------------------------------------------------------------------------------
# Use a simple Random Forest Classifier for baseline
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# ---------------------------------------------------------------------------------------------------------------------
# Evaluate the Model
# ---------------------------------------------------------------------------------------------------------------------
# Predictions
y_pred = model.predict(X_test_scaled)

# Classification Report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(conf_matrix)

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Baseline", "High Physical Activity"],
    yticklabels=["Baseline", "High Physical Activity"],
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Accuracy Score
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {accuracy * 100:.2f}%")

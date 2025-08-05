import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
import joblib
import os

# Create output directory
os.makedirs("artifacts", exist_ok=True)

# Load data
data = pd.read_csv("data/iris.csv")

# Split data
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y_train = train['species'].values
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
y_test = test['species'].values

# Encode target labels to integers for logistic regression
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Initialize Logistic Regression with warm_start to True for incremental fitting
model = LogisticRegression(max_iter=1, warm_start=True, solver='lbfgs', multi_class='auto')

epochs = 20
metrics_records = []

for epoch in range(1, epochs + 1):
    model.fit(X_train, y_train_enc)  # Each call does one more iteration

    # Predict probabilities and classes on train and test sets
    train_probs = model.predict_proba(X_train)
    test_probs = model.predict_proba(X_test)
    train_preds = model.predict(X_train)
    test_preds = model.predict(X_test)

    # Calculate metrics
    train_acc = accuracy_score(y_train_enc, train_preds)
    train_loss = log_loss(y_train_enc, train_probs)
    val_acc = accuracy_score(y_test_enc, test_preds)
    val_loss = log_loss(y_test_enc, test_probs)

    print(f"Epoch {epoch} - train_acc: {train_acc:.3f}, train_loss: {train_loss:.3f}, val_acc: {val_acc:.3f}, val_loss: {val_loss:.3f}")

    metrics_records.append({
        "epoch": epoch,
        "accuracy": train_acc,
        "loss": train_loss,
        "val_accuracy": val_acc,
        "val_loss": val_loss
    })

# Save final model and label encoder
joblib.dump(model, "artifacts/model.joblib")
joblib.dump(le, "artifacts/label_encoder.joblib")

# Save metrics
metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv("artifacts/metrics.csv", index=False)

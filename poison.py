import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import joblib
import os

# Create output directory
os.makedirs("artifacts", exist_ok=True)

# Load original clean data
data = pd.read_csv("data/iris.csv")

# Poisoning function
def poison_data(df, percent):
    df_poisoned = df.copy()
    n_poison = int(len(df) * percent / 100)

    # Randomly select indices to poison
    poison_indices = np.random.choice(df.index, size=n_poison, replace=False)

    for idx in poison_indices:
        # Replace features with random values within feature ranges
        for col in ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']:
            min_val, max_val = df[col].min(), df[col].max()
            df_poisoned.at[idx, col] = np.random.uniform(min_val, max_val)

        # Change label to a random incorrect class
        current_label = df_poisoned.at[idx, 'species']
        other_labels = [label for label in df['species'].unique() if label != current_label]
        df_poisoned.at[idx, 'species'] = np.random.choice(other_labels)

    return df_poisoned

# Poison levels to test
poison_levels = [0, 5, 10, 50]

for level in poison_levels:
    print(f"\n=== Training with {level}% Poisoned Training Data ===")

    # Step 1: Split first (before poisoning)
    train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)

    # Step 2: Poison only the training data
    train_poisoned = poison_data(train, percent=level)

    # Extract features and labels
    X_train = train_poisoned[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y_train = train_poisoned['species'].values
    X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
    y_test = test['species'].values

    # Encode labels
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_test_enc = le.transform(y_test)

    # Initialize logistic regression model
    model = LogisticRegression(max_iter=1, warm_start=True, solver='lbfgs', multi_class='auto')

    epochs = 20
    metrics_records = []

    for epoch in range(1, epochs + 1):
        model.fit(X_train, y_train_enc)

        train_probs = model.predict_proba(X_train)
        test_probs = model.predict_proba(X_test)
        train_preds = model.predict(X_train)
        test_preds = model.predict(X_test)

        train_acc = accuracy_score(y_train_enc, train_preds)
        train_loss = log_loss(y_train_enc, train_probs)
        val_acc = accuracy_score(y_test_enc, test_preds)
        val_loss = log_loss(y_test_enc, test_probs)

        print(f"Epoch {epoch:02d} - train_acc: {train_acc:.3f}, train_loss: {train_loss:.3f}, val_acc: {val_acc:.3f}, val_loss: {val_loss:.3f}")

        metrics_records.append({
            "epoch": epoch,
            "train_accuracy": train_acc,
            "train_loss": train_loss,
            "val_accuracy": val_acc,
            "val_loss": val_loss
        })

    # Save model and metrics for this poison level
    tag = f"{level}pct"
    joblib.dump(model, f"artifacts/model_{tag}.joblib")
    joblib.dump(le, f"artifacts/label_encoder_{tag}.joblib")
    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(f"artifacts/metrics_{tag}.csv", index=False)

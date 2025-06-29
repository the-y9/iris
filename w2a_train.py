# train.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import joblib
import os

# Create output directory
os.makedirs("artifacts", exist_ok=True)

# Load data
data = pd.read_csv("data/iris.csv")

# Split data
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_train = train['species']
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y_test = test['species']

# Train model
mod_dt = DecisionTreeClassifier(max_depth=3, random_state=1)
mod_dt.fit(X_train, y_train)

# Predict and evaluate
prediction = mod_dt.predict(X_test)
accuracy = metrics.accuracy_score(prediction, y_test)
print(f"The accuracy of the Decision Tree is {accuracy:.3f}")

# Save model
joblib.dump(mod_dt, "artifacts/model.joblib")

# Save metrics
metrics_df = pd.DataFrame({"metric": ["accuracy"], "value": [accuracy]})
metrics_df.to_csv("artifacts/metrics.csv", index=False)

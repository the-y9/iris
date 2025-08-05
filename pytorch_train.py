import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, log_loss
import os

# Create output directory
os.makedirs("artifacts", exist_ok=True)

# Load data
data = pd.read_csv("data/iris.csv")

# Split data
train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
X_train = train[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.astype(np.float32)
y_train = train['species'].values
X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values.astype(np.float32)
y_test = test['species'].values

# Encode target labels to integers
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

# Convert numpy arrays to torch tensors
X_train_t = torch.from_numpy(X_train)
y_train_t = torch.from_numpy(y_train_enc).long()
X_test_t = torch.from_numpy(X_test)
y_test_t = torch.from_numpy(y_test_enc).long()

# Define logistic regression model (linear layer + softmax inside CrossEntropyLoss)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_dim, num_classes)
    def forward(self, x):
        return self.linear(x)

input_dim = X_train.shape[1]  # 4 features
num_classes = len(le.classes_)  # number of classes

model = LogisticRegressionModel(input_dim, num_classes)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

epochs = 20
metrics_records = []

for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train_t)  # logits
    loss = criterion(outputs, y_train_t)
    
    loss.backward()
    optimizer.step()
    
    # Calculate train accuracy
    _, predicted_train = torch.max(outputs, 1)
    train_acc = accuracy_score(y_train_enc, predicted_train.numpy())
    train_loss = loss.item()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_test_t)
        val_loss = criterion(val_outputs, y_test_t).item()
        _, predicted_val = torch.max(val_outputs, 1)
        val_acc = accuracy_score(y_test_enc, predicted_val.numpy())
        
        # For log_loss calculation, get softmax probabilities
        val_probs = torch.softmax(val_outputs, dim=1).numpy()
        train_probs = torch.softmax(outputs, dim=1).numpy()
        
        val_log_loss = log_loss(y_test_enc, val_probs)
        train_log_loss = log_loss(y_train_enc, train_probs)
    
    print(f"Epoch {epoch} - train_acc: {train_acc:.3f}, train_loss: {train_log_loss:.3f}, val_acc: {val_acc:.3f}, val_loss: {val_log_loss:.3f}")
    
    metrics_records.append({
        "epoch": epoch,
        "accuracy": train_acc,
        "loss": train_log_loss,
        "val_accuracy": val_acc,
        "val_loss": val_log_loss
    })

# Save model and label encoder (using torch.save for model and joblib for label encoder)
torch.save(model.state_dict(), "artifacts/model.pth")
import joblib
joblib.dump(le, "artifacts/label_encoder.joblib")

# Save metrics CSV
metrics_df = pd.DataFrame(metrics_records)
metrics_df.to_csv("artifacts/metrics_torch.csv", index=False)

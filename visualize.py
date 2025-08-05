import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSVs
metrics_df = pd.read_csv('artifacts/metrics.csv')
metrics_df1 = pd.read_csv('artifacts/metrics_torch.csv')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))  # 2 rows, 1 column (vertical stack)

# Plot Accuracy for both datasets on ax1
ax1.plot(metrics_df['epoch'], metrics_df['accuracy'], label='Train Accuracy (metrics.csv)', marker='o')
ax1.plot(metrics_df['epoch'], metrics_df['val_accuracy'], label='Val Accuracy (metrics.csv)', marker='o')
ax1.plot(metrics_df1['epoch'], metrics_df1['accuracy'], label='Train Accuracy (metrics_torch.csv)', marker='x')
ax1.plot(metrics_df1['epoch'], metrics_df1['val_accuracy'], label='Val Accuracy (metrics_torch.csv)', marker='x')
ax1.set_title('Accuracy over Epochs')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy')
ax1.grid(True)
ax1.legend(loc='lower right')

# Plot Loss for both datasets on ax2
ax2.plot(metrics_df['epoch'], metrics_df['loss'], label='Train Loss (metrics.csv)', color='orange', linestyle='--', marker='o')
ax2.plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss (metrics.csv)', color='red', linestyle='--', marker='o')
ax2.plot(metrics_df1['epoch'], metrics_df1['loss'], label='Train Loss (metrics_torch.csv)', color='green', linestyle='--', marker='x')
ax2.plot(metrics_df1['epoch'], metrics_df1['val_loss'], label='Val Loss (metrics_torch.csv)', color='purple', linestyle='--', marker='x')
ax2.set_title('Loss over Epochs')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

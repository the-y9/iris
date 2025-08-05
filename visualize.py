import pandas as pd
import matplotlib.pyplot as plt

# Load the data from poisoned metrics CSVs
metrics_df0 = pd.read_csv('artifacts/metrics_0pct.csv')
metrics_df5 = pd.read_csv('artifacts/metrics_5pct.csv')
metrics_df10 = pd.read_csv('artifacts/metrics_10pct.csv')
metrics_df50 = pd.read_csv('artifacts/metrics_50pct.csv')

# Set up plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 5))

# ----- Accuracy Plot -----
ax1.plot(metrics_df0['epoch'], metrics_df0['val_accuracy'], label='Val Accuracy (0% poison)', marker='o')
ax1.plot(metrics_df5['epoch'], metrics_df5['val_accuracy'], label='Val Accuracy (5% poison)', marker='s')
ax1.plot(metrics_df10['epoch'], metrics_df10['val_accuracy'], label='Val Accuracy (10% poison)', marker='^')
ax1.plot(metrics_df50['epoch'], metrics_df50['val_accuracy'], label='Val Accuracy (50% poison)', marker='x')

ax1.set_title('Validation Accuracy vs Epochs at Different Poison Levels')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Validation Accuracy')
ax1.grid(True)
ax1.legend(loc='lower right')

# ----- Loss Plot -----
ax2.plot(metrics_df0['epoch'], metrics_df0['val_loss'], label='Val Loss (0% poison)', linestyle='--', marker='o')
ax2.plot(metrics_df5['epoch'], metrics_df5['val_loss'], label='Val Loss (5% poison)', linestyle='--', marker='s')
ax2.plot(metrics_df10['epoch'], metrics_df10['val_loss'], label='Val Loss (10% poison)', linestyle='--', marker='^')
ax2.plot(metrics_df50['epoch'], metrics_df50['val_loss'], label='Val Loss (50% poison)', linestyle='--', marker='x')

ax2.set_title('Validation Loss vs Epochs at Different Poison Levels')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Validation Loss')
ax2.grid(True)
ax2.legend(loc='upper right')

plt.tight_layout()
plt.show()

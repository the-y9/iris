import unittest
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

class TestDecisionTreeModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load data and split
        data = pd.read_csv("data/iris.csv")
        train, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
        cls.X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
        cls.y_test = test['species']

        # Load model
        cls.model_path = "artifacts/model.joblib"
        cls.model = joblib.load(cls.model_path)

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists(self.model_path), "Model file not found.")

    def test_model_prediction_accuracy(self):
        predictions = self.model.predict(self.X_test)
        acc = accuracy_score(self.y_test, predictions)
        print(f"Test accuracy: {acc:.3f}")
        self.assertGreaterEqual(acc, 0.9, "Model accuracy is below expected threshold (0.9)")

    def test_metrics_file_exists_and_valid(self):
        metrics_path = "artifacts/metrics.csv"
        self.assertTrue(os.path.exists(metrics_path), "Metrics file not found.")

        metrics_df = pd.read_csv(metrics_path)
        self.assertIn("accuracy", metrics_df["metric"].values, "Accuracy metric not found in metrics file.")

        recorded_accuracy = metrics_df.loc[metrics_df["metric"] == "accuracy", "value"].values[0]
        predictions = self.model.predict(self.X_test)
        actual_accuracy = accuracy_score(self.y_test, predictions)

        # Allow small floating point tolerance
        self.assertAlmostEqual(recorded_accuracy, actual_accuracy, places=3, msg="Recorded accuracy does not match actual accuracy")

if __name__ == '__main__':
    unittest.main()

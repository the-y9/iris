import unittest
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import os

class TestLogisticRegressionModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load data and split
        data = pd.read_csv("data/iris.csv")
        _, test = train_test_split(data, test_size=0.4, stratify=data['species'], random_state=42)
        cls.X_test = test[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].values
        cls.y_test = test['species'].values

        # Load model and label encoder
        cls.model_path = "artifacts/model.joblib"
        cls.label_encoder_path = "artifacts/label_encoder.joblib"
        cls.model = joblib.load(cls.model_path)
        cls.le = joblib.load(cls.label_encoder_path)

    def test_model_file_exists(self):
        self.assertTrue(os.path.exists(self.model_path), "Model file not found.")

    def test_label_encoder_file_exists(self):
        self.assertTrue(os.path.exists(self.label_encoder_path), "Label encoder file not found.")

    def test_model_prediction_accuracy(self):
        # Encode y_test labels
        y_test_enc = self.le.transform(self.y_test)
        # Predict encoded labels
        predictions_enc = self.model.predict(self.X_test)

        acc = accuracy_score(y_test_enc, predictions_enc)
        print(f"Test accuracy: {acc:.3f}")
        self.assertGreaterEqual(acc, 0.9, "Model accuracy is below expected threshold (0.9)")

if __name__ == '__main__':
    unittest.main()

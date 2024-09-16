import unittest
import joblib
from train import train_model
from evaluate import evaluate_model

class TestEvaluateModel(unittest.TestCase):
    def setUp(self):
        # Load the processed data
        self.X_train, self.X_test, self.y_train, self.y_test, self.vectorizer = joblib.load('processed_data.pkl')
        # Train a model for testing
        self.model = train_model(self.X_train, self.y_train)

    def test_evaluate_model(self):
        results = evaluate_model(self.model, self.X_test, self.y_test)
        self.assertIn('accuracy', results)
        self.assertIn('report', results)
        self.assertGreaterEqual(results['accuracy'], 0)
        self.assertLessEqual(results['accuracy'], 1)

if __name__ == '__main__':
    unittest.main()
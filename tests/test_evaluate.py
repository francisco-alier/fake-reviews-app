import joblib
from train import train_model
from evaluate import evaluate_model

class TestEvaluateModel:
    def setup_method(self):
        # Load the processed data
        self.X_train, self.X_test, self.y_train, self.y_test, self.vectorizer = joblib.load('processed_data.pkl')
        # Train a model for testing
        self.model = train_model(self.X_train, self.y_train)

    def test_evaluate_model(self):
        results = evaluate_model(self.model, self.X_test, self.y_test)
        assert 'accuracy' in results
        assert 'report' in results
        assert 0 <= results['accuracy'] <= 1

# To run the tests, use the following command in the terminal:
# pytest test_evaluate.py
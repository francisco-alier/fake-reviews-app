import joblib
from train import train_model

class TestTrainModel:
    def setup_method(self):
        # Load the processed data
        self.X_train, self.X_test, self.y_train, self.y_test, self.vectorizer = joblib.load('processed_data.pkl')

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        assert model is not None
        assert hasattr(model, 'predict')

# To run the tests, use the following command in the terminal:
# pytest test_train.py
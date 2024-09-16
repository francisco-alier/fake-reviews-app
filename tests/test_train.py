import joblib
from train import train_model

class TestTrainModel(unittest.TestCase):
    def setUp(self):
        # Load the processed data
        self.X_train, self.X_test, self.y_train, self.y_test, self.vectorizer = joblib.load('processed_data.pkl')

    def test_train_model(self):
        model = train_model(self.X_train, self.y_train)
        self.assertIsNotNone(model)
        self.assertTrue(hasattr(model, 'predict'))

if __name__ == '__main__':
    unittest.main()
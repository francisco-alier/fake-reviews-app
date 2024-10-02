import joblib
from sklearn.linear_model import LogisticRegression

def train_model(X_train, y_train):
	"""
	Train a Logistic Regression model.

	Parameters:
	X_train (array-like): Training feature data.
	y_train (array-like): Training labels.

	Returns:
	model: Trained Logistic Regression model.
	"""
	model = LogisticRegression()
	model.fit(X_train, y_train)
	return model

if __name__ == "__main__":
	# Load the processed data
	X_train, X_test, y_train, y_test, vectorizer = joblib.load('processed_data.pkl')

	# Train the model
	model = train_model(X_train, y_train)

	# Save the trained model
	joblib.dump(model, 'trained_model.pkl')
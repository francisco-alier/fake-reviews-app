import joblib
from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
	"""
	Evaluate the performance of a trained model.

	Parameters:
	model: Trained model.
	X_test (array-like): Test feature data.
	y_test (array-like): Test labels.

	Returns:
	dict: Dictionary containing accuracy and classification report.
	"""
	y_pred = model.predict(X_test)
	accuracy = accuracy_score(y_test, y_pred)
	report = classification_report(y_test, y_pred, output_dict=True)
	return {"accuracy": accuracy, "report": report}

if __name__ == "__main__":
	# Load the processed data and trained model
	X_train, X_test, y_train, y_test, vectorizer = joblib.load('processed_data.pkl')
	model = joblib.load('trained_model.pkl')

	# Evaluate the model
	results = evaluate_model(model, X_test, y_test)
	print(f'Accuracy: {results["accuracy"]}')
	print(results["report"])
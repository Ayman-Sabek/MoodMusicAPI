from flask import Flask, request, jsonify
import joblib
import os
import logging
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load models
try:
    rf_model = joblib.load('best_random_forest_model.pkl')
    knn_model = joblib.load('best_knn_model.pkl')
    svm_model = joblib.load('best_svm_model.pkl')
    logging.info("Models loaded successfully.")
except FileNotFoundError as e:
    logging.error(f"Error loading models: {e}")
    raise e

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        # Ensure data is a list of dictionaries
        if not isinstance(data, list):
            return jsonify({'error': 'Input data should be a list of dictionaries'}), 400

        # Prepare features for each entry in the list
        features_list = []
        for entry in data:
            features = [
                entry.get('energy', 0),
                entry.get('valence', 0),
                entry.get('tempo', 0),
                entry.get('danceability', 0),
                entry.get('mood_score', 0),
                entry.get('log_tempo', 0),
                entry.get('energy_danceability', 0)
            ]
            features_list.append(features)

        rf_predictions = rf_model.predict(features_list)
        knn_predictions = knn_model.predict(features_list)
        svm_predictions = svm_model.predict(features_list)

        return jsonify({
            'Random Forest': rf_predictions.tolist(),
            'K-Nearest Neighbors': knn_predictions.tolist(),
            'Support Vector Machine': svm_predictions.tolist()
        })
    except NotFittedError as e:
        logging.error(f"Model not fitted: {e}")
        return jsonify({'error': 'Model not fitted properly'}), 500
    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        return jsonify({'error': f"An error occurred during prediction: {e}"}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

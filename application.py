from flask import Flask, request, jsonify
import joblib
import os

app = Flask(__name__)

rf_model = joblib.load('best_random_forest_model.pkl')
knn_model = joblib.load('best_knn_model.pkl')
svm_model = joblib.load('best_svm_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No data provided'}), 400
    
    features = [
        data.get('energy', 0),
        data.get('valence', 0),
        data.get('tempo', 0),
        data.get('danceability', 0),
        data.get('mood_score', 0),
        data.get('log_tempo', 0),
        data.get('energy_danceability', 0)
    ]
    
    rf_prediction = rf_model.predict([features])[0]
    knn_prediction = knn_model.predict([features])[0]
    svm_prediction = svm_model.predict([features])[0]

    return jsonify({
        'Random Forest': rf_prediction,
        'K-Nearest Neighbors': knn_prediction,
        'Support Vector Machine': svm_prediction
    })

if __name__ == '__main__':
    app.run(debug=True)

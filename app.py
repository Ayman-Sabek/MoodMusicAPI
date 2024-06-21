from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError

app = Flask(__name__)

# Load the trained models
rf_model = joblib.load('best_random_forest_model.pkl')
svm_model = joblib.load('best_svm_model.pkl')
knn_model = joblib.load('best_knn_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "Empty input data"}), 400

        df = pd.DataFrame(data)
        required_features = ['energy', 'valence', 'tempo', 'danceability', 'mood_score', 'log_tempo', 'energy_danceability']
        if not all(feature in df.columns for feature in required_features):
            return jsonify({"error": "Missing required features"}), 400

        predictions = {
            "Random Forest": rf_model.predict(df).tolist(),
            "Support Vector Machine": svm_model.predict(df).tolist(),
            "K-Nearest Neighbors": knn_model.predict(df).tolist()
        }
        return jsonify(predictions)
    except NotFittedError as e:
        return jsonify({"error": str(e)}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

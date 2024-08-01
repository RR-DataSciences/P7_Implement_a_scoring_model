from flask import Flask, request, jsonify
import logging
import pandas as pd
import numpy as np
import joblib
import sys
import os
import dill

# Assurez-vous que my_functions.py est dans le chemin
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from my_functions import custom_score
# Test d'appel à custom_score pour vérifier l'importation
print(custom_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/')
def welcome():
    logger.info("Bienvenue sur votre API !")
    return "V6 - Bienvenue sur votre API !"

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json(force=True)
    app.logger.debug(f"Data received: {data_json}")

    # Charger votre modèle ici
    model_path = "/home/ec2-user/P7_Implement_a_scoring_model/models/model_rawdata.pkl"
    model = joblib.load(model_path)
    # model = dill.load(model_path)
    app.logger.debug(f"Model loaded from {model_path}")

    scaler_path = '/home/ec2-user/P7_Implement_a_scoring_model/models/scaler_rawdata.pkl'
    scaler = joblib.load(scaler_path)
    app.logger.debug(f"Scaler loaded from {scaler_path}")

    # Convertir les données JSON en DataFrame
    df = pd.DataFrame(data_json)
    app.logger.debug(f"DataFrame: {df}")

    # Assurer que les colonnes du DataFrame sont dans le même ordre que celles attendues par le modèle
    estimator = model.estimator_
    expected_features = estimator.booster_.feature_name()
    app.logger.debug(f"Expected features: {expected_features}")
    app.logger.debug(f"DataFrame columns: {df.columns.tolist()}")
    # df = df[expected_features]
    # app.logger.debug(f"Reordered DataFrame: {df}")
    app.logger.debug(f"Reordered DataFrame: {df}")

    # Prétraitement des données si nécessaire
    data_scaled = scaler.transform(df)

    # Convertir les données transformées en DataFrame pour faciliter le traitement ultérieur
    df_transformed = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)
    app.logger.debug(f"Transformed DataFrame: {df_transformed}")
        
    df_transformed = np.array(df_transformed)

    # Faire la prédiction
    prediction = model.predict(df_transformed)
    score = model.predict_proba(df_transformed)
    app.logger.debug(f"Prediction: {prediction}, Score: {score}")

    # Renvoyer la prédiction, le score et les IDs
    return jsonify({
        'ids': df.index.tolist(),
        'prediction': prediction.tolist(),
        'score': score.tolist()
    })
    #except Exception as e:
    #    app.logger.error(f"Error: {str(e)}")
    #    return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

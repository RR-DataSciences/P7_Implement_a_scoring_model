from flask import Flask, request, jsonify
import logging
import pandas as pd
import numpy as np
import joblib
from my_functions import custom_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Charger votre modèle ici
model_path = "C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/models/307511_rawdata_LGBM-[24-06-07 at 18_28].pkl"
model = joblib.load(model_path)

scaler_path = 'C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/models/scaler_rawdata.pkl'
scaler = joblib.load(scaler_path)

@app.route('/')
def welcome():
    logger.info("Bienvenue sur votre API !")
    return "Bienvenue sur votre API !"

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json(force=True)
    app.logger.debug(f"Data received: {data_json}")

    try:
        # Convertir les données JSON en DataFrame
        df = pd.DataFrame(data_json)
        app.logger.debug(f"DataFrame: {df}")

        # Assurer que les colonnes du DataFrame sont dans le même ordre que celles attendues par le modèle
        estimator = model.estimator_
        expected_features = estimator.booster_.feature_name()
        app.logger.debug(f"columns modèle: {expected_features}")
        # df = df[expected_features]
        # app.logger.debug(f"Reordered DataFrame: {df}")

        # Prétraitement des données si nécessaire
        data_scaled = scaler.transform(df)

        # Convertir les données transformées en DataFrame pour faciliter le traitement ultérieur
        df_transformed = pd.DataFrame(data_scaled, columns=df.columns, index=df.index)
        app.logger.debug(f"df_transformed: {df_transformed}")
        
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
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

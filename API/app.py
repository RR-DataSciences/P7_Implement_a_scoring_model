from flask import Flask, request, jsonify
import logging
import pandas as pd
import numpy as np
import dill
import my_functions as MF
from my_functions import custom_score
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = Flask(__name__)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Loading the scaler
scaler_path = 'C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/models/scaler_rawdata.dill'
with open(scaler_path, 'rb') as file:
    scaler = dill.load(file)

# Loading the scaler
acp_path = 'C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/models/pca_307511_rawdata_pca_dill_LGBM-[24-08-02 at 08_13].dill'
with open(acp_path, 'rb') as file:
    pca = dill.load(file)

# Loading the model
model_path = "C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/models/307511_rawdata_pca_dill_LGBM-[24-08-02 at 08_13].dill"
# Ouvre le fichier en mode binaire et charge le modèle
with open(model_path, 'rb') as file:
    model = dill.load(file)

@app.route('/')
def welcome():
    logger.info("Bienvenue sur votre API !")
    return "V38 - Bienvenue sur votre API !"

@app.route('/predict', methods=['POST'])
def predict():
    data_json = request.get_json(force=True)
    app.logger.debug(f"Data received: {data_json}")

    try:
        # Convertir les données JSON en DataFrame
        df = pd.DataFrame(data_json)
        app.logger.debug(f"DataFrame: {df}")

        # Prétraitement des données si nécessaire
        data_scaled = scaler.transform(df)
        app.logger.debug(f"Scaler: {data_scaled}")

        # Convert the scaled data back to DataFrame to keep column names
        features = list(df.columns)
        data_scaled = pd.DataFrame.from_records(data_scaled, columns=features, index=df.index)
        app.logger.debug(f"Scaler_df: {data_scaled}")

        # Apply special character deletion to column names
        data_scaled.columns = data_scaled.columns.map(MF.replace_special_chars)
        app.logger.debug(f"Apply special character deletion: {data_scaled.columns}")

        # Dimension reduction
        app.logger.debug(f"Pre PCA: {pd.DataFrame(data_scaled).shape}")
        data_scaled_pca = pca.transform(data_scaled)
        app.logger.debug(f"Post PCA: {pd.DataFrame(data_scaled_pca).shape}")

        # Faire la prédiction
        prediction = model.predict(data_scaled_pca)
        score = model.predict_proba(data_scaled_pca)
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

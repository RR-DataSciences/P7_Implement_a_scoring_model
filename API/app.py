from flask import Flask, request, jsonify
from werkzeug.middleware.proxy_fix import ProxyFix
import logging
import pandas as pd
import numpy as np
import dill
import shap
import sys
print(sys.path)
import my_functions as MF
import Copie_my_functions as cMF
from my_functions import custom_score
# from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

app = Flask(__name__)

# Configuration de ProxyFix
app.wsgi_app = ProxyFix(
    app.wsgi_app,
    x_for=1,       # Nombre de proxys définissant l'en-tête X-Forwarded-For
    x_proto=1,     # Nombre de proxys définissant l'en-tête X-Forwarded-Proto
    x_host=1,      # Nombre de proxys définissant l'en-tête X-Forwarded-Host
    x_prefix=1     # Nombre de proxys définissant l'en-tête X-Forwarded-Prefix
)

# Configuration du logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Loading the scaler
scaler_path = '/home/ec2-user/P7_Implement_a_scoring_model/models/scaler_rawdata.dill'
with open(scaler_path, 'rb') as file:
    scaler = dill.load(file)

# Loading the rfe
rfe_path = '/home/ec2-user/P7_Implement_a_scoring_model/models/rfe_307511_rawdata_rfe_dill_v4_LGBM-[24-08-23 at 11_42].dill'
with open(rfe_path, 'rb') as file:
    rfe = dill.load(file)

# Loading the model
model_path = "/home/ec2-user/P7_Implement_a_scoring_model/models/307511_rawdata_rfe_dill_v4_LGBM-[24-08-23 at 11_42].dill"
# Ouvre le fichier en mode binaire et charge le modèle
with open(model_path, 'rb') as file:
    model = dill.load(file)

# Loading the explainer
explainer_path = "/home/ec2-user/P7_Implement_a_scoring_model/models/explainer.dill"
# Ouvre le fichier en mode binaire et charge le modèle
with open(explainer_path, 'rb') as file:
    explainer = dill.load(file)

@app.route('/')
def welcome():
    logger.info("Bienvenue sur votre API !")
    
    # HTML et CSS pour l'image de fond centrée
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Bienvenue sur votre API</title>
        <style>
            body, html {
                height: 100%;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                # background-image: url('/static/background.jpg');
                # background-size: cover;
                # background-position: center;
                # background-repeat: no-repeat;
            }
            .content {
                text-align: center;
                color: white;
                font-family: Arial, sans-serif;
                font-size: 20px;
                background-color: #E8A463; /* Fond semi-transparent pour le texte */
                padding: 10px;
                border-radius: 10px;
            }
        </style>
    </head>
    <body>
        <div class="content">
            <h1>V93 - Bienvenue sur votre API !</h1>
        </div>
    </body>
    </html>
    """
    return html_content

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
        data_scaled.columns = data_scaled.columns.map(cMF.replace_special_chars)
        app.logger.debug(f"Apply special character deletion: {data_scaled.columns}")

        # Dimension reduction
        app.logger.debug(f"Pre RFE: {pd.DataFrame(data_scaled).shape}")
        data_scaled_rfe = rfe.transform(data_scaled)
        app.logger.debug(f"Post RFE: {pd.DataFrame(data_scaled_rfe).shape}")

        rfe_columns = data_scaled.columns[rfe.support_]
        app.logger.debug(f"Select columns Post RFE: {data_scaled[rfe_columns]}")
 
        # Faire la prédiction
        prediction = model.predict(data_scaled_rfe)
        score = model.predict_proba(data_scaled_rfe)
        app.logger.debug(f"Prediction: {prediction}, Score: {score}")

        # Calculer les valeurs SHAP
        shap_values = explainer(data_scaled_rfe)
        # Convertir les valeurs SHAP en un format JSON sérialisable
        shap_values_list = shap_values.values.tolist()
        base_values_list = shap_values.base_values.tolist()
        app.logger.debug(f"Affiche les shap_values_list: {shap_values_list}")
        app.logger.debug(f"Affiche les base_values_list: {base_values_list}")

        shap_dict = [{'shap_values': sv, 'base_value': bv, 'features': f} 
            for sv, bv, f in zip(shap_values_list, base_values_list, data_scaled_rfe.tolist())]

        # Renvoyer la prédiction, le score et les IDs
        return jsonify({
            'ids': df.index.tolist(),
            'prediction': prediction.tolist(),
            'score': score.tolist(),
            'shap_details': shap_dict,
            'rfe_columns': rfe_columns.to_list()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # app.run(host="0.0.0.0", port=5000)
    app.run(debug=True)
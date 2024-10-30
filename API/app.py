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
from lightgbm import LGBMClassifier

app = Flask(__name__)

# Configuration de ProxyFix pour gérer les en-têtes proxy
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

# Chargement du scaler
scaler_path = '/home/ec2-user/P7_Implement_a_scoring_model/models/scaler_LGBM_rawdata_rfe.dill'
with open(scaler_path, 'rb') as file:
    scaler = dill.load(file)

# Chargement de la sélection de caractéristiques (RFE)
rfe_path = '/home/ec2-user/P7_Implement_a_scoring_model/models/rfe_LGBM_rawdata_rfe.dill'
with open(rfe_path, 'rb') as file:
    rfe = dill.load(file)

# Chargement du modèle LightGBM
model_path = "/home/ec2-user/P7_Implement_a_scoring_model/models/model_LGBM_rawdata_rfe.dill"
with open(model_path, 'rb') as file:
    model = dill.load(file)

# Chargement de l'expliqueur SHAP
explainer_path = "/home/ec2-user/P7_Implement_a_scoring_model/models/explainer_LGBM_rawdata_rfe.dill"
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
                justify-content: center;
                align-items: center;
            }
            body {
                width: 100%;
                background-image: url('/static/fond_API_GPT.jpeg');
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }
            .content {
                text-align: center;
                color: white;
                font-family: Arial, sans-serif;
                font-size: 20px;
                background-color: #F0EE60; /* Fond semi-transparent pour le texte */
                border-radius: 10px;
                margin: auto;
                border: 2px solid #53325C;
                width: 50%;
                margin-top: 700px;
            }
            .content h1 {
                padding: 2px;
                margin: 0px;
                padding: 5px 0px;
                color: #53325C;
            }
        </style>
    </head>
    <body>
        <div class="content">
            <h1>API - Modèle de scoring</h1>
        </div>
    </body>
    </html>
    """
    return html_content

@app.route('/predict', methods=['POST'])
def predict():
    # Récupération des données JSON envoyées dans la requête POST
    data_json = request.get_json(force=True)
    app.logger.debug(f"Données reçues : {data_json}")

    try:
        # Conversion des données JSON en DataFrame pandas
        df = pd.DataFrame(data_json)
        app.logger.debug(f"DataFrame : {df}")

        # Prétraitement des données avec le scaler
        data_scaled = scaler.transform(df)
        app.logger.debug(f"Données après scaling : {data_scaled}")

        # Reconversion en DataFrame pour conserver les noms de colonnes
        features = list(df.columns)
        data_scaled = pd.DataFrame.from_records(data_scaled, columns=features, index=df.index)
        app.logger.debug(f"Données après scaling (DataFrame) : {data_scaled}")

        # Suppression des caractères spéciaux dans les noms de colonnes
        data_scaled.columns = data_scaled.columns.map(cMF.replace_special_chars)
        app.logger.debug(f"Noms de colonnes après suppression des caractères spéciaux : {data_scaled.columns}")

        # Réduction de dimension avec RFE
        app.logger.debug(f"Avant RFE : {pd.DataFrame(data_scaled).shape}")
        data_scaled_rfe = rfe.transform(data_scaled)
        app.logger.debug(f"Après RFE : {pd.DataFrame(data_scaled_rfe).shape}")

        rfe_columns = data_scaled.columns[rfe.support_]
        app.logger.debug(f"Colonnes sélectionnées après RFE : {data_scaled[rfe_columns]}")
 
        # Faire la prédiction avec le modèle LightGBM
        prediction = model.predict(data_scaled_rfe)
        score = model.predict_proba(data_scaled_rfe)
        app.logger.debug(f"Prédiction : {prediction}, Score : {score}")

        # Calcul des valeurs SHAP pour l'explicabilité du modèle
        shap_values = explainer(data_scaled_rfe)
        # Conversion des valeurs SHAP en format JSON sérialisable
        shap_values_list = shap_values.values.tolist()
        base_values_list = shap_values.base_values.tolist()
        app.logger.debug(f"Valeurs SHAP : {shap_values_list}")
        app.logger.debug(f"Valeurs de base SHAP : {base_values_list}")

        # Création d'un dictionnaire pour les détails SHAP
        shap_dict = [{'shap_values': sv, 'base_value': bv, 'features': f} 
            for sv, bv, f in zip(shap_values_list, base_values_list, data_scaled_rfe.tolist())]

        # Retourne la prédiction, le score et les détails SHAP sous forme de JSON
        return jsonify({
            'ids': df.index.tolist(),
            'prediction': prediction.tolist(),
            'score': score.tolist(),
            'shap_details': shap_dict,
            'rfe_columns': rfe_columns.to_list()
        })
    except Exception as e:
        # Gestion des erreurs : retourne un message d'erreur si une exception se produit
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Lancement de l'application Flask
    # app.run(host="0.0.0.0", port=5000)
    app.run(debug=True)

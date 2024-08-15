import requests
import pandas as pd
import streamlit as st
import plotly.graph_objs as go

# Lire le fichier CSV
path = "C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/data/API_test"
data = pd.read_csv(f'{path}/test_imputed_df_api_20.csv', sep=';', index_col='SK_ID_CURR')

# Sélecteur pour choisir un individu
selected_id = st.selectbox("Sélectionnez un identifiant client", data.index)

# Extraire les données pour l'individu sélectionné
selected_data = data.loc[[selected_id]]  # On conserve le format DataFrame

# Convertir en JSON
data_json = selected_data.to_dict(orient='records')

# URL de l'API
url = "http://52.49.55.85:5000/predict"

# Envoyer une requête POST à l'API avec les données JSON
if st.button("Faire une prédiction"):
    response = requests.post(url, json=data_json)
    
    # Vérifier et obtenir la réponse
    if response.status_code == 200:
        prediction = response.json()
        score = prediction['score'][0]  # Récupérer le score pour l'individu sélectionné
        
        # Configurer la jauge avec Plotly
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score[1],  # Affiche la probabilité du score positif (1)
            title={'text': "Probabilité de défaut (1)"},
            gauge={
                'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "orange"},
                'steps': [
                    {'range': [0, 0.5], 'color': "lightgreen"},
                    {'range': [0.5, 1], 'color': "red"}],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': 0.5}}  # Seuil de 0.5 pour classification
        ))

        # Afficher la jauge
        st.plotly_chart(gauge)

    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
else:
    st.info("Veuillez sélectionner un client et appuyer sur le bouton pour faire une prédiction.")
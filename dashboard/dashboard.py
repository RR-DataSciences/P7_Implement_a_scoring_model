import requests
import pandas as pd
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px

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
url = "http://3.252.151.234:5000/predict"

print("Version 1")

if st.button("Faire une prédiction"):
    response = requests.post(url, json=data_json)
    
    if response.status_code == 200:
        prediction = response.json()
        st.write("Réponse de l'API:", prediction)  # Ajoutez ceci pour débogage
        if 'shap_values' in prediction:
            score = prediction['score'][0]
            
            # Configurer la jauge avec Plotly
            gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score[1],
                title={'text': "Probabilité de défaut de remboursement (1)"},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "orange"},
                    'steps': [
                        {'range': [0, 0.9], 'color': "indianred"},
                        {'range': [0.9, 1], 'color': "lightgreen"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 5},
                        'thickness': 0.9,
                        'value': 0.9}}  
            ))

            st.write(f"**Score de probabilité pour le client sélectionné ({selected_id}):**")
            st.write(f"Probabilité de défaut de remboursement (1) : {score[1]:.3f}")
            st.plotly_chart(gauge)

            # Afficher les valeurs SHAP
            shap_values = prediction['shap_values']
            shap_df = pd.DataFrame(shap_values, columns=data.columns)
            
            st.write(f"**Valeurs SHAP pour le client sélectionné ({selected_id}):**")
            for column in shap_df.columns:
                fig = px.bar(shap_df, x=shap_df.index, y=column, title=f"Impact des caractéristiques sur {column}")
                st.plotly_chart(fig)
        else:
            st.error("Les valeurs SHAP ne sont pas présentes dans la réponse.")
    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
else:
    st.info("Veuillez sélectionner un client et appuyer sur le bouton pour faire une prédiction.")
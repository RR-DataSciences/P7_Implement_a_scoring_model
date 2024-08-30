import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import shap
import matplotlib.pyplot as plt

# Lire le fichier CSV - train
path_train = "C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/data/API_test"
df_train = pd.read_csv(f'{path_train}/train_imputed_df_api.csv', sep=';', index_col='SK_ID_CURR')

# Lire le fichier CSV - test
path_test = "C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring/data/API_test"
df_test = pd.read_csv(f'{path_test}/test_imputed_df_api_20.csv', sep=';', index_col='SK_ID_CURR')

# Sélecteur pour choisir un individu
selected_id = st.selectbox("Sélectionnez un identifiant client", df_test.index)

# Extraire les données pour l'individu sélectionné
selected_data = df_test.loc[[selected_id]]  # On conserve le format DataFrame

# Convertir en JSON
data_json = selected_data.to_dict(orient='records')

# URL de l'API
url = "http://34.254.146.135:5000/predict"

st.write(f"**Version 1**")

if st.button("Faire une prédiction"):
    response = requests.post(url, json=data_json)
    
    if response.status_code == 200:
        prediction = response.json()
        # st.write("Réponse de l'API:", prediction)  # Ajoutez ceci pour débogage
        if 'shap_details' in prediction:
            shap_details = prediction['shap_details']
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

            
            # st.write(f"Description du dataset: {data.shape}")
            # st.write(f"Description du dataset: {prediction['rfe_columns']}")
            # st.write(f"Description du dataset: {prediction['shap_details']}")

            # Since shap_details is a list, you need to access its first element (or the one you need)
            shap_detail = shap_details[0]  # Assuming you want the first dictionary in the list

            # Extracting shap_values, base_value, and features from the dictionary
            shap_values = np.array(shap_detail['shap_values'])
            base_value = shap_detail['base_value']
            features = np.array(shap_detail['features'])
            feature_names = prediction['rfe_columns']
            
            # Vérification initiale des données
            st.write("Données SHAP disponibles :", len(shap_details))
            st.write("Nombre de colonnes dans shap_values :", shap_values.shape if hasattr(shap_values, 'shape') else None)
            st.write("Nombre de colonnes dans feature_names :", len(feature_names))

            # Calcul des scores absolus pour chaque valeur
            scores_abs = np.abs(shap_values)
            # st.write("Scores absolus obtenus :", scores_abs)

            # Création d'un DataFrame avec les scores absolus et les noms des caractéristiques
            df_scores = pd.DataFrame({
                'feature': feature_names,
                'score': scores_abs.flatten()  # Flatten l'array 2D en 1D
            })
            # Tri des scores par ordre décroissant
            df_scores_sorted = df_scores.sort_values('score', ascending=False)
            # st.write("DataFrame des scores absolus (ordre décroissant) :", df_scores_sorted)
            
            # Sélection des trois variables les plus influentes
            top_features = df_scores_sorted.head(3)
            top_features_list = top_features['feature'].tolist()
            
            # Affichage des variables les plus influentes
            st.write("### Variables les plus influentes pour la prédiction")
            for index, row in top_features.iterrows():
                st.write(f"{index+1}. {row['feature']}: SHAP score = {row['score']:.2f}")
            
            # Creating a SHAP Explanation object
            shap_exp = shap.Explanation(values=shap_values, 
                                        base_values=base_value, 
                                        data=features, 
                                        feature_names=feature_names)

            # Displaying the SHAP Force Plot
            st.write(f"**SHAP Force Plot for the selected client ({selected_id}):**")
            shap.force_plot(shap_exp.base_values, shap_exp.values, shap_exp.data, feature_names=shap_exp.feature_names, text_rotation=25, matplotlib=True)
            st.set_option('deprecation.showPyplotGlobalUse', False)
            st.pyplot()

            st.write(f"Affiche les données d'entrainements: {df_train}")

        else:
            st.error("Les valeurs SHAP ne sont pas présentes dans la réponse.")
    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
else:
    st.info("Veuillez sélectionner un client et appuyer sur le bouton pour faire une prédiction.")
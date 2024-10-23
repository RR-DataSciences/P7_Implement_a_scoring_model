import requests
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff
import shap
import matplotlib.pyplot as plt
from plotly.subplots import make_subplots

projet_7 = 'C:/Users/remid/Documents/_OC_ParcoursDataScientist/P7_Implémentez_un_modèle_de_scoring'

# Lire le fichier CSV - train
path_train = f"{projet_7}/data/API_test"
df_train = pd.read_csv(f'{path_train}/train_imputed_df_api.csv', sep=';', index_col='SK_ID_CURR')

# Lire le fichier CSV - test
path_test = f"{projet_7}/data/API_test"
df_test = pd.read_csv(f'{path_test}/test_imputed_df_api_20.csv', sep=';', index_col='SK_ID_CURR')

# Fonction pour lire le fichier CSS
def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# Charger le fichier CSS
local_css(f"{projet_7}/P7_Implement_a_scoring_model/dashboard/style.css")

st.sidebar.markdown("<h1 class='naming'>Prêt à dépenser</h1>", unsafe_allow_html=True)

# Ajouter un logo dans la barre latérale
logo_path = f"{projet_7}/P7_Implement_a_scoring_model/dashboard/images/Logo_GPT_v2.png"
st.sidebar.image(logo_path, use_column_width="auto")

st.header("Dashboard de simulation pour l'attribution d'un prêt bancaire")

# Placer le sélecteur dans la barre latérale
selected_id = st.sidebar.selectbox("Sélectionnez un identifiant client", df_test.index)

# Extraire les données pour l'individu sélectionné
selected_data = df_test.loc[[selected_id]]  # On conserve le format DataFrame

# Convertir en JSON
data_json = selected_data.to_dict(orient='records')

# URL de l'API
url = "http://54.229.88.7:5000/predict"

if st.sidebar.button("Lancer la simulation"):
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
                title={'text': "Score du client"},
                gauge={
                    'axis': {'range': [0, 1], 'tickwidth': 1, 'tickcolor': "white"},
                    'bar': {'color': "#F0EE60"},
                    'steps': [
                        {'range': [0, 0.9], 'color': "indianred"},
                        {'range': [0.9, 1], 'color': "lightgreen"}],
                    'threshold': {
                        'line': {'color': "black", 'width': 5},
                        'thickness': 0.9,
                        'value': 0.9}}  
            ))

            # Supprimer la couleur de fond (mettre transparent)
            gauge.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',  # Fond transparent
                plot_bgcolor='rgba(0,0,0,0)',    # Fond transparent du graphique
                width=300,
                height=300
            )
            
            # Titre de la partie résultat
            st.markdown(f"<h1 class='simulation'>[Client {selected_id}] - Résultat de la simulation</h1>", unsafe_allow_html=True)
            # st.write(f"Probabilité de défaut de remboursement (1) : {score[1]:.3f}")
            
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
            # st.write("Données SHAP disponibles :", len(shap_details))
            # st.write("Nombre de colonnes dans shap_values :", shap_values.shape if hasattr(shap_values, 'shape') else None)
            # st.write("Nombre de colonnes dans feature_names :", len(feature_names))

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

            # Créer deux colonnes
            col1, col2 = st.columns([1, 1])  # Les valeurs dans la liste [1, 1] définissent la largeur des colonnes. Vous pouvez ajuster ces valeurs.

            # Afficher la jauge dans la première colonne
            with col1:
                # Affichage des variables les plus influentes
                # st.write("### Variables les plus influentes pour la prédiction")
                # for index, row in top_features.iterrows():
                #     st.write(f"{index+1}. {row['feature']}: SHAP score = {row['score']:.2f}")
                # Ajout du client sélection au données comparatives
                selected_customer = df_test.loc[df_test.index == selected_id]
                # st.write(f"Affiche les données d'entrainements: {selected_customer}")
                if score[1] < 90:
                    st.markdown(f"<h2 class='proba'>Probabilité de défaut de remboursement:</h2> </ br>{score[1]:.3f}", unsafe_allow_html=True)
                    target = 0
                    st.write(f"Target: {target}")
                    selected_customer['TARGET'] = target
                elif score[1] >= 90:
                    st.markdown(f"<h2 class='proba'>Probabilité de défaut de remboursement:</h2> </ br>{score[1]:.3f}", unsafe_allow_html=True)
                    target = 1
                    st.write(f"Target: {target}")
                    selected_customer['TARGET'] = target
            with col2:
                st.plotly_chart(gauge)

            # Creating a SHAP Explanation object
            shap_exp = shap.Explanation(values=shap_values, 
                                        base_values=base_value, 
                                        data=features, 
                                        feature_names=feature_names)

            tab1, tab2, tab3 = st.tabs(["SHAP - Poids des variables dans la prédiction", "Distribution des 3 variables les plus influentes", "Owl"])

            with tab1:
                # Displaying the SHAP Force Plot
                st.write(f"**SHAP Force Plot for the selected client ({selected_id}):**")
                shap.force_plot(shap_exp.base_values, shap_exp.values, shap_exp.data, feature_names=shap_exp.feature_names, text_rotation=25, matplotlib=True)
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot()
            with tab2:
                df_best_features = df_train[top_features_list+['TARGET']]
                # st.write(f"Affiche les données d'entrainements: {df_best_features}")
                
                # Création d'une grille de sous-graphiques
                fig = make_subplots(rows=1, cols=len(top_features_list), subplot_titles=top_features_list)

                # Créer des histogrammes pour chaque caractéristique sélectionnée
                for i, feature in enumerate(top_features_list):
                    x1 = df_best_features.loc[df_best_features['TARGET'] == 0, feature]
                    x2 = df_best_features.loc[df_best_features['TARGET'] == 1, feature]
                    
                    fig.add_trace(go.Histogram(x=x1, name='Clients ayant remboursé', marker_color='slategray'), row=1, col=i+1)
                    fig.add_trace(go.Histogram(x=x2, name='Clients avec défaut', marker_color='magenta'), row=1, col=i+1)

                # Mettre à jour le layout pour une meilleure lisibilité
                fig.update_layout(barmode='overlay')
                fig.update_traces(opacity=0.75)

                # Ajuster la taille et afficher la figure
                fig.update_layout(width=1200, height=400)
                st.plotly_chart(fig, use_container_width=True)
                with tab3:
                    st.header("An owl")
                    st.image("https://static.streamlit.io/examples/owl.jpg", width=200)

            # st.write(f"Target: {selected_customer[top_features_list+['TARGET']]}")

        else:
            st.error("Les valeurs SHAP ne sont pas présentes dans la réponse.")
    else:
        st.error(f"Erreur lors de la requête API: {response.status_code}")
else:
    st.info("Veuillez sélectionner un client et appuyer sur le bouton pour lancer une simulation.")
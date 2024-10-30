import sys
import dill
import pandas as pd
import numpy as np
import unittest

# Ajout du chemin du projet à la liste des chemins Python
sys.path.append('/home/ec2-user/P7_Implement_a_scoring_model')

# Importation des fonctions nécessaires
import my_functions as MF
from my_functions import train_models, missing_values_table, replace_special_chars, custom_score


class TestScoringModel(unittest.TestCase):
    """
    Classe contenant les tests unitaires pour le modèle de scoring.
    """

    def test_missing_values_table(self):
        """
        Test de la fonction missing_values_table.
        """
        # Créer un DataFrame de test avec des valeurs manquantes
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, None, 7, 8]
        })
        
        result = missing_values_table(df)
        
        # Vérifications
        self.assertEqual(result.shape[0], 2)
        self.assertIn('Missing Values', result.columns)
        self.assertIn('% of Total Values', result.columns)

    def test_replace_special_chars(self):
        """
        Test de la fonction replace_special_chars.
        """
        self.assertEqual(replace_special_chars("test@123"), "test_123")
        self.assertEqual(replace_special_chars("test space"), "test_space")

    def test_custom_score(self):
        """
        Test de la fonction custom_score.
        """
        y_test = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        
        score = custom_score(y_test, y_pred)
        self.assertEqual(score, 11)

    def test_predict(self):
        """
        Test de la fonction predict en utilisant les modèles chargés.
        """
        # Chemin du projet
        projet_7 = "/home/ec2-user/P7_Implement_a_scoring_model"

        ###################### Import des modèles ######################
        # Chargement du scaler
        scaler_path = f'{projet_7}/models/scaler_LGBM_rawdata_rfe.dill'
        with open(scaler_path, 'rb') as file:
            scaler = dill.load(file)
        
        # Chargement de RFE
        rfe_path = f'{projet_7}/models/rfe_LGBM_rawdata_rfe.dill'
        with open(rfe_path, 'rb') as file:
            rfe = dill.load(file)
        
        # Chargement du modèle LightGBM
        model_path = f'{projet_7}/models/model_LGBM_rawdata_rfe.dill'
        with open(model_path, 'rb') as file:
            model = dill.load(file)
        
        # Chargement de l'expliqueur SHAP
        explainer_path = f'{projet_7}/models/explainer_LGBM_rawdata_rfe.dill'
        with open(explainer_path, 'rb') as file:
            explainer = dill.load(file)

        ###################### Traitement des données ######################
        # Lecture des données de test
        data = pd.read_csv(f'{projet_7}/tests/test_data.csv', sep=';', index_col='SK_ID_CURR')
        
        # Prétraitement des données
        data_scaled = scaler.transform(data)
        features = list(data.columns)
        data_scaled = pd.DataFrame(data_scaled, columns=features, index=data.index)
        
        # Suppression des caractères spéciaux dans les noms de colonnes
        data_scaled.columns = data_scaled.columns.map(MF.replace_special_chars)
        
        # Réduction de dimension avec RFE
        print(f"Description avant RFE : {pd.DataFrame(data_scaled).shape}")
        data_scaled_rfe = rfe.transform(data_scaled)
        print(f"Description après RFE : {pd.DataFrame(data_scaled_rfe).shape}")

        ###################### Prédiction ######################
        prediction = model.predict(data_scaled_rfe)
        score = model.predict_proba(data_scaled_rfe)
        print(f"Résultats de prédiction : {prediction}, Score : {score}")

        ###################### Expliqueur SHAP ######################
        shap_values = explainer(data_scaled_rfe)
        print(f"Description des valeurs SHAP : {shap_values.shape}")

if __name__ == '__main__':
    # Exécution des tests unitaires
    unittest.main()
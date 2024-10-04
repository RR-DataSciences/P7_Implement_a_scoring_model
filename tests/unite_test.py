import sys
import dill
import requests
import unittest
# setting path
sys.path.append('/home/ec2-user/P7_Implement_a_scoring_model/API')
from Copie_my_functions import train_models, missing_values_table, replace_special_chars, custom_score
import pandas as pd
import numpy as np

class TestScoringModel(unittest.TestCase):

    def test_missing_values_table(self):
        # Créer un DataFrame de test avec des valeurs manquantes
        df = pd.DataFrame({
            'A': [1, 2, None, 4],
            'B': [5, None, 7, 8]
        })
        
        result = missing_values_table(df)
        
        self.assertEqual(result.shape[0], 2)
        self.assertIn('Missing Values', result.columns)
        self.assertIn('% of Total Values', result.columns)

    def test_replace_special_chars(self):
        self.assertEqual(replace_special_chars("test@123"), "test_123")
        self.assertEqual(replace_special_chars("test space"), "test_space")

    def test_custom_score(self):
        y_test = np.array([0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1])
        
        score = custom_score(y_test, y_pred)
        self.assertEqual(score, 11)

    def test_predict(self):
        projet_7 = "/home/ec2-user/P7_Implement_a_scoring_model"
        # # Loading the scaler
        # scaler_path = f'{projet_7}/scaler_rawdata.dill'
        # with open(scaler_path, 'rb') as file:
        #     scaler = dill.load(file)

        # # Loading the rfe
        # rfe_path = f'{projet_7}/models/rfe_307511_rawdata_rfe_dill_v4_LGBM-[24-08-23 at 11_42].dill'
        # with open(rfe_path, 'rb') as file:
        #     rfe = dill.load(file)

        # # Loading the model
        # model_path = f'{projet_7}/models/307511_rawdata_rfe_dill_v4_LGBM-[24-08-23 at 11_42].dill'
        # # Ouvre le fichier en mode binaire et charge le modèle
        # with open(model_path, 'rb') as file:
        #     model = dill.load(file)

        # # Loading the explainer
        # explainer_path = f'{projet_7}/models/explainer.dill'
        # # Ouvre le fichier en mode binaire et charge le modèle
        # with open(explainer_path, 'rb') as file:
        #     explainer = dill.load(file)

        data = pd.read_csv(f'{projet_7}/tests/test_data.csv', sep=';', index_col='SK_ID_CURR')
        # Convertir en JSON
        data_json = data.to_dict(orient='records')

        import os
        url = f"http://{os.getenv('AWS_EC2')}/predict"

        # Envoyer une requête POST à l'API avec les données JSON
        response = requests.post(url, json=data_json)

        # Vérifier que la requête a réussi (code 200)
        self.assertEqual(response.status_code, 200)

        # Vérifier la structure de la réponse (par exemple, si elle renvoie un JSON)
        try:
            response_json = response.json()
            self.assertIsInstance(response_json, list)  # Supposons que l'API renvoie une liste de résultats
        except ValueError:
            self.fail("La réponse n'est pas un JSON valide.")

if __name__ == '__main__':
    unittest.main()

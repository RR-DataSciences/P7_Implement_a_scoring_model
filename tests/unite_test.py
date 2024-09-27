import unittest
import sys
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

    # def test_predict():
    #     importer modèle + scaler
    #     Faire une prédiction

if __name__ == '__main__':
    unittest.main()

import unittest
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

    def test_train_models(self):
        # Créer un DataFrame de test simple
        df = pd.DataFrame({
            'feature1': [1, 2, 3, 4],
            'feature2': [5, 6, 7, 8],
            'TARGET': [0, 1, 1, 0]
        })

        X_train_rfe, X_test_rfe, y_train, y_test, rfe, model = train_models(df, "test_data")

        self.assertIsNotNone(X_train_rfe)
        self.assertIsNotNone(X_test_rfe)
        self.assertIsNotNone(y_train)
        self.assertIsNotNone(y_test)
        self.assertIsNotNone(rfe)
        self.assertIsNotNone(model)

if __name__ == '__main__':
    unittest.main()

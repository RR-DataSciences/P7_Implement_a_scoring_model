import re
import os
import joblib
import mlflow
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier
# from mlflow.models import infer_signature

# Function to calculate missing values by column# Funct
def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()

    # Percentage of missing values
    mis_val_percent = 100 * df.isnull().sum() / len(df)

    # Make a table with the results
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)

    # Rename the columns
    mis_val_table_ren_columns = mis_val_table.rename(
    columns = {0 : 'Missing Values', 1 : '% of Total Values'})

    # Sort the table by percentage of missing descending
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
    '% of Total Values', ascending=False).round(1)

    # Print some summary information
    print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")

    # Return the dataframe with missing information
    return mis_val_table_ren_columns
    
# Remplacer les caractères spéciaux par des caractères de soulignement - LGBM
def replace_special_chars(feature_name):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', feature_name)
    
def run_exists_by_partial_name(experiment_name, name):
  client = mlflow.tracking.MlflowClient()
  experiment = client.get_experiment_by_name(experiment_name)
  if experiment:
    runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"run_name LIKE '{name}%'")
    return len(runs) > 0
    
def custom_score(y_test, y_pred, X=None, sample_weight=None):
    
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))
    score = 10 * false_positives + false_negatives

    return score

def train_models(df, data_name, experiment_name):

    # Modèle XGB - Calculer le poids positif en fonction du déséquilibre de classe
    ratio = float(np.sum(df['TARGET'] == 0)) / np.sum(df['TARGET'] == 1)
    # Calculer les poids pour chaque classe dans la colonne 'TARGET'
    poids = (df['TARGET'].value_counts(normalize=True)*100).to_dict()
      
    # Defines the list of models
    models = {
        "LogisticRegression": LogisticRegression(),
        "LGBM": LGBMClassifier(class_weight='balanced'),
        "XGB": XGBClassifier(scale_pos_weight=ratio, class_weight=poids), # Initialiser le modèle XGBoost avec les poids de classe appropriés
        }
    # Define hyperparameters for each model
    param_grid = {
        "LogisticRegression": {
            'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            'estimator__class_weight': ['balanced'],
            'estimator__max_iter': [100, 200],
            'estimator__random_state': [42]
            },
        "LGBM": {
            'estimator__num_leaves': [31, 127],
            'estimator__max_depth': [-1, 5, 10],
            'estimator__learning_rate': [0.01, 0.1, 0.2],
            'estimator__n_estimators': [150],
            'estimator__class_weight': ['balanced'],
            'estimator__force_row_wise': [True],
            'estimator__random_state': [42]
            },
        "XGB": {
            'estimator__max_depth': [5, 7, 9],
            'estimator__learning_rate': [0.01, 0.1, 0.2, 0.3, 0.5],
            'estimator__n_estimators': [100],
            'estimator__scale_pos_weight': [ratio]
            },
        }
    # Boucle sur chaque modèle
    for model_name, model in models.items():
    
      # Construire le nom d'exécution avec le nombre d'individus, la date et l'heure actuelle
      nb_indiv = len(df)
      now = datetime.now().strftime("%y-%m-%d at %H:%M")
      name_prefix = f"{nb_indiv}_{data_name}_{model_name}-[{now}]"
    
      # Vérifier si un run avec le même nom existe déjà
      if not run_exists_by_partial_name(experiment_name, name_prefix.split('-')[0]):
        # Enregistrement du run dans MLflow
        with mlflow.start_run(run_name=name_prefix):
          # Définition de la stratégie de validation croisée
          cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
          # Appliquer la fonction à toutes les fonctionnalités
          df.columns = df.columns.map(replace_special_chars)
    
          y = df['TARGET']
          X = df.drop(columns='TARGET')
          # Division des données en ensemble d'entraînement et ensemble de test
          X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
          
          print(f"{'#'*40}\n[Dimension reduction - {model_name} in progress]")
          n_components = 0.99
          pca = PCA(n_components = n_components)
          X_train_acp = pca.fit_transform(X_train)
          X_test_pca = pca.transform(X_test)
          print(f'Dimension X_train: {X_train.shape}')
          print(f'Dimension après PCA: {X_train_acp.shape}')
          print(f'-'*10)
          print(f'Dimension X_test: {X_test.shape}')
          print(f'Dimension après PCA: {X_test_pca.shape}')
    
          # Création du scorer personnalisé pour le seuil actuel
          custom_scorer = make_scorer(custom_score, greater_is_better=False)
          print(f"{'#'*40}\n[TunedThresholdClassifierCV - {model_name} in progress]")
          
          thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
          model_threshold = TunedThresholdClassifierCV(model, scoring=custom_scorer, store_cv_results=True, thresholds=thresholds)
          model_threshold.fit(X_train_acp, y_train)
    
          # Initialisation de la recherche sur grille avec les paramètres spécifiques au modèle
          grid_search = GridSearchCV(estimator=model_threshold,
                                     param_grid=param_grid[model_name],
                                     cv=cv,
                                     scoring={'f1': 'f1', 'AUC': 'roc_auc', 'Unite_de_cout': custom_scorer},
                                     refit='Unite_de_cout',
                                     return_train_score=True,
                                     n_jobs=-1,
                                     verbose=3)
    
          # Affichage des meilleurs paramètres
          print(f"{'#'*40}\n[GridSearchCV - {model_name} in progress]")
          # Exécution de la recherche sur grille
          grid_search.fit(X_train_acp, y_train)
    
          ####------------------ DF RESULTS ------------------####
    
          # Instantiates the results of the best_model_threshold
          result_tuned_threshold = pd.DataFrame(model_threshold.cv_results_)
          result_tuned_threshold.to_csv('/content/drive/MyDrive/OC_Data_Scientist/P7_Implémentez_un_modèle_de_scoring/threshold.csv', sep=';')
          # Instantiates the results of the best model
          result = pd.DataFrame(grid_search.cv_results_)
          result.to_csv('/content/drive/MyDrive/OC_Data_Scientist/P7_Implémentez_un_modèle_de_scoring/result.csv', sep=';')
    
          ####---------------- RESULTS GRAPH ----------------####
    
          # Selects the best model result
          best_score = result.loc[result["rank_test_Unite_de_cout"] == 1][['mean_test_f1','mean_train_f1','mean_test_Unite_de_cout','mean_train_Unite_de_cout','mean_test_AUC','mean_train_AUC','params']].head(1)
          # Normalizes numerical results to 2 decimal places
          score_test_f1 = best_score['mean_test_f1'].values[0]
          score_train_f1 = best_score['mean_train_f1'].values[0]
          score_test_Unite_de_cout = best_score['mean_test_Unite_de_cout'].values[0]
          score_train_Unite_de_cout = best_score['mean_train_Unite_de_cout'].values[0]
          score_test_auc = best_score['mean_test_AUC'].values[0]
          score_train_auc = best_score['mean_train_AUC'].values[0]
    
          # Define the graph
          plt.figure(figsize=(10,6))
          # Displays test scores
          plt.plot(result["mean_test_Unite_de_cout"], color="#00317A")
          # Displays training results
          plt.plot(result["mean_train_Unite_de_cout"], color="#000")
          # Add a table with the model's best result data
          table = plt.table(cellText=best_score.drop(columns='params').values,
                          colLabels=best_score.drop(columns='params').columns,
                          loc='bottom',
                          bbox=[0, -0.5, 1, 0.35])
          # Graph formatting
          plt.title(f"{model_name} - Model Settings\n")
          plt.ylabel("mean_train_Unite_de_cout")
          plt.xlabel("Parameters used")
          # Enregistrement du graphique dans MLflow
          mlflow.log_figure(plt.gcf(), f"{model_name}_SettingHyperparameters.png")
          # plt.tight_layout()
          # plt.show()
    
          #################### BEST MODEL RESULTS ####################
    
          print(f"{'='*10}\nMeilleurs résultats obtenus par le modèle {model_name}")
          # Affiche les résultats des meilleurs paramètres
          print(f"{'-'*10}\n[Best Score (Unite_de_cout) -> {round(grid_search.best_score_,4)}]\n[Controle Unite_de_cout -> {round(grid_search.score(X_test_pca, y_test),4)}]")
          print(f"[Score f1 -> {round(score_test_f1,4)}]\n[Controle Score f1 -> {round(score_train_f1,4)}]")
          print(f"[Score AUC -> {round(score_test_auc,4)}]\n[Controle Score AUC -> {round(score_train_auc,4)}]")
    
          print(f"{'-'*10}\n[Best Score TunedThreshold (Unite_de_cout) -> {round(model_threshold.best_score_,4)}]\n[Controle TunedThreshold Unite_de_cout -> {round(model_threshold.score(X_test_pca, y_test),4)}]")
          print(f'TunedThreshold best seuil: {model_threshold.best_threshold_}')
    
          # Log des meilleurs paramètres dans MLflow
          mlflow.log_param("best_params", grid_search.best_params_)
          mlflow.log_param("X_train", X_train.shape)
          mlflow.log_param("X_train_acp", X_train_acp.shape)
          mlflow.log_param("X_test", X_test.shape)
          mlflow.log_param("X_test_pca", X_test_pca.shape)
    
          mlflow.log_metric("best_Unite_de_cout", round(grid_search.best_score_, 3))
    
          mlflow.log_metric("score_test_f1", round(score_test_f1*100, 3))
          mlflow.log_metric("score_train_auc", round(score_train_auc*100, 3))
    
          mlflow.log_metric("score_test_auc", round(score_test_auc*100, 3))
          mlflow.log_metric("score_train_auc", round(score_train_auc*100, 3))
    
          mlflow.log_metric("best_threshold_Unite_cout", round(model_threshold.best_score_,3))
          mlflow.log_metric("best_Threshold", round(model_threshold.best_threshold_, 3))
    
          # Enregistrement du meilleur modèle localement
          model_save_path = '/content/drive/MyDrive/OC_Data_Scientist/P7_Implémentez_un_modèle_de_scoring/models'
          os.makedirs(model_save_path, exist_ok=True)
          model_filename = f"{model_save_path}/{name_prefix}.pkl"
          joblib.dump(grid_search.best_estimator_, model_filename)
          print(f"Modèle enregistré localement sous {model_filename}")
          
          # Enregistrement local des données transformées par PCA
          pca_filename = f"/content/drive/MyDrive/OC_Data_Scientist/P7_Implémentez_un_modèle_de_scoring/pca_{name_prefix}.pkl"
          joblib.dump(pca, pca_filename)
          print(f"PCA enregistré localement sous {pca_filename}")
    
          print(f"{'-'*10}\n[Nom du run MLFlow] -> {name_prefix}\n")
    
      else:
          print(f"Un run avec le même nom existe déjà pour le modèle {model_name} dans MLflow!\n{'-'*10}")
          X_train = []
          X_test = []
          y_train = []
          y_test = []
          X_train_acp = []
          X_test_pca = []
          # Si la condition est fausse, afficher les noms des runs existants
          if run_exists_by_partial_name(experiment_name, name_prefix.split('-')[0]):
              client = mlflow.tracking.MlflowClient()
              experiment = client.get_experiment_by_name(experiment_name)
              if experiment:
                  runs = client.search_runs(experiment_ids=[experiment.experiment_id], filter_string=f"run_name LIKE '{name_prefix}-{model_name}%'")
                  for run in runs:
                      print(f'->', run.info.run_name)
                      
    return X_train_acp, X_test_pca, y_train, y_test

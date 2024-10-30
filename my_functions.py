import re
import os
import sys
import dill
import shap
import neptune
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, TunedThresholdClassifierCV
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
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
    
# Replace special characters with underscores - LGBM
def replace_special_chars(feature_name):
    return re.sub(r'[^a-zA-Z0-9_]+', '_', feature_name)

# Customised scoring function to assess model performance
def custom_score(y_test, y_pred, X=None, sample_weight=None):
    # Coût métier : coût d'un faux négatif est 10 fois supérieur au coût d'un faux positif
    fp_cost = 1
    fn_cost = 10
    
    # Calculer le nombre de faux positifs et faux négatifs
    false_positives = np.sum((y_pred == 1) & (y_test == 0))
    false_negatives = np.sum((y_pred == 0) & (y_test == 1))
    
    # Calculer le coût total
    score = fp_cost * false_positives + fn_cost * false_negatives
    
    return score

# Modelling function
def train_models(data, scaler, data_name):
    ####------------------ MODELE / HYPER-PARAMETRE ------------------####
    
    # Defines the list of models
    models = {
    #   "LogisticRegression": LogisticRegression(),
      "LGBM": LGBMClassifier(),
      }
    # Define hyperparameters for each model
    param_grid = {
    #   "LogisticRegression": {
    #       'estimator__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    #       'estimator__class_weight': ['balanced'],
    #       'estimator__max_iter': [100, 200],
    #       'estimator__random_state': [42]
    #   },
      "LGBM": {
        #   'estimator__num_leaves': [31, 127],
          'estimator__max_depth': [-1, 10],
          'estimator__learning_rate': [0.01, 0.1],
          'estimator__class_weight': ['balanced'],
          'estimator__force_row_wise': [True],
        #   'estimator__force_col_wise': [True],
          'estimator__random_state': [42]
      },
    }
  
    # Boucle sur chaque modèle
    for model_name, model in models.items():
    
        ####---------- NEPTUNE - INITIALISATION DU RUN ----------####
        
        experiment_name = f"{model_name}_{data_name}"
        run = neptune.init_run(
            project="remi.rogulski/OC-projet7",
            name=experiment_name,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI3OGVhZTE5ZS05Njc0LTQyYmUtODgxZS00ZTE2YTIyNTJlYzAifQ==",
            source_files=["my_functions.py"],
            capture_hardware_metrics=True,
            capture_stderr=True,
            capture_stdout=True,
            # mode="read-only",
        )
        
        ####----------  INSTANCIATION DU NAMING DES OBJETS ----------####
        
        data_name = data_name
        nb_indiv = len(data)
        now = datetime.now().strftime("%y-%m-%d at %H:%M")
        # Normalisation du nom des fichiers exportés
        name_prefix = f"{nb_indiv}_{data_name}_{model_name}-[{now}]"    
        
        ####------------------ PREPROCESSING ------------------####
        
        path_projet_7 = '/content/drive/MyDrive/OC_Data_Scientist/P7_Implémentez_un_modèle_de_scoring'
        
        # Enregistrement du scaler
        scaler_save_path = f'{path_projet_7}/models'
        os.makedirs(scaler_save_path, exist_ok=True)
        scaler_filename = f"{scaler_save_path}/scaler_{name_prefix}.dill"
        with open(scaler_filename, 'wb') as f:
            dill.dump(scaler, f)
        print(f"Scaler enregistré localement sous {scaler_filename}")
        
        # Enregistrement du scaler sur Neptune
        run[f"model/scaler_{model_name}_{data_name}"].upload(scaler_filename)
        run[f"description/scaler_{model_name}_{data_name}"] = 'Preprocessing utilisé sur les données pour l\'entrainement du modèle'
        
        # Normalise le nom des colonnes
        data.columns = data.columns.map(replace_special_chars)
        
        # Création du scorer personnalisé pour le seuil actuel
        custom_scorer = make_scorer(custom_score, greater_is_better=False)
        
        # Division des données en ensemble d'entraînement et ensemble de test
        y = data['TARGET']
        X = data.drop(columns='TARGET')
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        ####------------------------ DATA DRIFT ------------------------####
        
        # Analyse de la data drift
        column_mapping = ColumnMapping()
        column_mapping.target = 'TARGET'
        # Créer un rapport de data drift
        data_drift_report = Report(metrics=[DataDriftPreset()])
        data_drift_report.run(reference_data=X_train, current_data=X_test, column_mapping=column_mapping)
        
        # Enregistrement sur Neptune du rapport HTML
        drift_report_filename = f"{path_projet_7}/reports/data_drift_{name_prefix}.html"
        data_drift_report.save_html(drift_report_filename)
        run[f"rapport/data_drift_{data_name}"].upload(drift_report_filename)
        run[f"description/data_drift_{data_name}"] = f'Rapport HTML de l\'analyse de la data drift entre les jeux de données d\'entrainement et de test'
        # Afficher le rapport
        data_drift_report.show(mode='inline')
        
        ####------------------ RFE - FEATURE SELECTION ------------------####
        
        print(f"{'#'*40}\n[RFECV - {model_name} in progress]")
        # Appliquer RFECV pour la sélection des caractéristiques
        rfe = RFE(estimator=model, step=1, n_features_to_select=0.3, verbose=1, importance_getter='auto')
        rfe.fit(X_train, y_train)
        print(f"Modèle: {model_name}")
        print(f"Nombre de caractéristiques sélectionnées : {rfe.n_features_}")
        print(f"Caractéristiques sélectionnées : {X_train.columns[rfe.support_].tolist()}")
        # Réduire le jeu de données à ces caractéristiques
        X_train_rfe = rfe.transform(X_train)
        X_test_rfe = rfe.transform(X_test)
        
        # Enregistrement local de l'objet RFE
        rfe_filename = f"{path_projet_7}/models/rfe_{name_prefix}.dill"
        with open(rfe_filename, 'wb') as f:
            dill.dump(rfe, f)
        print(f"RFE enregistré localement sous {rfe_filename}")
        
        # Enregistrement de l'objet RFE sur Neptune
        run[f"model/rfe_{model_name}_{data_name}"].upload(rfe_filename)
        # run["model/saved_model"].track_files(rfe_filename)
        run[f"description/rfe_{model_name}_{data_name}"] = 'Methode RFE pour la réduction de dimension des données avant entrainement'
        
        # Enregistrement local des données transformées par RFE
        rfe_column_names = X_test.columns[rfe.support_].tolist()
        X_train_rfe_df = pd.DataFrame(X_train_rfe, columns=rfe_column_names)
        X_train_rfe_df.to_csv(f'{path_projet_7}/training_data/X_train_rfe.csv', sep=';', index=False, header=rfe_column_names)
        X_test_rfe_df = pd.DataFrame(X_test_rfe, columns=rfe_column_names)
        X_test_rfe_df.to_csv(f'{path_projet_7}/training_data/X_test_rfe.csv', sep=';', index=False, header=rfe_column_names)
        
        # Enregistrement sur Neptune des données transformées par RFE
        run[f"training_data/X_train_rfe_{model_name}_{data_name}"].upload(f'{path_projet_7}/training_data/X_train_rfe.csv')
        run[f"description/X_train_rfe_{model_name}_{data_name}"] = 'Données d\'entrainements post train_test_split'
        run[f"training_data/X_test_rfe_{model_name}_{data_name}"].upload(f'{path_projet_7}/training_data/X_test_rfe.csv')
        run[f"description/X_test_rfe_{model_name}_{data_name}"] = 'Données de tests post train_test_split'
        
        ####------------------ MODELISATION ------------------####
        
        # Optimisation du seuil de classification
        print(f"{'#'*40}\n[TunedThresholdClassifierCV - {model_name} in progress]")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        model_threshold = TunedThresholdClassifierCV(model, scoring=custom_scorer, store_cv_results=True, thresholds=thresholds)
        model_threshold.fit(X_train_rfe, y_train)
        
        # Choix du nombre de groupes pour la validation croisée
        cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
        # Initialisation de la recherche sur grille avec les paramètres spécifiques au modèle
        grid_search = GridSearchCV(estimator=model_threshold,
                                    param_grid=param_grid[model_name],
                                    cv=cv,
                                    scoring={'f1': 'f1', 'AUC': 'roc_auc', 'Unite_de_cout': custom_scorer},
                                    refit='Unite_de_cout',
                                    return_train_score=True,
                                    n_jobs=-1,
                                    verbose=3)
        
        # Exécution de la recherche sur grille
        print(f"{'#'*40}\n[GridSearchCV - {model_name} in progress]")
        grid_search.fit(X_train_rfe, y_train)
        model = grid_search.best_estimator_
        
        # Enregistrement local du meilleur modèle
        model_save_path = '/content/drive/MyDrive/OC_Data_Scientist/P7_Implémentez_un_modèle_de_scoring/models'
        os.makedirs(model_save_path, exist_ok=True)
        model_filename = f"{model_save_path}/model_{name_prefix}.dill"
        with open(model_filename, 'wb') as f:
            dill.dump(grid_search.best_estimator_, f)
        print(f"Modèle enregistré localement sous {model_filename}")
        
        # Enregistrement sur Neptune du meilleur modèle
        run[f"model/model_{model_name}_{data_name}"].upload(model_filename)
        # run["models/best_model"].track_files(model_filename)
        run[f"description/model_{model_name}_{data_name}"] = f'{name_prefix}.dill'
        
        # Enregistrement local des données issues du best_model_threshold
        result_tuned_threshold = pd.DataFrame(model_threshold.cv_results_)
        result_tuned_threshold.to_csv(f'{path_projet_7}/training_data/threshold.csv', sep=';')
        # Enregistrement sur Neptune des données issues du best_model_threshold
        run[f'training_data/threshold_{data_name}'].upload(f'{path_projet_7}/training_data/threshold.csv')
        run[f'description/threshold_{data_name}'] = 'Détail des résultats obtenus via best_model_threshold'
        
        # Enregistrement local des données issues du best_model
        result = pd.DataFrame(grid_search.cv_results_)
        result.to_csv(f'{path_projet_7}/training_data/result.csv', sep=';')
        # Enregistrement sur Neptune des données issues du best_model_threshold
        run[f'training_data/result_{data_name}'].upload(f'{path_projet_7}/training_data/result.csv')
        run[f'description/result_{data_name}'] = 'Détail des résultats obtenus via best_model'
        
        # Enregistrement des meilleurs hyperparamètres sur Neptune
        best_params = grid_search.best_params_
        run['model/best_params'] = best_params
        # Afficher les meilleurs hyperparamètres pour validation
        print(f"Meilleurs hyperparamètres pour {model_name} : {best_params}")
        # Enregistrement sur Neptune
        run[f'description/best_params'] = 'Meilleurs hyperparamètres obtenus pour ce run'
        
        ####---------------- RESULTS GRAPH ----------------####
        
        # Selects the best model result
        best_score = result.loc[result["rank_test_Unite_de_cout"] == 1][['mean_test_f1','mean_train_f1','mean_test_Unite_de_cout','mean_train_Unite_de_cout','mean_test_AUC','mean_train_AUC','params']].head(1)
        
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
        plt.title(f"{model_name} - Model Settings\n")
        plt.ylabel("mean_train_Unite_de_cout")
        plt.xlabel("Parameters used")
        plt.tight_layout()
        
        # Sauvegarder l'image localement
        graph_filename = f"{path_projet_7}/plots/{nb_indiv}_{data_name}_{model_name}_settings_graph.png"
        plt.savefig(graph_filename)
        # Enregistrer l'image dans Neptune
        run[f"plots/model_settings_{data_name}"].upload(graph_filename)
        
        plt.show()
        
        #################### BEST MODEL RESULTS ####################
        
        # Normalizes numerical results to 2 decimal places
        score_test_f1 = best_score['mean_test_f1'].values[0]
        score_train_f1 = best_score['mean_train_f1'].values[0]
        score_test_Unite_de_cout = best_score['mean_test_Unite_de_cout'].values[0]
        score_train_Unite_de_cout = best_score['mean_train_Unite_de_cout'].values[0]
        score_test_auc = best_score['mean_test_AUC'].values[0]
        score_train_auc = best_score['mean_train_AUC'].values[0]
        
        print(f"{'='*10}\nMeilleurs résultats obtenus par le modèle {model_name}")
        # Affiche les résultats des meilleurs paramètres
        print(f"{'-'*10}\n[Best Score (Unite_de_cout) -> {round(grid_search.best_score_,4)}]\n[Controle Unite_de_cout -> {round(grid_search.score(X_test_rfe, y_test),4)}]")
        print(f"[Score f1 -> {round(score_test_f1,4)}]\n[Controle Score f1 -> {round(score_train_f1,4)}]")
        print(f"[Score AUC -> {round(score_test_auc,4)}]\n[Controle Score AUC -> {round(score_train_auc,4)}]")
        
        print(f"{'-'*10}\n[Best Score TunedThreshold (Unite_de_cout) -> {round(model_threshold.best_score_,4)}]\n[Controle TunedThreshold Unite_de_cout -> {round(model_threshold.score(X_test_rfe, y_test),4)}]")
        print(f'TunedThreshold best seuil: {model_threshold.best_threshold_}')
        
        # Enregistrer les métriques
        run["metrics/best_Unite_de_cout"] = round(grid_search.best_score_, 3)
        
        run["metrics/score_test_f1"] = round(score_test_f1*100, 3)
        run["metrics/score_train_f1"] = round(score_train_f1*100, 3)
        
        run["metrics/score_test_auc"] = round(score_test_auc*100, 3)
        run["metrics/score_train_auc"] = round(score_train_auc*100, 3)
        
        run["metrics/best_threshold_Unite_cout"] = round(model_threshold.best_score_,3)
        run["metrics/best_Threshold"] = round(model_threshold.best_threshold_, 3)
        
        #################### SHAP ####################
        
        # Création d'un objet Explainer et calcul des valeurs SHAP
        explainer = shap.Explainer(model.estimator_, X_train_rfe_df, feature_names=rfe_column_names, feature_perturbation="interventional", approximate=True)
        shap_values = explainer(X_test_rfe_df)
        
        # Exporter l'objet explainer avec dill
        explainer_path = f'{path_projet_7}/models/explainer_{model_name}_{data_name}.dill'
        with open(explainer_path, 'wb') as f:
            dill.dump(explainer, f)
            
        run[f'model/explainer_{model_name}_{data_name}'].upload(explainer_path)
        run[f'description/explainer_{model_name}_{data_name}'] = f'explainer_{model_name}_{data_name}.dill'
        
        shap.summary_plot(shap_values, X_test_rfe_df, max_display=15, show=False)
        fig = plt.gcf()
        fig.set_size_inches(10, 8)
        plt.title("Distribution des valeurs shap et relation avec les valeurs réelles des caractéristiques")
        plt.tight_layout()
        # Sauvegarder l'image localement
        graph_filename = f"{path_projet_7}/plots/{nb_indiv}_{data_name}_{model_name}_summary_plot.png"
        plt.savefig(graph_filename, bbox_inches='tight')
        # Enregistrer l'image dans Neptune
        run[f"plots/summary_plot_{data_name}"].upload(graph_filename)
        plt.show()
        plt.clf()
        
        indiv = 4
        
        # Plot the SHAP waterfall
        shap.plots.waterfall(shap_values[indiv], show=False, max_display=15)
        fig = plt.gcf()
        fig.set_size_inches(18, 8)
        plt.title(f"Individu {indiv} - Contribution des caractéristiques à la prédiction du modèle")
        plt.tight_layout()
        # Sauvegarder l'image localement
        graph_filename = f"{path_projet_7}/plots/{nb_indiv}_{data_name}_{model_name}_waterfall.png"
        plt.savefig(graph_filename, bbox_inches='tight')
        # Enregistrer l'image dans Neptune
        run[f"plots/waterfall_{data_name}"].upload(graph_filename)
        plt.show()
        plt.clf()
        
        shap.plots.bar(shap_values[indiv], show=False)
        fig = plt.gcf()
        fig.set_size_inches(16, 8)
        plt.title(f"Individu {indiv} - Contribution des caractéristiques pour la prédiction finale")
        plt.tight_layout()
        # Sauvegarder l'image localement
        graph_filename = f"{path_projet_7}/plots/{nb_indiv}_{data_name}_{model_name}_bar.png"
        plt.savefig(graph_filename, bbox_inches='tight')
        # Enregistrer l'image dans Neptune
        run[f"plots/bar_{data_name}"].upload(graph_filename)
        plt.show()
        plt.clf()
        
        shap.plots.force(shap_values[indiv],
                        feature_names=rfe_column_names,
                        out_names='TARGET',
                        matplotlib=True,
                        text_rotation=25, show=False)
        fig = plt.gcf()
        fig.set_size_inches(25, 6)
        plt.title(f"Individu {indiv} - Contribution de la prédiction finale via disposition de force additive")
        plt.tight_layout()
        # Sauvegarder l'image localement
        graph_filename = f"{path_projet_7}/plots/{nb_indiv}_{data_name}_{model_name}_force.png"
        plt.savefig(graph_filename, bbox_inches='tight')
        # Enregistrer l'image dans Neptune
        run[f"plots/force_{data_name}"].upload(graph_filename)
        plt.show()
        plt.clf()
        
        run.stop()
    return X_train_rfe, X_test_rfe, y_train, y_test, scaler, rfe, model_threshold, model, explainer
# PRET A DEPENSER
<img src="API/static/Logo_GPT.jpg" alt="Prêt à dépenser" width="200">

## [Projet 7] - Implémentez un modèle de scoring

### Objectif du projet
Le projet Prêt à dépenser vise à développer un outil de scoring de crédit pour l'entreprise de prêt bancaire Prêt à dépenser. Cet outil permet de prédire la probabilité de remboursement d'un prêt par un client potentiel, ainsi que de visualiser et d'expliquer les décisions de crédit à l'aide de valeurs SHAP (SHapley Additive exPlanations). Le projet comprend un Dashboard interactif pour simuler des scénarios d'attribution de prêt, ainsi qu'une API pour effectuer des prédictions en temps réel.

### Structure des dossiers

* :file_folder: **API**
  * :memo: **app.py** - Script principal pour le déploiement de l'API Flask.
  * :memo: **my_functions.py** - Regroupe les fonctions utilitaires nécessaires au prétraitement des données et à la gestion des modèles.
  * :memo: **static/** - Stockez les images de fond et les éléments graphiques utilisés pour la page d'accueil de l'API.
* :file_folder: **dashboard**
  * :memo: **dashboard.py** - Fichier principal pour l'exécution de l'application Streamlit.
  * :memo: **style.css** - Fichier CSS pour la personnalisation de l'interface utilisateur.
  * :file_folder: **images/** - Stockez les logos et autres images utilisés dans le Dashboard.
* :file_folder: **models**
* :file_folder: **test**
  * :memo: **test_data.csv** - Contient les données nécessaires pour tester l'application.
  * :memo: **unite_test.py** - Regroupe les fonctions de tests unitaires appliquées lors du déploiement.
* :memo: **modelling.ipynb** - Notebook de modélisation.
* :memo: **rapport_data_drift.html** - Rapport d'analyse du drift sur les données app_train et app test.
* :memo: **requirements.txt** - Liste des dépendances nécessaires pour exécuter le projet.
* :memo: **run_API.sh** - Fichier permettant le lancement de l'API.
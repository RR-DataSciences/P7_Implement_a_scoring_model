#!/bin/bash

# Définir la variable d'environnement FLASK_APP
# export FLASK_APP=~/P7_Implement_a_scoring_model/API/app.py

# Lancer l'application Flask
# flask run --host=0.0.0.0

# Lancer l'application Flask avec gunicorn
cd ~/P7_Implement_a_scoring_model/API/ || exit 1

nohup gunicorn --bind 0.0.0.0:5000 app:app


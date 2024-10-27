#!/bin/bash
cd ~/P7_Implement_a_scoring_model/API/ || exit 1
# Lancer l'application Flask avec gunicorn
nohup gunicorn --bind 0.0.0.0:5000 app:app


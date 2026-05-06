import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. On charge le modèle gagnant et le scaler
model = joblib.load('best_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏦 Analyse de Risque de Crédit")

# 2. On crée les cases pour que l'utilisateur remplisse les infos
age = st.slider("Âge", 18, 90, 35)
income = st.number_input("Revenu Mensuel ($)", value=5000)
util = st.slider("Utilisation du crédit (0.0 à 1.0)", 0.0, 1.0, 0.3)
debt = st.number_input("Ratio d'endettement", value=0.3)
# ...ajoute les autres variables si nécessaire...

# 3. Le bouton pour prédire
if st.button("Est-ce que je peux prêter ?"):
    # On met les données dans le bon ordre pour le modèle
    # (Note: il faut mettre toutes les colonnes utilisées par le modèle)
    entree = np.array([[util, age, 0, debt, income, 0, 0, 0, 0, 0]])
    
    # On applique le scaler
    entree_scaled = scaler.transform(entree)
    
    # On prédit
    prediction = model.predict(entree_scaled)
    
    if prediction[0] == 0:
        st.success("✅ CRÉDIT ACCORDÉ : Profil fiable.")
    else:
        st.error("❌ CRÉDIT REFUSÉ : Risque de défaut élevé.")

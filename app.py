import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Config
st.set_page_config(page_title="Analyse Crédit", layout="wide")

# 2. Chargement (Vérifie bien les noms des fichiers sur GitHub !)
model = joblib.load('best_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

st.title("🏦 Système de Scoring de Crédit")

# 3. Formulaire
with st.form("my_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        # NOTE : Les noms à gauche (util, age, deps) doivent être identiques plus bas
        util = st.slider("Utilisation du crédit", 0.0, 1.0, 0.3)
        age = st.slider("Âge", 18, 95, 40)
        deps = st.number_input("Personnes à charge", 0, 10, 0)
        
    with col2:
        income = st.number_input("Revenu Mensuel ($)", value=5000)
        debt = st.number_input("Ratio d'endettement", value=0.35)

    # Le bouton DOIT être à l'intérieur du bloc 'with st.form'
    submit_button = st.form_submit_button("Lancer l'analyse")

# 4. Action après clic
if submit_button:
    # On crée le tableau avec les variables définies juste au-dessus
    # Si 'util' n'est pas défini au-dessus, le programme plante (NameError)
    data = np.array([[util, age, 0, debt, income, 0, 0, 0, 0, deps]])
    
    # Transformation et Prédiction
    data_scaled = scaler.transform(data)
    prob = model.predict_proba(data_scaled)[0][1]
    
    # Affichage
    if prob < 0.5:
        st.success(f"✅ CRÉDIT ACCORDÉ (Risque : {prob:.2%})")
    else:
        st.error(f"❌ CRÉDIT REFUSÉ (Risque : {prob:.2%})")

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Configuration de la page (à mettre tout en haut)
st.set_page_config(page_title="Credit Score AI", page_icon="🏦", layout="wide")

# Chargement du modèle et scaler
model = joblib.load('best_model_rf.pkl')
scaler = joblib.load('scaler.pkl')

# --- STYLE CSS PERSONNALISÉ ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 20px; height: 3em; background-color: #007bff; color: white; font-weight: bold; }
    .result-box { padding: 20px; border-radius: 15px; text-align: center; font-size: 24px; font-weight: bold; }
    </style>
    """, unsafe_allow_stdio=True)

st.title("🏦 Intelligence Artificielle : Analyse de Crédit")
st.markdown("---")

# --- LAYOUT : 2 COLONNES ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("👤 Profil du Client")
    age = st.slider("Âge du demandeur", 18, 95, 40)
    deps = st.number_input("Nombre de personnes à charge", 0, 10, 0)
    
with col2:
    st.subheader("💰 Situation Financière")
    income = st.number_input("Revenu Mensuel ($)", min_value=0, value=5000)
    util = st.slider("Utilisation du crédit", 0.0, 1.0, 0.3)
    debt = st.number_input("Ratio d'endettement (Debt Ratio)", value=0.35)

st.markdown("---")

# --- LOGIQUE DE PRÉDICTION ---
if st.button("🚀 LANCER L'ANALYSE DU RISQUE"):
    # Construction du tableau de données (vérifie bien l'ordre de tes 10 colonnes ici !)
    # Exemple pour les 10 colonnes typiques du dataset :
    features = np.array([[util, age, 0, debt, income, 0, 0, 0, 0, deps]])
    features_scaled = scaler.transform(features)
    
    # Calcul de probabilité
    prob = model.predict_proba(features_scaled)[0][1]
    prediction = model.predict(features_scaled)

    # --- AFFICHAGE DU RÉSULTAT AMÉLIORÉ ---
    st.subheader("📊 Résultats de l'analyse")
    
    # Barre de progression pour le risque
    st.write(f"Probabilité de défaut : **{prob:.2%}**")
    st.progress(prob)

    if prob < 0.3:
        st.markdown('<div class="result-box" style="background-color: #d4edda; color: #155724;">✅ CRÉDIT APPROUVÉ : Profil à faible risque</div>', unsafe_allow_html=True)
        st.balloons() # Petit effet de fête
    elif prob < 0.6:
        st.markdown('<div class="result-box" style="background-color: #fff3cd; color: #856404;">⚠️ DOSSIER À SURVEILLER : Risque modéré</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="result-box" style="background-color: #f8d7da; color: #721c24;">❌ CRÉDIT REFUSÉ : Risque trop élevé</div>', unsafe_allow_html=True)

# --- SIDEBAR ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2830/2830284.png", width=100)
st.sidebar.info("Ce modèle utilise un algorithme de **Random Forest** pour prédire la probabilité de défaut de paiement avec une précision de 84%.")

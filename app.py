import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from fpdf import FPDF

# Config page
st.set_page_config(page_title="❤️ Prédiction Maladies Cardiaques", layout="centered")

# Charger modèle et scaler
with open('logistic_model.pkl', 'rb') as f:
    data = pickle.load(f)
model = data['model']
scaler = data['scaler']
columns = data['columns']

# Initialiser historique
if 'history' not in st.session_state:
    st.session_state['history'] = []

# Moyenne population fictive (à ajuster selon ton dataset réel)
population_means = {
    'age': 50, 'cigsPerDay': 10, 'totChol': 220, 'sysBP': 130,
    'diaBP': 80, 'BMI': 26, 'heartRate': 75, 'glucose': 100
}

# Logo
if os.path.exists("logo.png"):
    st.image("logo.png", width=150)

st.title("Prédiction des Maladies Cardiaques")
st.markdown("Remplissez vos données, puis cliquez sur **Prédire**.")

# Interface utilisateur avec explications
user_input = {}
for col in columns:
    explain = ""
    if col == 'male':
        choice = st.selectbox('Sexe (Homme/Femme)', ['Homme', 'Femme'])
        user_input[col] = 1 if choice == 'Homme' else 0
    elif col == 'currentSmoker':
        choice = st.selectbox('Fumeur actuel ? (Oui/Non)', ['Oui', 'Non'])
        user_input[col] = 1 if choice == 'Oui' else 0
    elif col == 'BPMeds':
        choice = st.selectbox('Traitement antihypertenseur ?', ['Oui', 'Non'])
        user_input[col] = 1 if choice == 'Oui' else 0
    elif col == 'prevalentStroke':
        choice = st.selectbox('Antécédent d’AVC ?', ['Oui', 'Non'])
        user_input[col] = 1 if choice == 'Oui' else 0
    elif col == 'prevalentHyp':
        choice = st.selectbox('Hypertension préexistante ?', ['Oui', 'Non'])
        user_input[col] = 1 if choice == 'Oui' else 0
    elif col == 'diabetes':
        choice = st.selectbox('Diabète ?', ['Oui', 'Non'])
        user_input[col] = 1 if choice == 'Oui' else 0
    elif col == 'education':
        user_input[col] = st.selectbox('Éducation (1-4)', [1, 2, 3, 4])
    elif col == 'age':
        user_input[col] = st.slider('Âge (années)', 20, 100, 50, help='Âge en années')
    elif col == 'cigsPerDay':
        user_input[col] = st.slider('Cigarettes/jour', 0, 50, 0, help='Nombre moyen de cigarettes par jour')
    elif col == 'totChol':
        user_input[col] = st.slider('Cholestérol total (mg/dL)', 100, 400, 200)
    elif col == 'sysBP':
        user_input[col] = st.slider('Pression systolique (mmHg)', 80, 200, 120, help='Pression artérielle max')
    elif col == 'diaBP':
        user_input[col] = st.slider('Pression diastolique (mmHg)', 50, 150, 80, help='Pression artérielle min')
    elif col == 'BMI':
        user_input[col] = st.slider('IMC (kg/m²)', 10, 50, 25)
    elif col == 'heartRate':
        user_input[col] = st.slider('Fréquence cardiaque (bpm)', 40, 180, 70, help='BPM au repos')
    elif col == 'glucose':
        user_input[col] = st.slider('Glycémie (mg/dL)', 50, 300, 90, help='Taux de sucre sanguin')
    else:
        user_input[col] = st.number_input(col, value=0.0)

# Prédire
if st.button("Prédire"):
    input_df = pd.DataFrame([user_input])
    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[:, 1][0]
    prediction = int(proba >= 0.5)

    # Historique
    st.session_state['history'].append({'probabilité': proba, **user_input})

    st.subheader("🩺 Résultat")
    st.write(f"**Probabilité estimée : {proba*100:.1f}%**")
    if prediction:
        st.error("⚠️ Risque élevé détecté.")
    else:
        st.success("✅ Faible risque détecté.")

    # Prévention
    st.subheader("Conseils de prévention")
    alerts = []
    if user_input['cigsPerDay'] > 10:
        alerts.append("Réduire le tabac")
    if user_input['totChol'] > 240:
        alerts.append("Baisser le cholestérol")
    if user_input['sysBP'] > 140 or user_input['diaBP'] > 90:
        alerts.append("Surveiller la tension")
    if user_input['BMI'] > 30:
        alerts.append("Réduire l'IMC")
    if user_input['glucose'] > 140:
        alerts.append("Contrôler la glycémie")
    st.write(alerts if alerts else "Pas d’alerte particulière")

    # Graphique radar
    st.subheader("Comparaison à la moyenne")
    radar_vars = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'diaBP', 'BMI', 'heartRate', 'glucose']
    user_values = [user_input[v] for v in radar_vars]
    pop_values = [population_means[v] for v in radar_vars]
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))
    angles = np.linspace(0, 2*np.pi, len(radar_vars), endpoint=False)
    user_values += user_values[:1]
    pop_values += pop_values[:1]
    angles = np.concatenate((angles, [angles[0]]))
    ax.plot(angles, user_values, label='Vous')
    ax.plot(angles, pop_values, label='Moyenne')
    ax.fill(angles, user_values, alpha=0.25)
    ax.set_thetagrids(angles[:-1]*180/np.pi, radar_vars)
    ax.legend()
    st.pyplot(fig)

    # Historique
    st.subheader("🗂️ Historique de vos prédictions")
    hist_df = pd.DataFrame(st.session_state['history'])
    st.dataframe(hist_df)

    # PDF rapport
    st.subheader("📥 Rapport PDF")
    if st.button("Télécharger mon rapport"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Rapport de Prédiction Cardiaque", ln=True, align='C')
        pdf.ln(10)
        for key, value in user_input.items():
            pdf.cell(0, 10, f"{key}: {value}", ln=True)
        pdf.cell(0, 10, f"Probabilité: {proba*100:.1f}%", ln=True)
        pdf.cell(0, 10, f"Risque: {'Élevé' if prediction else 'Faible'}", ln=True)
        pdf.ln(10)
        pdf.cell(0, 10, "Conseils:", ln=True)
        for alert in alerts:
            pdf.cell(0, 10, f"- {alert}", ln=True)
        buffer = BytesIO()
        pdf.output(buffer)
        st.download_button(label="📄 Télécharger le rapport",
                           data=buffer.getvalue(),
                           file_name="rapport_cardiaque.pdf",
                           mime='application/pdf')


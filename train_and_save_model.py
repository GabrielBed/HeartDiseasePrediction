import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# Chargement des données
df = pd.read_csv("heart_disease.csv")
df_clean = df.dropna()
X = df_clean.drop('TenYearCHD', axis=1)
y = df_clean['TenYearCHD']

# Standardisation
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# GridSearch LogisticRegression
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
grid = GridSearchCV(LogisticRegression(class_weight='balanced', random_state=42, max_iter=500),
                    param_grid, cv=5, scoring='f1', n_jobs=-1)
grid.fit(X_train_res, y_train_res)

best_model = grid.best_estimator_

# Sauvegarder le modèle et le scaler
with open('logistic_model.pkl', 'wb') as f:
    pickle.dump({'model': best_model, 'scaler': scaler, 'columns': X.columns.tolist()}, f)

print("✅ Modèle et scaler sauvegardés dans logistic_model.pkl")

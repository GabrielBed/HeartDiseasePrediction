import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE

# Chargement des données
df = pd.read_csv("heart_disease.csv")

# Suppression des lignes avec valeurs manquantes
df_clean = df.dropna()
print(f"Lignes après suppression des valeurs manquantes : {df_clean.shape[0]}")

# Séparation X / y
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

# Modèles et grilles
models_params = {
    'LogisticRegression': (LogisticRegression(class_weight='balanced', random_state=42, max_iter=500),
                           {'C': [0.001, 0.01, 0.1, 1, 10, 100],
                            'penalty': ['l1', 'l2'],
                            'solver': ['liblinear']}),

    'RandomForest': (RandomForestClassifier(class_weight='balanced', random_state=42),
                     {'n_estimators': [100, 200, 500],
                      'max_depth': [3, 5, 10, None],
                      'min_samples_split': [2, 5, 10],
                      'min_samples_leaf': [1, 2, 4]}),
}

# Seuils à tester
thresholds = [0.3, 0.5, 0.6,0.7]

# Résultats stockés
results = []

for name, (model, params) in models_params.items():
    print(f"\n🔍 Optimisation de {name} avec GridSearchCV...")
    grid = GridSearchCV(model, params, cv=5, scoring='f1', n_jobs=-1)
    grid.fit(X_train_res, y_train_res)
    best_estimator = grid.best_estimator_
    y_proba = best_estimator.predict_proba(X_test)[:, 1]

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        results.append({
            'Model': name,
            'Threshold': threshold,
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1': f1
        })

# Convertir en DataFrame
results_df = pd.DataFrame(results)

# Afficher tableau récapitulatif
print("\nRésultats résumé :")
print(results_df)

# Tracer le F1-score par seuil et modèle
plt.figure(figsize=(10, 6))
sns.lineplot(data=results_df, x='Threshold', y='F1', hue='Model', marker='o')
plt.title('F1-score par seuil et modèle')
plt.show()

# Trouver le meilleur seuil par modèle
best_models = results_df.groupby('Model')['F1'].idxmax()
for idx in best_models:
    row = results_df.loc[idx]
    print(f"\n=== {row['Model']} meilleur seuil {row['Threshold']} ===")
    best_estimator = GridSearchCV(models_params[row['Model']][0],
                                  models_params[row['Model']][1],
                                  cv=5, scoring='f1', n_jobs=-1).fit(X_train_res, y_train_res).best_estimator_
    y_proba = best_estimator.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= row['Threshold']).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{row["Model"]} Confusion Matrix (Threshold {row["Threshold"]})')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()



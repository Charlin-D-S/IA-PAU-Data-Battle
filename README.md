# 🏆 Data Battle IA PAU 2026 – Système d'alerte météorologique foudre

## 👥 Équipe
- **Nom de l'équipe :**
- **Membres :**
  - Charlin DJIOKO
  - Pierrette Josiane MAKAMWE
  - Cynthia FANTA TCHAKOUNTE

---

## 🎯 Problématique

Les aéroports doivent suspendre toute activité au sol dès qu'un éclair CG (nuage-sol) est détecté à moins de 20 km. Le système opérationnel actuel (**baseline**) applique une règle fixe : déclencher une alerte de **30 minutes** à chaque éclair CG ≤ 20 km détecté.

Cette approche est **sûre mais coûteuse** : elle génère de longues interruptions même lorsque le danger est passé. L'enjeu est double :

- **Sécurité** : ne manquer aucun éclair dangereux (maximiser le POD)
- **Opérationnel** : minimiser la durée totale des alertes pour réduire les interruptions d'activité

---

## 💡 Solution proposée

Nous remplaçons la règle fixe de 30 minutes par un **système d'alerte adaptatif** basé sur 4 modèles XGBoost entraînés sur 7 ans de données (507 000+ éclairs, 6 aéroports).

**Architecture en deux niveaux :**

1. **Modèles prédictifs** — 4 classifieurs XGBoost prédisent la probabilité qu'un éclair CG survienne dans les prochaines 10, 15 ou 30 minutes (et dans un rayon de 3 km). Un seuil de décision est optimisé par validation croisée temporelle.

2. **Système de décision** — Un arbre de règles combine les 4 probabilités pour décider en temps réel de déclencher, prolonger ou lever une alerte :
   - Règle absolue sur les CG détectés (sécurité non négociable)
   - Anticipation : alerte possible *avant* le premier éclair
   - Levée rapide : si les modèles prédisent une faible activité future

**Résultats sur le jeu de test :**

| Métrique | Notre modèle | Baseline 30 min |
|---|---|---|
| POD (couverture CG20) | **94.9%** | 94.0% |
| FAR (fausses alertes) | **31.2%** | 35.1% |
| CSI | **0.925** | 0.921 |
| Durée totale alertes | **687 h** | 1 077 h |
| **Gain durée** | **−36%** | — |

---

## ⚙️ Stack technique

- **Langages :** Python 3.10+
- **Frameworks :** Streamlit (interface web), scikit-learn (preprocessing)
- **Outils :** Jupyter Notebook, joblib, pandas, numpy, matplotlib, plotly
- **IA :** XGBoost (gradient boosting, 4 modèles), optimisation bayésienne des hyperparamètres
- **Autre :** codecarbon (suivi de la consommation énergétique)

---

## 🚀 Installation & exécution

### Prérequis

- Python 3.10+
- pip

### Installation

```bash
git clone https://github.com/Charlin-D-S/IA-PAU-Data-Battle
cd IA-PAU-Data-Battle
pip install -r requirements.txt
```

### Exécution — Application web

```bash
streamlit run app.py
```

L'interface permet de charger un fichier CSV de données d'éclairs, de sélectionner le modèle et le seuil, et de visualiser les prédictions de risque par aéroport.

### Exécution — Notebooks (ordre recommandé)

```
1. exploration_features/data_exploration.ipynb       # EDA initiale
2. exploration_features/eda_meteorage.ipynb          # Analyse détaillée
3. exploration_features/feature_engineering.ipynb    # Développement des features

4. entrainement_models/t15_xgboost.ipynb             # Modèle principal 15 min
5. entrainement_models/t10_xgboost_bis.ipynb         # Modèle 10 min
6. entrainement_models/t30_xgboost.ipynb             # Modèle 30 min
7. entrainement_models/cg_3_t15_xgboost.ipynb        # Modèle 3 km

8. dataset_test/threshold_cv_optimization.ipynb      # Optimisation du seuil
9. dataset_test/Evaluation_databattle_meteorage.ipynb # Évaluation finale
10. dataset_test/rapport_approche_modele.ipynb        # Rapport complet
```

---

## 📁 Structure du projet

```
IA-PAU-Data-Battle/
├── data/                        # Jeux de données (voir data/README.md)
├── models/                      # Modèles entraînés .pkl (voir models/README.md)
├── src/                         # Feature engineering (voir src/README.md)
├── exploration_features/        # EDA et développement features
├── entrainement_models/         # Entraînement des 4 modèles XGBoost
├── dataset_test/                # Évaluation, optimisation du seuil, rapport
├── app.py                       # Application web Streamlit
└── requirements.txt
```

> Chaque sous-dossier contient son propre `README.md` détaillé.

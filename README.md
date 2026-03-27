# IA PAU Data Battle 2026 — Système d'alerte météorologique par apprentissage automatique

> Compétition Météorage — Prédiction du risque foudre à proximité des aéroports

---

## Table des matières

1. [Contexte et objectif](#1-contexte-et-objectif)
2. [Structure du projet](#2-structure-du-projet)
3. [Données](#3-données)
4. [Feature Engineering](#4-feature-engineering)
5. [Modèles](#5-modèles)
6. [Système d'alerte](#6-système-dalerte)
7. [Évaluation et comparaison](#7-évaluation-et-comparaison)
8. [Application Streamlit](#8-application-streamlit)
9. [Installation et utilisation](#9-installation-et-utilisation)
10. [Résultats](#10-résultats)

---

## 1. Contexte et objectif

Les aéroports doivent surveiller en permanence l'activité orageuse à proximité de leurs pistes. Toute opération est suspendue dès qu'un éclair CG (nuage-sol) est détecté à moins de 20 km. L'enjeu est double :

- **Sécurité** : ne manquer aucun éclair dangereux
- **Opérationnel** : minimiser la durée des alertes pour limiter les interruptions d'activité

Le système de référence actuel (**baseline**) déclenche une alerte fixe de **30 minutes** à chaque éclair CG ≤ 20 km. Notre approche remplace cette règle fixe par des **modèles prédictifs adaptatifs** capables d'anticiper la fin d'un épisode orageux.

**Données couvertes :** 695 000+ éclairs sur 6 aéroports (Ajaccio, Bastia, Bron, Nantes, Pise, Biarritz) de 2016 à 2023.

---

## 2. Structure du projet

```
IA-PAU-Data-Battle/
│
├── data/                                          # Jeux de données
│   ├── segment_alerts_all_airports_train.csv      # Entraînement (507 071 lignes)
│   ├── segment_alerts_all_airports_eval.csv       # Évaluation (188 175 lignes)
│   └── variables_description.csv                  # Dictionnaire des variables (99 variables)
│
├── models/                                        # Modèles entraînés (joblib)
│   ├── xgb_cg10_artefacts.pkl                    # Horizon 10 min, CG ≤ 20 km
│   ├── xgb_cg15_artefacts.pkl                    # Horizon 15 min, CG ≤ 20 km
│   ├── xgb_cg30_artefacts.pkl                    # Horizon 30 min, CG ≤ 20 km
│   └── xgb_cg15_3km_artefacts.pkl                # Horizon 15 min, CG ≤ 3 km
│
├── src/                                           # Code source Python
│   ├── feature_builder.py                         # Pipeline de feature engineering (278 lignes)
│   └── feature_engineering_function.py            # Variante avec cibles étendues (409 lignes)
│
├── exploration_features/                          # Exploration et développement
│   ├── data_exploration.ipynb                     # EDA initiale
│   ├── eda_meteorage.ipynb                        # Analyse statistique détaillée
│   └── feature_engineering.ipynb                  # Développement des features
│
├── entrainement_models/                           # Entraînement des modèles
│   ├── t10_xgboost_bis.ipynb                      # XGBoost horizon 10 min
│   ├── t15_xgboost.ipynb                          # XGBoost horizon 15 min (modèle principal)
│   ├── t30_xgboost.ipynb                          # XGBoost horizon 30 min
│   └── cg_3_t15_xgboost.ipynb                     # XGBoost 15 min rayon 3 km
│
├── dataset_test/                                  # Évaluation et optimisation
│   ├── Evaluation_databattle_meteorage.ipynb      # Évaluation finale complète
│   ├── threshold_cv_optimization.ipynb            # Optimisation du seuil par CV temporelle
│   └── rapport_approche_modele.ipynb              # Rapport complet de l'approche
│
├── app.py                                         # Application web Streamlit
├── requirements.txt                               # Dépendances Python
└── .streamlit/config.toml                         # Configuration UI Streamlit
```

---

## 3. Données

### Variables brutes

| Variable | Type | Description |
|---|---|---|
| `date` | datetime (UTC) | Horodatage de l'éclair |
| `airport` | str | Code de l'aéroport (6 valeurs) |
| `dist` | float | Distance à l'aéroport (km) |
| `azimuth` | float | Angle par rapport à l'aéroport (degrés) |
| `amplitude` | float | Amplitude électrique (kA) |
| `icloud` | bool | `True` = intra-nuage, `False` = nuage-sol (CG) |
| `airport_alert_id` | float | Identifiant d'épisode d'alerte |

### Variable cible

| Cible | Description |
|---|---|
| `time_to_next_cg20` | Temps (s) avant le prochain CG ≤ 20 km (plafonné à 1h) |
| `time_to_next_cg3` | Temps (s) avant le prochain CG ≤ 3 km (plafonné à 1h) |

Les modèles prédisent **P(aucun éclair CG dans les X prochaines minutes)** :
- probabilité **faible** → danger imminent
- probabilité **haute** → situation calme

### Aéroports couverts

| Aéroport | Coordonnées |
|---|---|
| Bron (Lyon) | 4.939°E, 45.729°N |
| Bastia | 9.484°E, 42.553°N |
| Ajaccio | 8.803°E, 41.924°N |
| Nantes | -1.611°W, 47.153°N |
| Pise | 10.399°E, 43.695°N |
| Biarritz | -1.524°W, 43.468°N |

---

## 4. Feature Engineering

Le pipeline (`src/feature_engineering_function.py`) construit **70+ variables dérivées** organisées en 13 familles à partir des données brutes.

| Famille | Exemples | N |
|---|---|---|
| **Temporel** | `hour`, `month`, `season` | 4 |
| **Délais passés** | `time_since_last_CG20_2`, `time_since_last_intra_cloud2` | 8 |
| **Comptages glissants** | `log_count_1/5/10/20/30min` | 10 |
| **Types d'éclairs** | `cg_20km`, `is_cloud_ground`, `cg_ratio`, `log_cg_count_*` | 13 |
| **Taux d'activité** | `rate_trend`, `activity_decay`, `activity_acceleration` | 6 |
| **Spatial** | `min_dist_1/5/10min`, `mean_dist_10min`, `storm_spread` | 11 |
| **Azimuth** | `mean_azimuth_10min`, `std_azimuth_10min`, `storm_direction_change` | 6 |
| **Amplitude** | `max_amplitude_10min`, `amplitude_change`, `log_std_amplitude_10min` | 7 |
| **Alerte** | `alert_duration`, `burst_indicator`, `silence_30min` | 7 |
| **Dynamique de la cellule** | `storm_velocity`, `storm_center_distance`, `storm_center_move` | 9 |

---

## 5. Modèles

### Architecture

Quatre modèles **XGBoost** de classification binaire, chacun répondant à une question différente :

| Modèle | Fichier | Question | Horizon |
|---|---|---|---|
| `xgb_cg10` | `xgb_cg10_artefacts.pkl` | CG ≤ 20 km dans les 10 min ? | Court terme |
| `xgb_cg15` | `xgb_cg15_artefacts.pkl` | CG ≤ 20 km dans les 15 min ? | **Principal** |
| `xgb_cg30` | `xgb_cg30_artefacts.pkl` | CG ≤ 20 km dans les 30 min ? | Long terme |
| `xgb_cg15_3km` | `xgb_cg15_3km_artefacts.pkl` | CG ≤ 3 km dans les 15 min ? | Danger proche |

### Contenu des artefacts

Chaque fichier `.pkl` contient :
- `model` : modèle XGBoost entraîné
- `imputer` : `SimpleImputer` sklearn ajusté sur les données d'entraînement
- `vars_to_use` : liste des features attendues par le modèle

### Entraînement

- Données : `segment_alerts_all_airports_train.csv` (507 071 lignes)
- Validation : stratifiée par aéroport
- Optimisation des hyperparamètres par recherche bayésienne
- Notebooks dédiés dans `entrainement_models/`

---

## 6. Système d'alerte

Le système de décision combine les 4 modèles selon un arbre de décision appliqué à chaque éclair détecté :

```
1. [RÈGLE ABSOLUE]  CG ≤ 20 km détecté hors alerte
   └── → Alerte immédiate 10 min

2. [FIN D'ALERTE]   p10(dernier éclair) ≤ seuil ?
   ├── OUI → Prolongation 10 min
   └── NON → Fin d'alerte

3. [DANGER LOINTAIN]  p30 > seuil ET p15 > seuil ET p10 > seuil
   └── → Attente 30 secondes

4. [DANGER IMMÉDIAT]  p15_3km ≤ seuil OU p10 ≤ seuil
   └── → Alerte 10 min

5. [FENÊTRE 5 MIN]  Aucun éclair dans les 5 prochaines minutes ?
   └── → Alerte différée de 10 min (déclenchée dans 5 min)
```

### Optimisation du seuil

Le seuil de décision (commun aux 4 modèles) est optimisé via une **validation croisée temporelle** (5 folds) sur les données d'entraînement, en minimisant un score combiné :

```
Score(s) = (durée_normalisée) + (taux_CG20_manqués_normalisé)
```

Voir `dataset_test/threshold_cv_optimization.ipynb` pour les détails.

---

## 7. Évaluation et comparaison

### Métriques

| Métrique | Description |
|---|---|
| **POD** | Probability of Detection — % de CG20 survenus pendant une alerte |
| **FAR** | False Alarm Rate — % d'alertes sans CG20 |
| **CSI** | Critical Success Index — score combinant POD et FAR |
| **Durée totale** | Somme de toutes les durées d'alerte (heures) |
| **CG20 manqués** | Éclairs CG20 survenus sans alerte active |

### Résultats sur le jeu d'évaluation

| Métrique | Modèle | Baseline 30 min |
|---|---|---|
| POD | **0.9493** | 0.9400 |
| FAR | **0.3117** | 0.3506 |
| CSI | **0.9251** | 0.9206 |
| CG20 manqués | **914** | 1081 |
| Durée totale | **687 h** | 1077 h |
| Durée moyenne/alerte | **27 min** | 60 min |
| Taux d'alerte | **81%** | 85% |

**Gain principal : −36% de durée totale d'alerte** avec une meilleure couverture des CG20.

### Notebooks d'évaluation

- `dataset_test/Evaluation_databattle_meteorage.ipynb` — évaluation complète avec toutes les métriques et visualisations
- `dataset_test/threshold_cv_optimization.ipynb` — recherche du seuil optimal par CV temporelle
- `dataset_test/rapport_approche_modele.ipynb` — rapport complet avec comparaisons et graphiques

---

## 8. Application Streamlit

Une interface web interactive permet d'utiliser les modèles sans code.

```bash
streamlit run app.py
```

**Fonctionnalités :**
- Sélection du modèle (4 horizons disponibles)
- Upload d'un fichier CSV ou utilisation des données exemples
- Ajustement interactif du seuil de décision (0.10 – 0.90)
- Visualisation des distributions de risque par aéroport
- Export des prédictions au format CSV

**Niveaux de risque :**

| Score | Niveau |
|---|---|
| ≥ 0.80 | Very High Risk |
| ≥ 0.60 | High Risk |
| ≥ 0.40 | Moderate Risk |
| ≥ 0.20 | Low Risk |
| < 0.20 | Very Low Risk |

---

## 9. Installation et utilisation

### Prérequis

```bash
pip install -r requirements.txt
```

Dépendances principales : `streamlit`, `pandas`, `numpy`, `xgboost`, `scikit-learn`, `joblib`, `plotly`, `matplotlib`

### Utilisation des modèles en Python

```python
import joblib
import pandas as pd
from src.feature_engineering_function import build_features

# Charger un modèle
artefacts   = joblib.load('models/xgb_cg15_artefacts.pkl')
model       = artefacts['model']
imputer     = artefacts['imputer']
vars_to_use = artefacts['vars_to_use']

# Préparer les données
df_raw = pd.read_csv('data/segment_alerts_all_airports_eval.csv')
df, *_ = build_features(df_raw)
df = df.reset_index(drop=True)

# Imputer + prédire
import pandas as pd
df[vars_to_use] = imputer.transform(df[vars_to_use])
df_enc = pd.get_dummies(df[vars_to_use])
probas = model.predict_proba(df_enc)[:, 1]
# probas[i] ≈ P(aucun CG20 dans les 15 prochaines minutes)
```

### Reproduire l'optimisation du seuil

Exécuter dans l'ordre :
1. `entrainement_models/t15_xgboost.ipynb` (et les autres horizons)
2. `dataset_test/threshold_cv_optimization.ipynb`
3. `dataset_test/Evaluation_databattle_meteorage.ipynb`

---

## 10. Résultats

Le système développé apporte des améliorations mesurables sur tous les indicateurs par rapport à la baseline de 30 minutes :

- **−36% de durée totale d'alerte** (687 h vs 1 077 h)
- **+1% de POD** (plus de CG20 couverts)
- **−12% de FAR** (moins de fausses alertes)
- **−167 CG20 manqués** de moins qu'avec la baseline

Ces résultats sont obtenus grâce à la combinaison de :
1. Une prédiction multi-horizon (10, 15, 30 min)
2. Une règle absolue sur les CG20 détectés
3. Un seuil de décision optimisé par validation croisée temporelle

---

*Projet réalisé dans le cadre du Data Battle IA PAU 2026 — Météorage*

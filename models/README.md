# models/

Modèles XGBoost entraînés, sérialisés avec `joblib`.

---

## Fichiers

| Fichier | Taille | Horizon | Rayon cible | Description |
|---|---|---|---|---|
| `xgb_cg10_artefacts.pkl` | 2.8 MB | 10 min | ≤ 20 km | Prédiction court terme |
| `xgb_cg15_artefacts.pkl` | 4.7 MB | 15 min | ≤ 20 km | **Modèle principal** |
| `xgb_cg30_artefacts.pkl` | 3.5 MB | 30 min | ≤ 20 km | Anticipation longue durée |
| `xgb_cg15_3km_artefacts.pkl` | 1.3 MB | 15 min | ≤ 3 km | Danger de proximité immédiate |

---

## Contenu d'un artefact

Chaque fichier `.pkl` est un dictionnaire avec les clés suivantes :

```python
artefacts = joblib.load('models/xgb_cg15_artefacts.pkl')

artefacts['model']        # XGBoost classifier entraîné
artefacts['imputer']      # SimpleImputer sklearn (médiane), ajusté sur le train
artefacts['vars_to_use']  # Liste des features attendues en entrée
```

---

## Ce que les modèles prédisent

Chaque modèle est un **classificateur binaire** :

- **Sortie :** `predict_proba(X)[:, 1]` = P(aucun CG dans les X prochaines minutes)
- Probabilité **faible** → danger imminent → l'alerte doit être déclenchée
- Probabilité **haute** → situation calme → l'alerte peut être levée

```python
import joblib, pandas as pd
from src.feature_engineering_function import build_features

artefacts   = joblib.load('models/xgb_cg15_artefacts.pkl')
model       = artefacts['model']
imputer     = artefacts['imputer']
vars_to_use = artefacts['vars_to_use']

df, *_ = build_features(pd.read_csv('data/segment_alerts_all_airports_eval.csv'))
df = df.reset_index(drop=True)
df[vars_to_use] = imputer.transform(df[vars_to_use])
df_enc = pd.get_dummies(df[vars_to_use])

probas = model.predict_proba(df_enc)[:, 1]
# probas[i] ≈ P(pas de CG20 dans les 15 prochaines minutes)
```

---

## Entraînement

Chaque modèle a son notebook dédié dans `entrainement_models/`.
Les hyperparamètres ont été optimisés par recherche bayésienne sur le jeu d'entraînement.

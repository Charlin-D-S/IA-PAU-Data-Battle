# entrainement_models/

Notebooks d'entraînement des modèles XGBoost.
Chaque notebook produit un fichier `.pkl` dans `models/`.

---

## Notebooks

### `t15_xgboost.ipynb` — Modèle principal ⭐
- **Cible :** `time_to_next_cg20 > 15 min` (classification binaire)
- **Rayon :** CG ≤ 20 km
- **Produit :** `models/xgb_cg15_artefacts.pkl` (4.7 MB)
- **Taille :** 723 KB de sorties
- C'est le modèle le plus performant et le plus utilisé dans le système d'alerte

### `t10_xgboost_bis.ipynb`
- **Cible :** `time_to_next_cg20 > 10 min`
- **Rayon :** CG ≤ 20 km
- **Produit :** `models/xgb_cg10_artefacts.pkl` (2.8 MB)
- **Taille :** 741 KB
- Horizon court : réactivité maximale, utile pour les décisions de prolongation d'alerte

### `t30_xgboost.ipynb`
- **Cible :** `time_to_next_cg20 > 30 min`
- **Rayon :** CG ≤ 20 km
- **Produit :** `models/xgb_cg30_artefacts.pkl` (3.5 MB)
- **Taille :** 654 KB
- Horizon long : anticipation des épisodes orageux lointains

### `cg_3_t15_xgboost.ipynb`
- **Cible :** `time_to_next_cg3 > 15 min`
- **Rayon :** CG ≤ **3 km** (danger de proximité immédiate)
- **Produit :** `models/xgb_cg15_3km_artefacts.pkl` (1.3 MB)
- **Taille :** 513 KB
- Déclenche des alertes pour les éclairs très proches

---

## Workflow commun à chaque notebook

```
1. Chargement des données train
2. build_features() → 70+ features
3. Définition de la cible binaire (ex. time_to_next_cg20 > 15*60)
4. Split train / validation stratifié par aéroport
5. Imputation des valeurs manquantes (SimpleImputer médiane)
6. Optimisation hyperparamètres (recherche bayésienne)
7. Entraînement XGBoost final
8. Évaluation : AUC, precision, recall, F1
9. Analyse de l'importance des features
10. Sauvegarde : joblib.dump({'model', 'imputer', 'vars_to_use'}, ...)
```

---

## Rôle de chaque modèle dans le système d'alerte

| Modèle | Rôle dans la décision |
|---|---|
| `p30` | Si > seuil → danger lointain, pas d'alerte immédiate |
| `p15` | Si > seuil → pas d'alerte à 15 min |
| `p10` | Si ≤ seuil → alerte ou prolongation |
| `p15_3km` | Si ≤ seuil → danger de proximité → alerte immédiate |

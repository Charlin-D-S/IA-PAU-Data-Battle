# exploration_features/

Notebooks d'exploration des données et de développement des features.
À exécuter en premier, avant tout entraînement.

---

## Notebooks

### `data_exploration.ipynb`
Exploration initiale du jeu de données brut.

- Chargement et inspection du CSV train (507 071 lignes, 13 colonnes)
- Distribution des aéroports
- Agrégation par `airport_alert_id`
- Inspection de la variable cible (`time_to_next_cg20`)
- Taux d'éclairs CG vs IC par aéroport

### `eda_meteorage.ipynb`
Analyse exploratoire approfondie (1.4 MB de sorties).

- Qualité des données : valeurs manquantes, outliers
- Distributions statistiques par aéroport (distance, amplitude, azimuth)
- Patterns temporels : heure, mois, saison
- Activité orageuse : épisodes, durées, fréquences
- Corrélations entre features et cible
- Visualisations cartographiques des éclairs

### `feature_engineering.ipynb`
Développement itératif des features.

- Tests des fenêtres glissantes (1, 5, 10, 20, 30 min)
- Calcul des délais inter-éclairs
- Dérivation des features spatiales (vitesse, direction, centre de masse)
- Validation des transformations log
- Analyse de l'importance des features
- Consolidation dans `src/feature_engineering_function.py`

---

## Ordre d'exécution recommandé

```
data_exploration.ipynb
       ↓
eda_meteorage.ipynb
       ↓
feature_engineering.ipynb
       ↓
  src/feature_engineering_function.py  (version finale)
```

# dataset_test/

Notebooks d'évaluation, d'optimisation du seuil, et de rapport final.
S'exécutent après l'entraînement des modèles (`entrainement_models/`).

---

## Notebooks

### `threshold_cv_optimization.ipynb`
Recherche du seuil de décision optimal par **validation croisée temporelle**.

**Objectif :** trouver la valeur de seuil (commune aux 4 modèles) qui minimise conjointement :
- la durée totale des alertes
- le taux de CG20 manqués (sans alerte active)

**Méthode :**
1. Chargement du jeu train + feature engineering + prédictions des 4 modèles
2. Découpage en 5 folds temporels (chronologiques, pas aléatoires)
3. Pour chaque seuil dans `[0.30, 0.96]` par pas de 0.02, simulation du système d'alerte sur chaque fold
4. Score combiné normalisé : `durée_norm + missed_rate_norm`
5. Graphique double axe : durée totale (bleu) + % CG20 manqués (orange) vs seuil
6. Application du seuil optimal sur le jeu eval

**Sorties :**
- `threshold_cv_results.png` — graphique d'optimisation
- Valeur du seuil optimal à reporter dans `Evaluation_databattle_meteorage.ipynb`

---

### `Evaluation_databattle_meteorage.ipynb`
Évaluation complète du système d'alerte sur le jeu de test.

**Contenu :**
1. Chargement du jeu eval + feature engineering
2. Prédictions des 4 modèles XGBoost
3. Simulation du système d'alerte modèle (avec seuil optimisé)
4. Simulation de la baseline 30 min
5. Comparaison avec `compare_alert_systems()` :

| Métrique | Modèle | Baseline |
|---|---|---|
| POD | **0.9493** | 0.9400 |
| FAR | **0.3117** | 0.3506 |
| CSI | **0.9251** | 0.9206 |
| Durée totale | **687 h** | 1 077 h |
| CG20 manqués | **914** | 1 081 |

---

### `rapport_approche_modele.ipynb`
Rapport complet de l'approche, destiné à la présentation.

**Sections :**
1. Contexte et problématique
2. Description des données (statistiques, graphiques exploratoires)
3. Feature engineering (tableau des familles)
4. Les 4 modèles prédictifs (distributions des probabilités)
5. Logique du système d'alerte (arbre de décision)
6. Simulation et comparaison (métriques + graphiques)
7. Détail par aéroport
8. Bilan énergétique (`codecarbon`)
9. Conclusion et pistes d'amélioration

---

## Fonctions utilitaires définies dans les notebooks

| Fonction | Notebook | Description |
|---|---|---|
| `simulate_alert_system(df_airport, SEUIL)` | Evaluation + rapport | Simule le système modèle sur un aéroport |
| `simulate_baseline_cg20(df_airport)` | Evaluation + rapport | Simule la baseline 30 min |
| `compute_metrics(data_fold, seuil_val)` | threshold_cv | Retourne (durée_h, missed_rate) pour un seuil |
| `compare_alert_systems(data)` | Evaluation | Compare modèle vs baseline, retourne un DataFrame de métriques |
| `plot_comparison(df_global, data)` | rapport | Visualise les résultats globaux et par aéroport |

---

## Ordre d'exécution

```
threshold_cv_optimization.ipynb   →  trouve le seuil optimal
         ↓
Evaluation_databattle_meteorage.ipynb   →  évalue avec ce seuil
         ↓
rapport_approche_modele.ipynb   →  rapport final complet
```

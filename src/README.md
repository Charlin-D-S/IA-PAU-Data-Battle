# src/

Code source Python — pipeline de feature engineering.

---

## Fichiers

### `feature_engineering_function.py` ← **à utiliser dans les notebooks**
- **Lignes :** 409
- **Fonction principale :** `build_features(df_raw)`
- **Retourne :** `(df, VAR, TARGET, IDS, new_dummies)`
  - `df` : DataFrame enrichi avec toutes les features (index = entier)
  - `VAR` : liste des ~70 features numériques
  - `TARGET` : `['time_to_next_cg20', 'time_to_next_cg3', 'target_log_time', 'target_bins']`
  - `IDS` : colonnes identifiants (`lightning_id`, `date`, `airport`, …)
  - `new_dummies` : noms des colonnes one-hot (saison, aéroport)
- **Particularité :** Construit aussi `time_to_next_cg3` (rayon 3 km) en plus de `time_to_next_cg20`

### `feature_builder.py`
- **Lignes :** 278
- **Fonction principale :** `build_features(df_raw, compute_target=True)`
- Version allégée, sans `time_to_next_cg3`
- Retourne directement le DataFrame avec index = `date`

---

## Features construites (70+)

Le pipeline applique 13 étapes de transformation dans l'ordre :

| Étape | Famille | Exemples |
|---|---|---|
| 1 | Typage & tri | Conversion UTC, sort par `airport + date` |
| 2 | Temporel | `hour`, `month`, `season` |
| 3 | Délais passés | `time_since_last_CG20_2`, `time_since_last_intra_cloud2` |
| 4 | Comptages glissants | `log_count_1/5/10/20/30min` |
| 5 | Types d'éclairs | `log_cg_count_10min`, `log_ic_count_10min`, `cg_ratio` |
| 6 | Taux d'activité | `rate_trend`, `activity_decay`, `activity_acceleration` |
| 7 | Spatial | `min_dist_1/5/10min`, `storm_spread`, `std_lat_10min` |
| 8 | Azimuth | `mean_azimuth_10min`, `std_azimuth_10min`, `azimuth_diff` |
| 9 | Amplitude | `max_amplitude_10min`, `amplitude_change`, `log_std_amplitude_10min` |
| 10 | Alerte | `alert_duration`, `burst_indicator`, `silence_30min` |
| 11 | Dynamique | `delta_dist`, `storm_velocity` |
| 12 | Direction | `storm_direction_change`, `azimuth_change` |
| 13 | Centre de masse | `storm_center_distance`, `storm_center_move`, `storm_center_velocity` |

---

## Usage

```python
import sys
sys.path.append('..')  # depuis un sous-dossier

from src.feature_engineering_function import build_features

df, VAR, TARGET, IDS, new_dummies = build_features(df_raw)
df = df.reset_index(drop=True)  # important si 'date' est l'index
```

> **Note :** `build_features` définit `date` comme index. Toujours appeler `.reset_index(drop=True)` avant un `sort_values('date')` pour éviter l'ambiguïté pandas.

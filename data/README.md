# data/

Jeux de données brutes utilisés pour l'entraînement et l'évaluation des modèles.

---

## Fichiers

### `segment_alerts_all_airports_train.csv`
- **Rôle :** Données d'entraînement des modèles XGBoost
- **Lignes :** 507 071 éclairs
- **Période :** 2016 – 2022
- **Usage :** Feature engineering + entraînement dans `entrainement_models/`

### `segment_alerts_all_airports_eval.csv`
- **Rôle :** Jeu de test final (holdout), jamais vu pendant l'entraînement
- **Lignes :** 188 175 éclairs
- **Période :** 2023
- **Usage :** Évaluation dans `dataset_test/`

### `variables_description.csv`
- **Rôle :** Dictionnaire des 99 variables (nom, type, description, catégorie)
- **Usage :** Référence pour comprendre les colonnes des deux CSV

---

## Structure des CSV

| Colonne | Type | Description |
|---|---|---|
| `lightning_id` | int | Identifiant unique de l'éclair |
| `lightning_airport_id` | int | Identifiant éclair-aéroport |
| `date` | datetime UTC | Horodatage |
| `lon`, `lat` | float | Coordonnées GPS de l'éclair |
| `dist` | float | Distance à l'aéroport (km) |
| `azimuth` | float | Angle depuis l'aéroport (degrés) |
| `amplitude` | float | Amplitude électrique (kA) |
| `maxis` | float | Nombre de maxima du signal |
| `icloud` | bool | `True` = intra-nuage, `False` = nuage-sol (CG) |
| `airport` | str | Code aéroport (6 valeurs) |
| `airport_alert_id` | float | Identifiant d'épisode d'alerte |
| `time_to_next_cg20` | float | **Cible** — secondes avant le prochain CG ≤ 20 km |
| `time_to_next_cg3` | float | **Cible** — secondes avant le prochain CG ≤ 3 km |

## Aéroports

| Code | Ville | Coordonnées |
|---|---|---|
| `Bron` | Lyon | 4.939°E, 45.729°N |
| `Bastia` | Bastia (Corse) | 9.484°E, 42.553°N |
| `Ajaccio` | Ajaccio (Corse) | 8.803°E, 41.924°N |
| `Nantes` | Nantes | -1.611°W, 47.153°N |
| `Pise` | Pise (Italie) | 10.399°E, 43.695°N |
| `Biarritz` | Biarritz | -1.524°W, 43.468°N |

> Les fichiers `.pkl` (meteo_data) ne sont pas versionnés (trop volumineux).

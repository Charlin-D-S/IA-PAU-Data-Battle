"""
Feature Engineering Pipeline — Version fonctionnelle
=====================================================
Reproduit exactement les transformations du notebook `feature_engineering.ipynb`
dans une seule fonction `build_features(df)`.

Usage:
    import pandas as pd
    from feature_engineering_function import build_features

    df_raw = pd.read_csv('segment_alerts_all_airports_train.csv')
    df_feat, VAR, TARGET, IDS, new_dummies = build_features(df_raw)
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# ── Constantes ────────────────────────────────────────────────────────────────
AIRPORTS = ['Ajaccio', 'Bastia', 'Bron', 'Nantes', 'Biarritz', 'Pise']
AIRPORT_COORDS = {
    'Bron':     (4.9389,  45.7294),
    'Bastia':   (9.4837,  42.5527),
    'Ajaccio':  (8.8029,  41.9236),
    'Nantes':   (-1.6107, 47.1532),
    'Pise':     (10.399,  43.695),
    'Biarritz': (-1.524,  43.4683),
}


def build_features(df: pd.DataFrame, verbose: bool = True) -> tuple:
    """
    Applique toutes les étapes de feature engineering du notebook.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame brut (lecture directe du CSV).
    verbose : bool
        Affiche les messages de progression.

    Returns
    -------
    df : pd.DataFrame
        DataFrame enrichi (index = date, après suppression des lignes sans cible).
    VAR : list[str]
        Liste des features numériques.
    TARGET : list[str]
        Liste des colonnes cibles.
    IDS : list[str]
        Colonnes d'identification.
    new_dummies : list[str]
        Colonnes one-hot ajoutées.
    """
    df = df.copy()

    IDS = ['lightning_id', 'lightning_airport_id', 'date', 'lon', 'lat',
           'airport', 'airport_alert_id']
    VAR = ['dist', 'azimuth']

    # ══════════════════════════════════════════════════════════════════════════
    # 2. Typage & tri
    # ══════════════════════════════════════════════════════════════════════════
    df['date'] = pd.to_datetime(df['date'], utc=True)
    df['icloud'] = df['icloud'].astype(bool)
    df['is_last_lightning_cloud_ground'] = df['is_last_lightning_cloud_ground'].astype('boolean')
    df = df.sort_values(['airport', 'date']).reset_index(drop=True)

    if verbose:
        print('✅ Typage & tri')

    # ══════════════════════════════════════════════════════════════════════════
    # 3. Variables temporelles
    # ══════════════════════════════════════════════════════════════════════════
    df['year']   = df['date'].dt.year
    df['month']  = df['date'].dt.month
    df['hour']   = df['date'].dt.hour
    df['season'] = df['month'].map({
        12: 'Hiver',  1: 'Hiver',  2: 'Hiver',
        3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
        6: 'Été', 7: 'Été', 8: 'Été',
        9: 'Automne', 10: 'Automne', 11: 'Automne'
    })
    VAR += ['month', 'hour']

    if verbose:
        print('✅ Variables temporelles')

    # ══════════════════════════════════════════════════════════════════════════
    # 4. Délais depuis les derniers éclairs
    # ══════════════════════════════════════════════════════════════════════════
    df = df.sort_values(['airport', 'date'])

    df['date_cg20'] = df['date'].where(~df['airport_alert_id'].isna())
    df['date_cg']   = df['date'].where(df['icloud'] == False)
    df['date_ic']   = df['date'].where(df['icloud'] == True)

    # Dernier événement STRICT (shift avant ffill)
    df['_last_lightning'] = df.groupby('airport')['date'].shift(1)
    df['last_lightning_date'] = df.groupby('airport')['_last_lightning'].ffill()

    df['_last_cg20'] = df.groupby('airport')['date_cg20'].shift(1)
    df['last_cg20_date'] = df.groupby('airport')['_last_cg20'].ffill()

    df['_last_cg'] = df.groupby('airport')['date_cg'].shift(1)
    df['last_cg_date'] = df.groupby('airport')['_last_cg'].ffill()

    df['_last_ic'] = df.groupby('airport')['date_ic'].shift(1)
    df['last_ic_date'] = df.groupby('airport')['_last_ic'].ffill()

    df.drop(columns=['_last_lightning', '_last_cg20', '_last_cg', '_last_ic'], inplace=True)

    # Délais
    df['time_since_last_lightning']    = (df['date'] - df['last_lightning_date']).dt.total_seconds()
    df['time_since_last_CG20']         = (df['date'] - df['last_cg20_date']).dt.total_seconds()
    df['time_since_last_cloud_ground'] = (df['date'] - df['last_cg_date']).dt.total_seconds()
    df['time_since_last_intra_cloud']  = (df['date'] - df['last_ic_date']).dt.total_seconds()

    # Prochain CG20 STRICT
    df['_cg20_shifted'] = df.groupby('airport')['date_cg20'].shift(-1)
    df['next_cg20_date'] = df.groupby('airport')['_cg20_shifted'].bfill()
    df.drop(columns='_cg20_shifted', inplace=True)

    df['time_to_next_cg20'] = (df['next_cg20_date'] - df['date']).dt.total_seconds()
    
    # 1️⃣ identifier les éclairs cloud-ground dans 3 km
    df["cg_3km"] = (
        (~df["icloud"]) &
        (df["dist"] <= 3)
    )

    # 2️⃣ créer une colonne avec date seulement pour ces événements
    df["cg3_date"] = df["date"].where(df["cg_3km"])

    # 3️⃣ prochain événement dans chaque aéroport
    df["next_cg3_date"] = (
        df.groupby("airport")["cg3_date"]
        .bfill()
    )

    # 4️⃣ temps jusqu'au prochain éclair
    df["time_to_next_cg3"] = (
        df["next_cg3_date"] - df["date"]
    ).dt.total_seconds()

    df["time_to_next_cg3"]  = df["time_to_next_cg3"].clip(0,3600)

    # Transformations log (clippées à 1h)
    df['time_since_last_lightning2']    = np.log(df['time_since_last_lightning'].clip(0, 3600) + 1)
    df['time_since_last_cloud_ground2'] = np.log(df['time_since_last_cloud_ground'].clip(0, 3600) + 1)
    df['time_since_last_intra_cloud2']  = np.log(df['time_since_last_intra_cloud'].clip(0, 3600) + 1)
    df['time_since_last_CG20_2']        = np.log(df['time_since_last_CG20'].clip(0, 3600) + 1)

    VAR += ['time_since_last_lightning2', 'time_since_last_intra_cloud2',
            'time_since_last_cloud_ground2', 'time_since_last_CG20_2']
    VAR = list(set(VAR))

    if verbose:
        print('✅ Délais passés + futur strict')

    # ══════════════════════════════════════════════════════════════════════════
    # 5. Comptages rolling
    # ══════════════════════════════════════════════════════════════════════════
    df = df.set_index('date')

    for window in ['1min', '5min', '10min', '20min', '30min']:
        df[f'count_{window}'] = (
            df.groupby('airport')['lightning_id']
            .rolling(window).count()
            .reset_index(level=0, drop=True)
        )
        df[f'log_count_{window}'] = np.log(df[f'count_{window}'] + 1)
        VAR.append(f'log_count_{window}')

    if verbose:
        print('✅ Comptages rolling')

    # ══════════════════════════════════════════════════════════════════════════
    # 6. Comptage par type (IC / CG)
    # ══════════════════════════════════════════════════════════════════════════
    df['cg'] = (df['icloud'] == False).astype(int)
    df['ic'] = (df['icloud'] == True).astype(int)

    for window in ['5min', '10min', '20min']:
        df[f'ic_count_{window}'] = (
            df.groupby('airport')['ic'].rolling(window).sum()
            .reset_index(level=0, drop=True)
        )
        df[f'cg_count_{window}'] = (
            df.groupby('airport')['cg'].rolling(window).sum()
            .reset_index(level=0, drop=True)
        )
        df[f'log_ic_count_{window}'] = np.log(df[f'ic_count_{window}'] + 1)
        df[f'log_cg_count_{window}'] = np.log(df[f'cg_count_{window}'] + 1)
        VAR += [f'log_ic_count_{window}', f'log_cg_count_{window}']

    if verbose:
        print('✅ Comptages par type')

    # ══════════════════════════════════════════════════════════════════════════
    # 7. Taux d'activité
    # ══════════════════════════════════════════════════════════════════════════
    df['rate_1min']  = df['count_1min']
    df['rate_5min']  = df['count_5min'] / 5
    df['rate_10min'] = df['count_10min'] / 10

    df['rate_trend']            = np.log(df['count_10min'] - df['count_5min'] + 1)
    df['activity_decay']        = df['rate_5min'] / (df['rate_10min'] + 1e-6)
    df['activity_acceleration'] = df['rate_1min'] - df['rate_5min']

    VAR += ['rate_trend', 'activity_decay', 'activity_acceleration']

    if verbose:
        print('✅ Taux d\'activité')

    # ══════════════════════════════════════════════════════════════════════════
    # 8. Variables spatiales & azimut
    # ══════════════════════════════════════════════════════════════════════════
    for window in ['1min', '5min', '10min']:
        df[f'mean_dist_{window}'] = (
            df.groupby('airport')['dist'].rolling(window).mean()
            .reset_index(level=0, drop=True)
        )
        df[f'min_dist_{window}'] = (
            df.groupby('airport')['dist'].rolling(window).min()
            .reset_index(level=0, drop=True)
        )
        VAR += [f'mean_dist_{window}', f'min_dist_{window}']

    df['distance_trend'] = df['mean_dist_1min'] - df['mean_dist_10min']

    df['std_lat_10min'] = (
        df.groupby('airport')['lat'].rolling('10min').std()
        .reset_index(level=0, drop=True)
    ).fillna(0)

    df['std_lon_10min'] = (
        df.groupby('airport')['lon'].rolling('10min').std()
        .reset_index(level=0, drop=True)
    ).fillna(0)

    df['storm_spread'] = df['std_lat_10min'] + df['std_lon_10min']

    for window in ['1min', '10min']:
        df[f'mean_azimuth_{window}'] = (
            df.groupby('airport')['azimuth'].rolling(window).mean()
            .reset_index(level=0, drop=True)
        )
        df[f'std_azimuth_{window}'] = (
            df.groupby('airport')['azimuth'].rolling(window).std()
            .reset_index(level=0, drop=True)
        ).fillna(0)
        VAR += [f'mean_azimuth_{window}', f'std_azimuth_{window}']

    df['azimuth_change'] = df['mean_azimuth_1min'] - df['mean_azimuth_10min']

    VAR += ['distance_trend', 'std_lat_10min', 'std_lon_10min',
            'storm_spread', 'azimuth_change']

    if verbose:
        print('✅ Variables spatiales & azimut')

    # ══════════════════════════════════════════════════════════════════════════
    # 9. Intensité (amplitude)
    # ══════════════════════════════════════════════════════════════════════════
    for window in ['1min', '10min']:
        df[f'mean_amplitude_{window}'] = (
            df.groupby('airport')['amplitude'].rolling(window).mean()
            .reset_index(level=0, drop=True)
        )
        df[f'max_amplitude_{window}'] = (
            df.groupby('airport')['amplitude'].rolling(window).max()
            .reset_index(level=0, drop=True)
        )
        VAR += [f'mean_amplitude_{window}', f'max_amplitude_{window}']

    df['amplitude_change'] = df['mean_amplitude_1min'] - df['mean_amplitude_10min']

    df['std_amplitude_10min'] = (
        df.groupby('airport')['amplitude'].rolling('10min').std()
        .reset_index(level=0, drop=True)
    )
    df['log_std_amplitude_10min'] = np.log(df['std_amplitude_10min'].fillna(0) + 1)

    VAR += ['amplitude_change', 'log_std_amplitude_10min']

    if verbose:
        print('✅ Variables amplitude')

    # ══════════════════════════════════════════════════════════════════════════
    # 10. Durée d'alerte & indicateurs de burst
    # ══════════════════════════════════════════════════════════════════════════
    df['cg_ratio']        = df['cg_count_10min'] / (df['count_10min'] + 1e-6)
    df['burst_indicator'] = (df['count_1min'] > 5).astype(int)

    df['date'] = pd.to_datetime(df.index, utc=True)
    df['alert_start'] = (
        df.groupby(['airport', 'airport_alert_id'])['date']
        .transform('min')
    )
    df['alert_duration'] = (
        df.index - df['alert_start']
    ).dt.total_seconds()

    df['alert_duration'] = df['alert_duration'].clip(0, 3600)

    VAR += ['cg_ratio', 'burst_indicator', 'alert_duration']

    if verbose:
        print('✅ Variables alerte')

    # ══════════════════════════════════════════════════════════════════════════
    # 11. Dynamique de déplacement
    # ══════════════════════════════════════════════════════════════════════════
    df['delta_t']    = df.groupby('airport')['date'].diff().dt.total_seconds()
    df['delta_dist'] = df.groupby('airport')['dist'].diff()
    df['storm_velocity'] = df['delta_dist'] / (df['delta_t'] + 1e-6)

    bol = df['time_since_last_lightning'] >= 3600
    df.loc[bol, ['delta_dist', 'storm_velocity']] = 0
    df[['delta_dist', 'storm_velocity']] = df[['delta_dist', 'storm_velocity']].fillna(0)

    VAR += ['delta_dist', 'storm_velocity']

    if verbose:
        print('✅ Dynamique orage')

    # ══════════════════════════════════════════════════════════════════════════
    # 12. Silence & changement de direction
    # ══════════════════════════════════════════════════════════════════════════
    df['silence_30min'] = (df['time_since_last_lightning'] > 1800).astype(int)

    df['azimuth_diff'] = df.groupby('airport')['azimuth'].diff()
    df['azimuth_diff'] = df['azimuth_diff'].fillna(0)
    df.loc[bol, 'azimuth_diff'] = 0

    df['storm_direction_change'] = np.log(df['azimuth_diff'].abs() + 1)

    VAR += ['silence_30min', 'azimuth_diff', 'storm_direction_change']

    if verbose:
        print('✅ Silence & direction')

    # ══════════════════════════════════════════════════════════════════════════
    # 13. Centre de masse de l'orage
    # ══════════════════════════════════════════════════════════════════════════
    df['storm_lat_center'] = (
        df.groupby('airport')['lat'].rolling('10min').mean()
        .reset_index(level=0, drop=True)
    )
    df['storm_lon_center'] = (
        df.groupby('airport')['lon'].rolling('10min').mean()
        .reset_index(level=0, drop=True)
    )

    df['airport_lat'] = df['airport'].map(lambda x: AIRPORT_COORDS[x][1])
    df['airport_lon'] = df['airport'].map(lambda x: AIRPORT_COORDS[x][0])

    df['storm_center_distance'] = np.sqrt(
        (df['storm_lat_center'] - df['airport_lat'])**2 +
        (df['storm_lon_center'] - df['airport_lon'])**2
    )

    df['storm_center_move'] = df.groupby('airport')['storm_center_distance'].diff()
    df.loc[bol, 'storm_center_move'] = 0

    df['storm_center_velocity'] = df['storm_center_move'] / (df['time_since_last_lightning'] + 1)

    VAR += ['storm_center_velocity', 'storm_spread', 'storm_center_distance', 'storm_center_move']

    if verbose:
        print('✅ Centre de masse')

    # ══════════════════════════════════════════════════════════════════════════
    # 14. Construction de la cible
    # ══════════════════════════════════════════════════════════════════════════
    df['is_cloud_ground'] = (df['icloud'] == False).astype(int)
    df['cg_20km'] = (~df['icloud']) & (df['dist'] <= 20)

    #df = df[df['time_to_next_cg20'].notna()]

    df['target_log_time'] = np.log(df['time_to_next_cg20'] + 1)

    bins   = [0, 300, 600, 1200, 1800, np.inf]
    labels = [0, 1, 2, 3, 4]
    df['target_bins'] = pd.cut(df['time_to_next_cg20'], bins=bins,
                               labels=labels, include_lowest=True)

    VAR += ['is_cloud_ground', 'cg_20km']
    TARGET = ['time_to_next_cg20', "time_to_next_cg3",'target_log_time', 'target_bins']

    if verbose:
        print(f'✅ Cible créée | {len(df):,} lignes conservées')

    # ══════════════════════════════════════════════════════════════════════════
    # Dé-duplication de VAR
    # ══════════════════════════════════════════════════════════════════════════
    VAR = list(set(VAR))

    new_dummies = [
        'season_Automne', 'season_Hiver', 'season_Printemps',
        'airport_Ajaccio', 'airport_Pise', 'airport_Bastia',
        'airport_Biarritz', 'airport_Nantes'
    ]

    return df, VAR, TARGET, IDS, new_dummies

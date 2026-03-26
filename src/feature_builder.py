"""
feature_builder.py
------------------
Fonction unique de construction des features météo-orageuses.
Encapsule toute la logique du notebook feature_engineering_documented.ipynb.

Usage
-----
from src.feature_builder import build_features

df_featured, VAR = build_features(df_raw, compute_target=True)
"""

import pandas as pd
import numpy as np

AIRPORT_COORDS = {
    'Bron':     (4.9389,  45.7294),
    'Bastia':   (9.4837,  42.5527),
    'Ajaccio':  (8.8029,  41.9236),
    'Nantes':   (-1.6107, 47.1532),
    'Pise':     (10.399,  43.695),
    'Biarritz': (-1.524,  43.4683),
}

SEASON_MAP = {
    12: 'Hiver', 1: 'Hiver',  2: 'Hiver',
    3: 'Printemps', 4: 'Printemps', 5: 'Printemps',
    6: 'Été',  7: 'Été',  8: 'Été',
    9: 'Automne', 10: 'Automne', 11: 'Automne',
}


def build_features(df_raw: pd.DataFrame, compute_target: bool = True) -> tuple[pd.DataFrame, list]:
    """
    Construit toutes les features à partir d'un DataFrame brut.

    Paramètres
    ----------
    df_raw         : DataFrame brut avec les colonnes originales du CSV
                     (lightning_id, date, lon, lat, airport, icloud, dist,
                      azimuth, amplitude, maxis, airport_alert_id, ...)
    compute_target : Si True, calcule les colonnes cible (time_to_next_cg20,
                     target_log_time, target_bins). Mettre False en inférence.

    Retourne
    --------
    df  : DataFrame enrichi (index = date, trié par airport + date)
    VAR : liste des features numériques construites
    """
    df  = df_raw.copy()
    VAR = ['dist', 'azimuth']

    # ── 1. Typage & tri ──────────────────────────────────────────────────────
    df['date']    = pd.to_datetime(df['date'], utc=True)
    df['icloud']  = df['icloud'].astype(bool)
    if 'is_last_lightning_cloud_ground' in df.columns:
        df['is_last_lightning_cloud_ground'] = df['is_last_lightning_cloud_ground'].astype('boolean')

    df = df.sort_values(['airport', 'date']).reset_index(drop=True)

    # ── 2. Variables temporelles ─────────────────────────────────────────────
    df['year']   = df['date'].dt.year
    df['month']  = df['date'].dt.month
    df['hour']   = df['date'].dt.hour
    df['season'] = df['month'].map(SEASON_MAP)
    VAR += ['month', 'hour']

    # ── 3. Délais depuis les derniers événements (passé strict) ──────────────
    df['date_cg20'] = df['date'].where(~df['airport_alert_id'].isna()) if 'airport_alert_id' in df.columns else pd.NaT
    df['date_cg']   = df['date'].where(~df['icloud'])
    df['date_ic']   = df['date'].where(df['icloud'])

    for col, src in [('_last_lightning', 'date'),
                     ('_last_cg20',      'date_cg20'),
                     ('_last_cg',        'date_cg'),
                     ('_last_ic',        'date_ic')]:
        df[col] = df.groupby('airport')[src].shift(1)

    df['last_lightning_date'] = df.groupby('airport')['_last_lightning'].ffill()
    df['last_cg20_date']      = df.groupby('airport')['_last_cg20'].ffill()
    df['last_cg_date']        = df.groupby('airport')['_last_cg'].ffill()
    df['last_ic_date']        = df.groupby('airport')['_last_ic'].ffill()
    df.drop(columns=['_last_lightning', '_last_cg20', '_last_cg', '_last_ic'], inplace=True)

    df['time_since_last_lightning']    = (df['date'] - df['last_lightning_date']).dt.total_seconds()
    df['time_since_last_CG20']         = (df['date'] - df['last_cg20_date']).dt.total_seconds()
    df['time_since_last_cloud_ground'] = (df['date'] - df['last_cg_date']).dt.total_seconds()
    df['time_since_last_intra_cloud']  = (df['date'] - df['last_ic_date']).dt.total_seconds()

    # Prochain CG20 (futur strict)
    df['_cg20_shifted']  = df.groupby('airport')['date_cg20'].shift(-1)
    df['next_cg20_date'] = df.groupby('airport')['_cg20_shifted'].bfill()
    df['time_to_next_cg20'] = (df['next_cg20_date'] - df['date']).dt.total_seconds()
    df.drop(columns='_cg20_shifted', inplace=True)

    # Transformations log (variance stabilisation, censure à 1h)
    for raw, log in [('time_since_last_lightning',    'time_since_last_lightning2'),
                     ('time_since_last_cloud_ground', 'time_since_last_cloud_ground2'),
                     ('time_since_last_intra_cloud',  'time_since_last_intra_cloud2'),
                     ('time_since_last_CG20',         'time_since_last_CG20_2')]:
        df[log] = np.log(df[raw].clip(0, 3600).fillna(0) + 1)
        VAR.append(log)

    # ── 4. Comptages rolling ─────────────────────────────────────────────────
    df = df.set_index('date')

    # Indicateur éclair isolé (> 1h) — calculé APRÈS set_index pour alignement correct
    bol = (df['time_since_last_lightning'] >= 3600).values  # .values → tableau numpy, pas de pb d'index

    for window in ['1min', '5min', '10min', '20min', '30min']:
        df[f'count_{window}'] = (
            df.groupby('airport')['lightning_id']
            .rolling(window).count()
            .reset_index(level=0, drop=True)
        )
        df[f'log_count_{window}'] = np.log(df[f'count_{window}'] + 1)
        VAR.append(f'log_count_{window}')

    # ── 5. Comptages par type (CG / IC) ──────────────────────────────────────
    df['cg'] = (~df['icloud']).astype(int)
    df['ic'] = df['icloud'].astype(int)

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

    # ── 6. Taux d'activité ───────────────────────────────────────────────────
    df['rate_1min']  = df['count_1min']
    df['rate_5min']  = df['count_5min'] / 5
    df['rate_10min'] = df['count_10min'] / 10

    df['rate_trend']            = np.log(df['count_10min'] - df['count_5min'] + 1)
    df['activity_decay']        = df['rate_5min'] / (df['rate_10min'] + 1e-6)
    df['activity_acceleration'] = df['rate_1min'] - df['rate_5min']

    VAR += ['rate_trend', 'activity_decay', 'activity_acceleration']

    # ── 7. Variables spatiales & azimut ──────────────────────────────────────
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
    VAR += ['distance_trend', 'std_lat_10min', 'std_lon_10min', 'storm_spread', 'azimuth_change']

    # ── 8. Amplitude ─────────────────────────────────────────────────────────
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
    df['log_std_amplitude_10min'] = np.log(
        df.groupby('airport')['amplitude'].rolling('10min').std()
        .reset_index(level=0, drop=True).fillna(0) + 1
    )
    VAR += ['amplitude_change', 'log_std_amplitude_10min']

    # ── 9. Variables alerte ──────────────────────────────────────────────────
    df['cg_ratio']        = df['cg_count_10min'] / (df['count_10min'] + 1e-6)
    df['burst_indicator'] = (df['count_1min'] > 5).astype(int)

    df['date'] = pd.to_datetime(df.index, utc=True)
    if 'airport_alert_id' in df.columns:
        df['alert_start'] = (
            df.groupby(['airport', 'airport_alert_id'])['date']
            .transform('min')
        )
        df['alert_duration'] = (df.index - df['alert_start']).dt.total_seconds().clip(0, 3600)
    else:
        df['alert_duration'] = 0.0

    VAR += ['cg_ratio', 'burst_indicator', 'alert_duration']

    # ── 10. Dynamique de l'orage ─────────────────────────────────────────────
    df['delta_t']    = df.groupby('airport')['date'].diff().dt.total_seconds()
    df['delta_dist'] = df.groupby('airport')['dist'].diff()
    df['storm_velocity'] = df['delta_dist'] / (df['delta_t'] + 1e-6)

    df.loc[bol, ['delta_dist', 'storm_velocity']] = 0
    df[['delta_dist', 'storm_velocity']] = df[['delta_dist', 'storm_velocity']].fillna(0)
    VAR += ['delta_dist', 'storm_velocity']

    # ── 11. Silence & direction ──────────────────────────────────────────────
    df['silence_30min'] = (df['time_since_last_lightning'] > 1800).astype(int)
    df['azimuth_diff']  = df.groupby('airport')['azimuth'].diff().fillna(0)
    df.loc[bol, 'azimuth_diff'] = 0
    df['storm_direction_change'] = np.log(df['azimuth_diff'].abs() + 1)
    VAR += ['silence_30min', 'azimuth_diff', 'storm_direction_change']

    # ── 12. Centre de masse de l'orage ───────────────────────────────────────
    df['storm_lat_center'] = (
        df.groupby('airport')['lat'].rolling('10min').mean()
        .reset_index(level=0, drop=True)
    )
    df['storm_lon_center'] = (
        df.groupby('airport')['lon'].rolling('10min').mean()
        .reset_index(level=0, drop=True)
    )

    df['airport_lat'] = df['airport'].map(lambda x: AIRPORT_COORDS.get(x, (np.nan, np.nan))[1])
    df['airport_lon'] = df['airport'].map(lambda x: AIRPORT_COORDS.get(x, (np.nan, np.nan))[0])

    df['storm_center_distance'] = np.sqrt(
        (df['storm_lat_center'] - df['airport_lat'])**2 +
        (df['storm_lon_center'] - df['airport_lon'])**2
    )
    df['storm_center_move'] = df.groupby('airport')['storm_center_distance'].diff().fillna(0)
    df.loc[bol, 'storm_center_move'] = 0
    df['storm_center_velocity'] = df['storm_center_move'] / (df['time_since_last_lightning'] + 1)

    VAR += ['storm_center_velocity', 'storm_spread', 'storm_center_distance', 'storm_center_move']

    # ── 13. Variables cibles (train uniquement) ──────────────────────────────
    df['is_cloud_ground'] = (~df['icloud']).astype(int)
    df['cg_20km']         = (~df['icloud']) & (df['dist'] <= 20)
    VAR += ['is_cloud_ground', 'cg_20km']

    if compute_target:
        df = df[df['time_to_next_cg20'].notna()].copy()
        df['target_log_time'] = np.log(df['time_to_next_cg20'] + 1)
        df['target_bins'] = pd.cut(
            df['time_to_next_cg20'],
            bins=[0, 300, 600, 1200, 1800, np.inf],
            labels=[0, 1, 2, 3, 4],
            include_lowest=True
        )

    VAR = list(set(VAR))
    print(f'build_features : {len(df):,} lignes | {len(VAR)} features construites')
    return df, VAR

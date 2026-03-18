import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title='Meteorage Demo', layout='wide')

# -----------------------------
# Config
# -----------------------------
DEFAULT_HORIZON = 30
TIME_COLS = [f"t_{j}" for j in range(1, DEFAULT_HORIZON + 1)]

# Adjust these paths if needed
MODEL_PATH = Path('models/lgb_hazard_model.joblib')
DATA_PATH = Path('data/df_model_demo.parquet')
STATIC_FEATURES_PATH = Path('models/static_features.joblib')


@st.cache_resource

def load_model(model_path: Path):
    if not model_path.exists():
        return None
    return joblib.load(model_path)


@st.cache_data

def load_data(data_path: Path):
    if not data_path.exists():
        return None
    if data_path.suffix == '.parquet':
        return pd.read_parquet(data_path)
    return pd.read_csv(data_path, parse_dates=['date'])


@st.cache_data

def load_static_features(path: Path):
    if not path.exists():
        return None
    return joblib.load(path)


def infer_model_features(static_features, horizon=DEFAULT_HORIZON):
    return list(static_features) + [f"t_{j}" for j in range(1, horizon + 1)]



def predict_survival_curve(model, row_features: dict, static_features, horizon=DEFAULT_HORIZON):
    seq_rows = []
    for k in range(1, horizon + 1):
        r = {f: row_features[f] for f in static_features}
        for j in range(1, horizon + 1):
            r[f't_{j}'] = int(j == k)
        seq_rows.append(r)

    X_seq = pd.DataFrame(seq_rows)
    model_features = infer_model_features(static_features, horizon)
    X_seq = X_seq[model_features]

    # convert booleans if any
    bool_cols = X_seq.select_dtypes(include='bool').columns
    if len(bool_cols) > 0:
        X_seq[bool_cols] = X_seq[bool_cols].astype(int)

    hazards = model.predict_proba(X_seq)[:, 1]
    survival = np.cumprod(1 - hazards)

    return pd.DataFrame({
        'minute': np.arange(1, horizon + 1),
        'hazard': hazards,
        'survival': survival,
        'prob_new_cg_within_h': 1 - survival,
    })


st.title('Démo Meteorage — Probabilité de levée d’alerte')
st.caption("Prototype de démonstration basé sur un modèle de durée discret.")

with st.sidebar:
    st.header('Configuration')
    horizon = st.slider('Horizon maximum (minutes)', min_value=5, max_value=30, value=30, step=5)
    decision_h = st.select_slider('Horizon de décision', options=[5, 10, 15, 20, 30], value=15)
    tau = st.slider('Seuil de sécurité', min_value=0.50, max_value=0.99, value=0.90, step=0.01)

model = load_model(MODEL_PATH)
df_model = load_data(DATA_PATH)
static_features = load_static_features(STATIC_FEATURES_PATH)

if model is None or df_model is None or static_features is None:
    st.warning(
        "Fichiers manquants. Pour que la démo fonctionne, place ces fichiers :\n"
        "- models/lgb_hazard_model.joblib\n"
        "- models/static_features.joblib\n"
        "- data/df_model_demo.parquet\n\n"
        "Le code de l'application est prêt ; il suffit maintenant d'exporter le modèle et un échantillon de données."
    )
    st.stop()

# Clean expected columns
if 'date' in df_model.columns:
    df_model['date'] = pd.to_datetime(df_model['date'], utc=True, errors='coerce')

required = [c for c in static_features if c not in df_model.columns]
if required:
    st.error(f"Colonnes manquantes dans df_model_demo : {required[:10]}")
    st.stop()

# UI filters
col1, col2, col3 = st.columns(3)
with col1:
    airports = sorted(df_model['airport'].dropna().unique().tolist()) if 'airport' in df_model.columns else []
    airport = st.selectbox('Aéroport', airports)
with col2:
    airport_df = df_model[df_model['airport'] == airport].copy() if 'airport' in df_model.columns else df_model.copy()
    available_dates = sorted(airport_df['date'].dt.date.dropna().unique().tolist()) if 'date' in airport_df.columns else []
    selected_day = st.selectbox('Jour', available_dates)
with col3:
    day_df = airport_df[airport_df['date'].dt.date == selected_day].copy() if 'date' in airport_df.columns else airport_df.copy()
    day_df = day_df.sort_values('date')
    display_idx = day_df.reset_index(drop=True).index.tolist()
    idx = st.select_slider('Observation dans la journée', options=display_idx, value=display_idx[min(0, len(display_idx)-1)] if display_idx else 0)

if day_df.empty:
    st.info('Aucune observation disponible pour cette sélection.')
    st.stop()

row = day_df.reset_index(drop=True).iloc[idx]
row_features = row[static_features].to_dict()
curve = predict_survival_curve(model, row_features, static_features, horizon=horizon)

# Metrics
p_safe = float(curve.loc[curve['minute'] == decision_h, 'survival'].iloc[0])
p_event = 1 - p_safe
recommendation = 'Levée possible' if p_safe >= tau else 'Alerte à maintenir'

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"P(aucun CG ≤ {decision_h} min)", f"{p_safe:.1%}")
m2.metric(f"P(nouveau CG ≤ {decision_h} min)", f"{p_event:.1%}")
m3.metric("Recommandation", recommendation)
m4.metric("Observation", row['date'].strftime('%Y-%m-%d %H:%M:%S UTC') if pd.notna(row.get('date')) else 'NA')

left, right = st.columns([1.4, 1])

with left:
    fig = px.line(curve, x='minute', y='survival', markers=True,
                  title='Courbe de survie prédite',
                  labels={'minute': 'Minutes', 'survival': 'Probabilité de ne pas avoir de nouveau CG'})
    fig.add_vline(x=decision_h, line_dash='dash')
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.bar(curve, x='minute', y='hazard',
                  title='Hazard discret par minute',
                  labels={'minute': 'Minute', 'hazard': 'Probabilité d’un nouveau CG exactement à cette minute'})
    st.plotly_chart(fig2, use_container_width=True)

with right:
    st.subheader('Contexte de l’observation')
    show_cols = [
        c for c in [
            'airport', 'date', 'dist', 'azimuth', 'mean_dist_10min', 'mean_dist_5min',
            'storm_center_distance', 'log_cg_count_10min', 'log_ic_count_10min',
            'mean_amplitude_10min', 'amplitude_change', 'hour', 'month'
        ] if c in row.index
    ]
    st.dataframe(pd.DataFrame({'variable': show_cols, 'valeur': [row[c] for c in show_cols]}), hide_index=True)

    if 'time_to_next_cg20' in row.index:
        st.subheader('Vérité terrain (si disponible)')
        st.write(f"Temps réel jusqu’au prochain CG20 : **{row['time_to_next_cg20'] / 60:.1f} min**")

st.subheader('Timeline de la journée')
plot_df = day_df.copy()
plot_df['type_eclair'] = np.where(plot_df.get('icloud', False), 'IC', 'CG') if 'icloud' in plot_df.columns else 'NA'

if 'dist' in plot_df.columns and 'date' in plot_df.columns:
    fig3 = px.scatter(
        plot_df,
        x='date', y='dist', color='type_eclair',
        title='Éclairs observés dans la journée',
        labels={'date': 'Heure', 'dist': 'Distance à l’aéroport (km)'}
    )
    if pd.notna(row.get('date')):
        fig3.add_vline(x=row['date'], line_dash='dash')
    st.plotly_chart(fig3, use_container_width=True)

st.markdown('---')
st.markdown(
    "**Lecture métier.** Le modèle n’essaie pas d’identifier directement le dernier éclair. "
    "Il estime, à chaque instant, la probabilité qu’un nouvel éclair nuage-sol survienne dans les prochaines minutes, "
    "à partir de l’état courant de l’activité orageuse."
)

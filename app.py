import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

from src.feature_builder import build_features


# ============================================================
# CONFIG GÉNÉRALE
# ============================================================
st.set_page_config(
    page_title="Météorage - Prédiction de fin d'orage",
    page_icon="⛈️",
    layout="wide"
)


# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
@st.cache_resource
def load_artifact(model_path: str):
    """
    Charge le dictionnaire d'artefacts :
    {
        'model': ...,
        'vars_to_use': ...,
        'imputer': ...,
        ...
    }
    """
    return joblib.load(model_path)


@st.cache_data
def load_example_data(example_path: str):
    return pd.read_csv(example_path)


def prepare_input_dataframe(uploaded_file):
    """
    Lit le CSV uploadé dans Streamlit.
    """
    return pd.read_csv(uploaded_file)


def run_inference(df_raw: pd.DataFrame, artifact: dict):
    """
    Pipeline complet d'inférence :
    1. construction des features
    2. sélection des variables attendues par l'imputer
    3. imputation
    4. encodage de season
    5. alignement exact sur les variables du modèle
    6. prédiction
    """
    df_feat, _ = build_features(df_raw, compute_target=False)

    model = artifact["model"]
    imputer = artifact["imputer"]

    # 1) Colonnes attendues par l'imputer
    imputer_features = list(imputer.feature_names_in_)

    missing_for_imputer = [col for col in imputer_features if col not in df_feat.columns]
    if missing_for_imputer:
        raise ValueError(
            "Colonnes manquantes pour l'imputer : " + ", ".join(missing_for_imputer)
        )

    # 2) Sous-ensemble exact pour l'imputer
    X_imp_input = df_feat[imputer_features].copy()

    # 3) Imputation
    try:
        X_imp_array = imputer.transform(X_imp_input) 
    except AttributeError as e:
        if "_fill_dtype" in str(e):
            imputer._fill_dtype = X_imp_input.dtypes
            X_imp_array = imputer.transform(X_imp_input)
        else:
            raise

    # Revenir en DataFrame pour garder les noms
    X_imp_df = pd.DataFrame(
        X_imp_array,
        columns=imputer_features,
        index=df_feat.index
    )

    # 4) Gestion de season
    if "season" in df_feat.columns:
        season_dummies = pd.get_dummies(df_feat["season"].astype(str).str.strip(), prefix="season")
    else:
        season_dummies = pd.DataFrame(index=df_feat.index)

    # 5) Construire le dataset final pour le modèle
    X_model = X_imp_df.copy()

    if "season" in X_model.columns:
        X_model = X_model.drop(columns=["season"])

    X_model = pd.concat([X_model, season_dummies], axis=1)

    # 6) Aligner exactement les colonnes attendues par le modèle
    model_features = list(model.feature_names_in_)

    for col in model_features:
        if col not in X_model.columns:
            X_model[col] = 0

    X_model = X_model[model_features]

    # 7) Prédiction
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X_model)[:, 1]
    else:
        proba = model.predict(X_model)

    result = df_feat.copy()
    result["score_proba"] = proba

    return result, model_features


def make_download_csv(df: pd.DataFrame):
    return df.to_csv(index=False).encode("utf-8")


def human_label_from_score(score: float):
    if score >= 0.80:
        return "Risque très élevé"
    elif score >= 0.60:
        return "Risque élevé"
    elif score >= 0.40:
        return "Risque modéré"
    elif score >= 0.20:
        return "Risque faible"
    return "Risque très faible"


def add_risk_label(df: pd.DataFrame):
    df = df.copy()
    df["niveau_risque"] = df["score_proba"].apply(human_label_from_score)
    return df


# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("⚙️ Paramètres")

model_options = {
    "CG 10 min": "models/xgb_cg10_artefacts.pkl",
    "CG 15 min": "models/xgb_cg15_artefacts.pkl",
    "CG 30 min": "models/xgb_cg30_artefacts.pkl",
    "CG 15 min - 3km": "models/xgb_cg15_3km_artefacts.pkl",
}

selected_model_label = st.sidebar.selectbox(
    "Choisir le modèle",
    list(model_options.keys()),
    index=1
)

selected_model_path = model_options[selected_model_label]

threshold = st.sidebar.slider(
    "Seuil de classification",
    min_value=0.10,
    max_value=0.90,
    value=0.50,
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.write("Projet Data Battle - Météorage")
st.sidebar.write("Application de démonstration Streamlit")


# ============================================================
# TITRE
# ============================================================
st.title("⛈️ Prédiction de fin d'orage")
st.markdown(
    """
Cette application permet de :
- charger des données d'éclairs,
- construire automatiquement les variables du projet,
- appliquer un modèle XGBoost déjà entraîné,
- afficher les probabilités de risque et les observations les plus critiques.
"""
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Accueil", "Données", "Prédiction", "Résultats"]
)


# ============================================================
# ONGLET 1 - ACCUEIL
# ============================================================
with tab1:
    st.header("Présentation")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.write(
            """
**Objectif métier**  
Prévoir l'évolution d'un orage afin d'aider à la reprise d'activité dans des zones sensibles,
notamment les aéroports.

**Idée générale**  
À partir d'une séquence d'éclairs observés, on fabrique des variables temporelles,
spatiales et dynamiques, puis on calcule un score de risque via un modèle XGBoost.

**Ce que fait l'application**  
1. lecture du fichier brut,  
2. construction des features via `build_features`,  
3. application du modèle choisi,  
4. restitution des scores et téléchargement des résultats.
"""
        )

    with col2:
        st.info(
            f"""
**Modèle sélectionné**
- {selected_model_label}

**Seuil actuel**
- {threshold:.2f}
"""
        )

    st.markdown("---")
    st.subheader("Format attendu du fichier")

    st.write("Le CSV doit contenir au minimum des colonnes comme :")
    st.code(
        "lightning_id, date, lon, lat, amplitude, icloud, dist, azimuth, airport, airport_alert_id",
        language="text"
    )


# ============================================================
# ONGLET 2 - DONNÉES
# ============================================================
with tab2:
    st.header("Chargement des données")

    example_path = "dataset_test/dataset_set.csv"

    colA, colB = st.columns(2)

    with colA:
        use_example = st.checkbox("Utiliser le fichier d'exemple du projet", value=True)

    with colB:
        uploaded_file = st.file_uploader("Ou charger votre propre CSV", type=["csv"])

    df_raw = None

    if use_example:
        if os.path.exists(example_path):
            df_raw = load_example_data(example_path)
            st.success("Fichier d'exemple chargé.")
        else:
            st.error("Le fichier d'exemple n'a pas été trouvé.")
    elif uploaded_file is not None:
        df_raw = prepare_input_dataframe(uploaded_file)
        st.success("Fichier utilisateur chargé.")

    if df_raw is not None:
        st.subheader("Aperçu des données brutes")
        st.dataframe(df_raw.head(20), use_container_width=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Nombre de lignes", f"{df_raw.shape[0]:,}".replace(",", " "))
        c2.metric("Nombre de colonnes", df_raw.shape[1])
        c3.metric("Nombre d'aéroports", df_raw["airport"].nunique() if "airport" in df_raw.columns else "N/A")

        st.subheader("Colonnes disponibles")
        st.write(list(df_raw.columns))

        st.session_state["df_raw"] = df_raw
    else:
        st.warning("Charge un fichier ou active le fichier d'exemple.")


# ============================================================
# ONGLET 3 - PRÉDICTION
# ============================================================
with tab3:
    st.header("Lancer la prédiction")

    if "df_raw" not in st.session_state:
        st.warning("Aucune donnée n'est encore chargée. Va d'abord dans l'onglet Données.")
    else:
        df_raw = st.session_state["df_raw"]

        if st.button("Construire les features et prédire", type="primary"):
            try:
                artifact = load_artifact(selected_model_path)
                result_df, vars_used = run_inference(df_raw, artifact)

                # Recalcul de la classe avec le seuil choisi
                result_df["prediction"] = (result_df["score_proba"] >= threshold).astype(int)
                result_df = add_risk_label(result_df)

                st.session_state["result_df"] = result_df
                st.session_state["vars_used"] = vars_used
                st.session_state["selected_model_label"] = selected_model_label
                st.success("Prédiction terminée avec succès.")

            except Exception as e:
                st.error(f"Erreur pendant l'inférence : {e}")

        if "vars_used" in st.session_state:
            st.subheader("Variables utilisées par le modèle")
            st.write(st.session_state["vars_used"])


# ============================================================
# ONGLET 4 - RÉSULTATS
# ============================================================
with tab4:
    st.header("Résultats")

    if "result_df" not in st.session_state:
        st.info("Aucun résultat disponible. Lance d'abord la prédiction.")
    else:
        result_df = st.session_state["result_df"]

        st.write(f"**Modèle utilisé :** {st.session_state.get('selected_model_label', 'N/A')}")

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Nb. observations scorées", f"{len(result_df):,}".replace(",", " "))
        k2.metric("Score moyen", f"{result_df['score_proba'].mean():.3f}")
        k3.metric("Score max", f"{result_df['score_proba'].max():.3f}")
        k4.metric("Nb. prédictions positives", int(result_df["prediction"].sum()))

        st.subheader("Tableau des résultats")
        cols_to_show = [
            col for col in [
                "airport", "date", "dist", "azimuth", "amplitude",
                "score_proba", "prediction", "niveau_risque"
            ] if col in result_df.columns
        ]
        st.dataframe(result_df[cols_to_show].head(100), use_container_width=True)

        st.subheader("Distribution des scores")
        fig_hist = px.histogram(
            result_df,
            x="score_proba",
            nbins=40,
            title="Distribution des probabilités prédites"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        if "airport" in result_df.columns:
            st.subheader("Score moyen par aéroport")
            agg_airport = (
                result_df.groupby("airport", as_index=False)["score_proba"]
                .mean()
                .sort_values("score_proba", ascending=False)
            )
            fig_bar = px.bar(
                agg_airport,
                x="airport",
                y="score_proba",
                title="Score moyen par aéroport"
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        st.subheader("Top 20 observations les plus risquées")
        top_risk = result_df.sort_values("score_proba", ascending=False).head(20)
        st.dataframe(top_risk[cols_to_show], use_container_width=True)

        st.subheader("Téléchargement")
        csv_bytes = make_download_csv(result_df)
        st.download_button(
            label="Télécharger les résultats en CSV",
            data=csv_bytes,
            file_name="predictions_meteorage.csv",
            mime="text/csv"
        )
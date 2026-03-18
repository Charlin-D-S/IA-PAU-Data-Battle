import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, brier_score_loss,average_precision_score
from math import log

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

import pandas as pd
from sklearn.linear_model import LogisticRegression

def get_dummies(df, target_col='default_t_plus_1'):
    df = df.copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])
    
    # Identifier automatiquement les colonnes object ou category
    cat_cols = X.columns.tolist()
    categ = {}
    for col in cat_cols:
        X[col] = X[col].astype('str')
        X[col] = X[col].astype('category')
        
        # Calculer le taux moyen de la cible par catégorie
        risk_order = df.groupby(col)[target_col].mean().sort_values()
        # least_risk_category = risk_order.index[0]  # catégorie avec le risque le plus faible
        
        # Réordonner les catégories pour que la moins risquée soit la première
        new_categories = risk_order.index.tolist()
        X[col] = X[col].cat.reorder_categories(new_categories, ordered=False)  # <--- plus de inplace
        categ[col] = new_categories

    
    # Créer les dummies en supprimant la catégorie de base (la moins risquée)
    X_dummies = pd.get_dummies(X, drop_first=True,prefix_sep="&")*1
    
    return X_dummies, categ

def get_test_dummies(df, categ):
    X = df.copy()
    
    # Identifier automatiquement les colonnes object ou category
    cat_cols = X.columns.tolist()
    
    for col in cat_cols:
        try :
            #print(f"Processing column: {col}")
            X[col] = X[col].astype('str')
            X[col] = X[col].astype('category')
            #print(f"Categories before reordering for {col}: {X[col].cat.categories.tolist()}")
            
            # Utiliser l'ordre des catégories appris sur le train
            new_categories = categ[col]
            print(f"New categories for {col}: {new_categories}")
            X[col] = X[col].cat.rename_categories(
                {cat: str(cat) for cat in X[col].cat.categories}
            )

            X[col] = X[col].cat.reorder_categories(new_categories, ordered=False)  # <--- plus de inplace
        except Exception as e :
            raise e
                                 
    
    # Créer les dummies en supprimant la catégorie de base (la moins risquée)
    X_dummies = pd.get_dummies(X, drop_first=True,prefix_sep='&')*1
    
    return X_dummies




def plot_gini_evolution(results):

    sns.set_theme(style="whitegrid", font_scale=1.2)
    
    history = results['history'].copy()
    history['gini_test_pct_gain'] = history['gini_test'].pct_change() * 100

    # Normalisation BIC pour affichage comparé
    sc = MinMaxScaler()
    history['bic_scaled'] = sc.fit_transform(history[['bic']])

    # ---- FIGURE ----
    fig, ax1 = plt.subplots(figsize=(13, 7))

    # Palette harmonisée
    palette1 = {
        "Gini Train": "#1f77b4",
        "Gini Test": "#ff7f0e",
        "AUC-PR Test": "#2ca02c",
        "BIC (scaled)": "#9467bd"
    }
    palette2 = {
        "Ratio Performance": "#d62728",
        "Gain Gini (%)": "#8c564b"
    }

    # ---- AXE 1 : GINI, AUC-PR, BIC ----
    sns.lineplot(
        data=history, x="step", y="gini_train",
        marker="o", label="Gini Train", ax=ax1, color=palette1["Gini Train"]
    )
    sns.lineplot(
        data=history, x="step", y="gini_test",
        marker="o", label="Gini Test", ax=ax1, color=palette1["Gini Test"]
    )
    sns.lineplot(
        data=history, x="step", y="auc_pr",
        marker="o", label="AUC-PR Test", ax=ax1, color=palette1["AUC-PR Test"]
    )
    sns.lineplot(
        data=history, x="step", y="bic_scaled",
        marker="o", label="BIC (scaled)", ax=ax1, linestyle="--", color=palette1["BIC (scaled)"]
    )

    # Valeurs annotées pour Gini Test
    for _, row in history.iterrows():
        ax1.text(
            row['step'],
            row['gini_test'],
            f"{row['gini_test']:.3f}",
            ha='center', va='bottom', fontsize=10,
            color=palette1["Gini Test"]
        )

    ax1.set_xlabel("Étapes d’ajout des variables", fontsize=13)
    ax1.set_ylabel("GINI / AUC-PR / BIC (normalisé)", fontsize=13)
    ax1.set_title(
        "Évolution des métriques du modèle logistique au fil des itérations",
        fontsize=15, fontweight='bold'
    )

    # ---- AXE 2 : Ratio Perf + Gain Gini ----
    ax2 = ax1.twinx()

    sns.lineplot(
        data=history, x='step', y='rp',
        marker='D', linestyle='-', label="Ratio Performance",
        ax=ax2, color=palette2["Ratio Performance"]
    )
    sns.lineplot(
        data=history, x='step', y='gini_test_pct_gain',
        marker='D', linestyle='--', label="Gain Gini (%)",
        ax=ax2, color=palette2["Gain Gini (%)"]
    )

    # Annotations
    for _, row in history.iterrows():
        ax2.text(
            row['step'],
            row['rp'],
            f"{row['rp']:.1f}",
            ha='center', va='bottom', fontsize=10,
            color=palette2["Ratio Performance"]
        )
        ax2.text(
            row['step'],
            row['gini_test_pct_gain'],
            f"{row['gini_test_pct_gain']:.1f}%",
            ha='center', va='bottom', fontsize=10,
            color=palette2["Gain Gini (%)"]
        )

    ax2.set_ylabel("Ratio Performance / Gain Gini (%)", fontsize=13)

    # ---- Fusion élégante des légendes ----
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()

    all_handles = handles1 + handles2
    all_labels  = labels1  + labels2

    ax1.legend(
        all_handles, all_labels,
        loc='upper left',
        frameon=True,
        facecolor="white",
        fontsize=11,
        title="Métriques"
    )

    plt.tight_layout()
    plt.show()



def fit_logit_and_metrics(X_train, y_train, X_test, y_test,prevalence):
# ✅ Entraînement du modèle*
    n = len(y_train)  # nombre d'observations
    model = LogisticRegression(max_iter=1000,penalty=None,solver='newton-cg',random_state=42)
    model.fit(X_train, y_train)

    # ✅ Probabilités prédites
    proba_train = model.predict_proba(X_train)[:, 1]
    proba_test  = model.predict_proba(X_test)[:, 1]

    # ✅ Métriques
    auc_train = roc_auc_score(y_train, proba_train)
    gini_train = 2 * auc_train - 1
    brier_train = brier_score_loss(y_train, proba_train)

    auc_test = roc_auc_score(y_test, proba_test)
    auc_pr_test = average_precision_score(y_test, proba_test)
    rp = auc_pr_test/prevalence
    gini_test = 2 * auc_test - 1
    brier_test = brier_score_loss(y_test, proba_test)

    # ✅ Calcul manuel du BIC
    p = X_train.shape[1] + 1  # +1 pour l'intercept
    eps = 1e-15  # éviter log(0)
    loglik = np.sum(y_train * np.log(proba_train + eps) + (1-y_train) * np.log(1 - proba_train + eps))
    bic = -2 * loglik + p * log(n)
    aic = -2 * loglik + 2 * p


    metrics = {
        'aic': aic,
        'bic': bic,
        'gini_train': gini_train,
        'brier_train': brier_train,
        'gini_test': gini_test,
        'auc_pr': auc_pr_test,
        'rp':rp,
        'brier_test': brier_test
    }
    return model, metrics

    # Helper to build dummies for a set of variables and align train/test


def build_dummies(train_df, test_df, variables):
    if len(variables) == 0:
        # return empty frames with shape (n,0)
        return pd.DataFrame(index=train_df.index), pd.DataFrame(index=test_df.index)
    cols_subset = [c for var in variables for c in train_df.columns if c.startswith(var + "&") and '#' not in c[len(var + "&"):] and '&' not in c[len(var + "&"):]]
    Xtr = train_df[cols_subset]
    Xte = test_df[cols_subset]
    # align columns
    
    return Xtr, Xte

def stepwise_selection_categorical(
    df,                      # dataframe containing all original categorical cols + target
    y_test,              # name of target column (0/1)
    y_train,              # name of target column (0/1)
    cat_vars,                # list of categorical variable names to consider (original variables)
    X_test=None,             # optional test dataframe (same schema as df)
    direction='both',        # 'forward' or 'both'
    criterion='bic',         # 'bic', 'aic' or 'gini' (uses test gini when criterion='gini')
    max_vars=None,           # optional cap on number of variables to select
    verbose=True,
    threshold_in=0.0005,
    threshold_out=0,
    start_vars = []
):
    """
    Stepwise selection on categorical variables.
    Returns a dict: { 'history': DataFrame, 'best_vars': tuple, 'best_model': statsmodels result }
    """
    assert direction in ('forward', 'both')
    assert criterion in ('bic', 'aic', 'gini','rp','auc_pr')

    # Split train/test from inputs
    df_train = df.copy()
    if X_test is None:
        raise ValueError("Vous devez fournir X_test (dataframe test) pour calculer metrics sur test.")
    df_test = X_test.copy()

    remaining = list(cat_vars)
    selected = list(start_vars)

    history = []
    n_iter = 0
    best_score = np.inf if criterion in ('bic','aic') else -np.inf
    best_vars = None
    best_model = None
    print("Starting stepwise selection...")
    prevalence = y_test.mean()
    while True:
        n_iter += 1
        changed = False

        # --- FORWARD STEP: try adding each remaining variable and pick best improvement ---
        if remaining:
            candidates = []
            for var in remaining:
                trial_vars = selected + [var]
                Xtr, Xte = build_dummies(df_train, df_test, trial_vars)
                model, metrics = fit_logit_and_metrics(Xtr, y_train, Xte, y_test,prevalence)
                # prepare score according to criterion
                if model is None:
                    score = np.nan
                else:
                    if criterion == 'bic':
                        score = metrics['bic']
                    elif criterion == 'aic':
                        score = metrics['aic']
                    elif criterion == 'rp':
                        score = -metrics['rp']
                    elif criterion == 'auc_pr':
                        score = -metrics['auc_pr']
                    else:  # criterion == 'gini', maximize test gini -> convert to -score for minimization
                        score = - metrics['gini_test'] if metrics['gini_test'] is not None else np.nan

                candidates.append((var, score, metrics, model))

            # choose best candidate according to criterion
            # for bic/aic lower is better; for gini we negated so lower is better too.
            candidates = [c for c in candidates if not (pd.isna(c[1]))]
            if candidates:
                candidates.sort(key=lambda x: x[1])
                best_candidate = candidates[0]
                if verbose:
                    print(f"Iteration {n_iter} forward best candidate: {best_candidate[0]} score={best_candidate[1]}")
                # decide if improvement over current best
                improved = False
                if best_vars is None:
                    improved = True
                else:
                    # compare to current best_score
                    if criterion in ('bic','aic'):
                        improved = best_candidate[1] < best_score
                    else:  # gini (we stored negated)
                        improved = best_candidate[1] - best_score<-threshold_in  # require at least +0.005 improvement in gini

                if improved:
                    var_to_add = best_candidate[0]
                    selected.append(var_to_add)
                    remaining.remove(var_to_add)
                    best_score = best_candidate[1]
                    best_vars = tuple(selected)
                    best_model = best_candidate[3]
                    print(f"Iteration {n_iter} forward added {var_to_add}")
                    history.append({
                        'step': len(history)+1,
                        'best_score': best_score,
                        'metrics': best_candidate[2],
                        'action': 'add',
                        'variable': var_to_add,
                        'selected_vars': tuple(selected),
                        **{k: v for k,v in best_candidate[2].items()}
                    })
                    changed = True

                    # enforce max_vars
                    if max_vars is not None and len(selected) >= max_vars:
                        if verbose:
                            print("Reached max_vars limit.")
                        break

        # --- BACKWARD STEP: try removing each variable in selected (only if direction == 'both') ---
        if direction == 'both' and len(selected)>2:
            removal_candidates = []
            for var in list(selected):
                trial_vars = [v for v in selected if v != var]
                Xtr, Xte = build_dummies(df_train, df_test, trial_vars)
                model, metrics = fit_logit_and_metrics(Xtr, y_train, Xte, y_test,prevalence)
                if model is None:
                    score = np.nan
                else:
                    if criterion == 'bic':
                        score = metrics['bic']
                    elif criterion == 'aic':
                        score = metrics['aic']
                    elif criterion == 'rp':
                        score = -metrics['rp']
                    elif criterion == 'auc_pr':
                        score = -metrics['auc_pr']
                    else:
                        score = - metrics['gini_test'] if metrics['gini_test'] is not None else np.nan
                removal_candidates.append((var, score, metrics, model))

            removal_candidates = [c for c in removal_candidates if not (pd.isna(c[1]))]

            if removal_candidates:
                print('try to remove')
                removal_candidates.sort(key=lambda x: x[1])
                best_removal = removal_candidates[0]
                remove_bool = False
                if criterion in ('bic','aic'):
                    remove_bool = best_removal[1] < best_score
                else:  # gini
                    remove_bool = best_removal[1] - best_score<threshold_out  # require at least +0.005 improvement in gini
                # if removing improves score (lower), perform removal
                
                if remove_bool:
                    var_to_remove = best_removal[0]
                    
                    print('removing....'+var_to_remove)
                    selected.remove(var_to_remove)
                    remaining.append(var_to_remove)
                    best_score = best_removal[1]
                    best_vars = tuple(selected)
                    best_model = best_removal[3]
                    history.append({
                        'step': len(history)+1,
                        'action': 'remove',
                        'best_score': best_score,
                        'metrics': best_removal[2],
                        'variable': var_to_remove,
                        'selected_vars': tuple(selected),
                        **{k: v for k,v in best_removal[2].items()}
                    })
                    changed = True
                    if verbose:
                        print(f"Iteration {n_iter} backward removed {var_to_remove} improved score -> {best_score}")
                    if var_to_remove == var_to_add:
                        # added and removed same variable in one iteration; stop here to avoid oscillation
                        if verbose:
                            print("Added and removed same variable in one iteration; stopping to avoid oscillation.")
                        #remaining.remove(var_to_remove)
                        break
        # stop if nothing changed or reached max_vars
        if not changed:
            if verbose:
                print("No improvement; stopping stepwise.")
            break
        if max_vars is not None and len(selected) >= max_vars:
            if verbose:
                print("Reached max_vars stopping.")
            break

    # Build final model on best_vars
    if best_vars is None or len(best_vars) == 0:
        final_model = None
    else:
        Xtr_final, Xte_final = build_dummies(df_train, df_test, list(best_vars))
        final_model, final_metrics = fit_logit_and_metrics(Xtr_final, y_train, Xte_final, y_test,prevalence)

    history_df = pd.DataFrame(history)
    return {
        'history': history_df,
        'best_vars': best_vars,
        'best_model': final_model
    }


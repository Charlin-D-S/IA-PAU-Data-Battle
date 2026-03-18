import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
from scipy.stats import chi2_contingency
from itertools import combinations
class TemporalStability:
    def __init__(self, X_train, dico, data_type='Train'):
        self.X_train = X_train
        self.y_train = X_train['cible']
        self.dico = dico
        self.missing_df = self.compute_missing_stats()
        self.data_type = data_type
        
        self.mean_y = self.y_train.mean()
    
    def compute_missing_stats(self):
        X_quali = self.X_train
        # --- Missing values (jeu complet) ---
        missing_count = X_quali.isna().sum()
        missing_pct = (missing_count / len(X_quali) * 100).round(2)
        missing_df = pd.DataFrame({
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "nunique": X_quali.nunique(),
            "dtype": X_quali.dtypes.astype(str)
        }).sort_values("missing_count", ascending=False)
        # Ajouter signification depuis dico si disponible
        if 'Signification' in self.dico.columns:
            missing_df = missing_df.join(self.dico[['Signification']], how='left', on=missing_df.index).rename_axis('variable')
        return missing_df
    
    def plot_categorical_distribution(self,var_quant):
        X = pd.DataFrame({'categorie': var_quant, 'cible': self.y_train})
        table = pd.crosstab(X["categorie"], X['cible'], normalize='index') * 100
        table = table.reset_index().melt(id_vars='categorie', var_name='cible', value_name='pourcentage')
        cramers_v = self.cramers_v(var_quant,self.y_train)
        # print(isinstance(cramers_v,float))
        # print(cramers_v)
        plt.figure(figsize=(8,5))
        ax = sns.barplot(x='categorie', y='pourcentage', hue='cible', data=table, palette='Set2')
        plt.title("Distribution (%) de la cible selon la variable catégorielle")

        # Ajouter les valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=8)
        plt.suptitle(f"V de Cramer: {cramers_v:.3f}", y=1.02, fontsize=10)  
        plt.show()
    @staticmethod
    def cramers_v(x, y):
        """Calcule Cramér's V entre deux séries (gère valeurs manquantes)."""
        # garder uniquement les lignes où les deux variables sont non-nulles
        mask = x.notna() & y.notna()
        x2 = x[mask].astype(str)
        y2 = y[mask].astype(str)
        if x2.empty or y2.empty:
            return np.nan
        ct = pd.crosstab(x2, y2)
        if ct.size == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
            return np.nan
        chi2, p, dof, expected = chi2_contingency(ct, correction=False)
        n = ct.values.sum()
        r, k = ct.shape
        denom = n * (min(r - 1, k - 1))
        if denom == 0:
            return np.nan
        return np.sqrt(chi2 / denom)

    def describe_categorical(self,var):
        """Affiche les statistiques descriptives d'une variable catégorielle."""
        X_qual = self.X_train
        ser = X_qual[var]
        desc = ser.describe()
        counts = ser.value_counts(dropna=False)
        pct = (counts / len(ser) * 100).round(2)
        desc_df = pd.DataFrame({
            "count": counts,
            "pct": pct
        })
        print(f"Statistiques descriptives pour la variable catégorielle '{var}':")
        display(self.missing_df.loc[var,])
        display(desc)
        display(desc_df)
        cat = X_qual[var]
        self.plot_categorical_distribution(cat)

    def plot_evolution_par_mois(self,var):
        df_tmp = (
            self.X_train[['DATDELHIS_Mm0', var, 'cible']]
            .copy()
            .dropna(subset=['DATDELHIS_Mm0'])
            .assign(cible=lambda d: d['cible'].astype(float))
        )
        
        # DataFrame pour les taux de défaut
        df_month = (
            df_tmp
            .assign(month=lambda d: d['DATDELHIS_Mm0'].dt.to_period('M').dt.to_timestamp())
            .groupby(['month', var])['cible']
            .agg(rate='mean', positives='sum', n='count')
            .reset_index()
            .assign(rate_pct=lambda d: d['rate'] * 100)  # Conversion en pourcentage
        )
        
        # DataFrame pour les fréquences des catégories
        df_cat = (
            df_tmp
            .assign(month=lambda d: d['DATDELHIS_Mm0'].dt.to_period('M').dt.to_timestamp())
            .groupby(['month', var]).size().reset_index(name='n')
            .assign(frequence=lambda d: d['n'] / d.groupby('month')['n'].transform('sum') * 100)
        )
        
        # Création de la figure avec deux subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Premier graphique : Taux de défaut
        sns.lineplot(data=df_month, x='month', y='rate_pct', hue=var, 
                    marker='o', linewidth=2, markersize=6, ax=ax1)
        ax1.axhline(y=self.mean_y, color='purple', linestyle='--')
        ax1.set_title(self.data_type + ': Évolution du taux de cible par catégorie ' + var, fontsize=14, fontweight='bold')
        ax1.set_xlabel('Mois', fontsize=12)
        ax1.set_ylabel('Taux de défaut (%)', fontsize=12)
        ax1.legend(title=var)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        
        # Deuxième graphique : Répartition des catégories
        sns.lineplot(data=df_cat, x='month', y='frequence', hue=var, 
                    marker='o', linewidth=2, markersize=6, ax=ax2)
        ax2.set_title(self.data_type + ': Évolution de la répartition : ' + var, fontsize=14, fontweight='bold')
        ax2.set_xlabel('Mois', fontsize=12)
        ax2.set_ylabel('Répartition (%)', fontsize=12)
        ax2.legend(title=var)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
    def correlation_with_target(self,VAR=None):
        if not VAR : 
            VAR = self.X_train.columns.tolist()
            VAR.remove('cible')
        if 'DATDELHIS_Mm0' in VAR:
            VAR.remove('DATDELHIS_Mm0')
        
        # colonnes catégorielles à tester (utilise cat_cols_in_X si défini sinon détecte automatiquement)
        cat_cols_in_X = VAR
        results = []
        for col in cat_cols_in_X:
            # tableau de contingence entre la variable catégorielle et la cible
            ct = pd.crosstab(self.X_train[col], self.y_train)
            n = ct.values.sum()
            # si tableau trop petit, on ne calcule pas
            if n == 0 or ct.shape[0] < 2 or ct.shape[1] < 2:
                results.append({
                    "variable": col,
                    "n": n,
                    "cramers_v": np.nan,
                    "tschuprow_t": np.nan,
                    "chi2_pvalue": np.nan
                })
                continue

            chi2, p, dof, expected = chi2_contingency(ct, correction=False)
            r, k = ct.shape

            # V de Cramer
            denom_v = n * (min(r - 1, k - 1))
            cramers_v = np.sqrt(chi2 / denom_v) if denom_v > 0 else np.nan

            # T de Tschuprow
            denom_t = n * np.sqrt((r - 1) * (k - 1))
            tschuprows_t = np.sqrt(chi2 / denom_t) if denom_t > 0 else np.nan

            results.append({
                "variable": col,
                "n": n,
                "cramers_v": cramers_v,
                "tschuprow_t": tschuprows_t,
                "chi2_pvalue": p
            })

        # DataFrame trié par V de Cramer décroissant
        assoc_cat = pd.DataFrame(results).sort_values("cramers_v", ascending=False).reset_index(drop=True)

        # affichage arrondi pour lisibilité
        assoc_cat[["cramers_v", "tschuprow_t", "chi2_pvalue"]] = assoc_cat[["cramers_v", "tschuprow_t", "chi2_pvalue"]].round(4)
        assoc_cat = assoc_cat.join(self.dico[['Signification']], how='left', on='variable')
        return assoc_cat


    def all_correlation_categorical(self,VAR=None):
        if not VAR : 
            VAR = self.X_train.columns.tolist()
        if 'DATDELHIS_Mm0' in VAR:
            VAR.remove('DATDELHIS_Mm0')

        X_quali2 = self.X_train
        vars_to_use = list({"cible"} | set(VAR))
        m = len(vars_to_use)

        # matrice vide
        cramer_mat = pd.DataFrame(np.nan, index=vars_to_use, columns=vars_to_use, dtype=float)

        # calcul pairwise (symétrique)
        from itertools import combinations
        for i, j in combinations(range(m), 2):
            a = vars_to_use[i]
            b = vars_to_use[j]
            v = self.cramers_v(X_quali2[a], X_quali2[b])
            cramer_mat.at[a, b] = v
            cramer_mat.at[b, a] = v

        # diagonale = 1
        np.fill_diagonal(cramer_mat.values, 1)

        # Trier par corrélation avec cible
        cramer_mat = cramer_mat.sort_values('cible', ascending=False)
        cramer_mat = cramer_mat[cramer_mat.index]
        # Garder seulement les 30 premières lignes & colonnes
        if m>30:
            cramer_ma = cramer_mat.iloc[:30, :30]
        else:
            cramer_ma = cramer_mat
        #print(f"Affichage de la matrice de corrélation Cramér's V pour les {cramer_ma.shape[0]} premières variables les plus corrélées avec la cible.")
        # Heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(cramer_ma, dtype=bool))
        sns.heatmap(
            cramer_ma, mask=mask, cmap="coolwarm", vmin=0, vmax=1,
            annot=True, fmt=".2f", linewidths=0.5, cbar_kws={"shrink": .8}
        )

        plt.title("Top 30 corrélations Cramér’s V avec la cible")
        plt.tight_layout()
        plt.show()
        return cramer_mat



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import chi2_contingency

class Binner:

    def __init__(self, X, cible_col='default_t_plus_1', date_col='obs_year'):
        self.X = X
        self.cible_col = cible_col
        self.date_col = date_col

    def compute_psi(self, ref_dist, cur_dist, eps=1e-6):
        """
        Calcule le PSI entre deux distributions
        """
        psi = np.sum(
            (ref_dist - cur_dist) *
            np.log((ref_dist + eps) / (cur_dist + eps))
        )
        return psi

    def plot_bin_stability_over_time(
        self,
        var_binned,
        ref_period=None,
        min_obs=1,
        min_pop=0.05,
        mask=None,
    ):
        """
        Visualise l'évolution dans le temps :
        - des volumes par bin
        - des taux de défaut par bin

        Parameters
        ----------
        var_binned : str
            Variable discrétisée (bins)
        freq : str
            'Y' = annuel, 'Q' = trimestriel, 'M' = mensuel
        min_obs : int
            Seuil minimum d'observations pour afficher le DR
        """
        # --------------------
        # AGGREGATION DES DONNÉES
        # --------------------
        if mask is not None:
            X = self.X[mask]
        else:
            X = self.X
        df = pd.DataFrame({
            'bin': X[var_binned],
            'target': X[self.cible_col],
            'date': X[self.date_col]
        })

        df['period'] = df['date']#.astype(str)

        agg = (
            df
            .groupby(['period', 'bin'])
            .agg(
                n_obs=('target', 'count'),
                n_defaults=('target', 'sum')
            )
            .reset_index()
        )

        agg['default_rate'] = agg['n_defaults'] / agg['n_obs']
        # Total observations par période
        agg['total_period'] = agg.groupby('period')['n_obs'].transform('sum')

        # Taux de volume par bin
        agg['pct_obs'] = agg['n_obs'] / agg['total_period']

        # --------------------
        # PSI CALCULATION
        # --------------------
        psi_values = []

        if ref_period is None:
            ref_period = agg['period'].min()

        ref_dist = (
            agg[agg['period'] == ref_period]
            .set_index('bin')['n_obs']
        )
        ref_dist = ref_dist / ref_dist.sum()

        for p in sorted(agg['period'].unique()):
            cur_dist = (
                agg[agg['period'] == p]
                .set_index('bin')['n_obs']
                .reindex(ref_dist.index, fill_value=0)
            )
            cur_dist = cur_dist / cur_dist.sum()

            psi = self.compute_psi(ref_dist.values, cur_dist.values)
            psi_values.append({'period': p, 'psi': psi})

        psi_df = pd.DataFrame(psi_values)

                # --------------------
        # PLOT VOLUME (%) + PSI
        # --------------------
        fig, ax1 = plt.subplots(figsize=(13, 5))

        sns.lineplot(
            data=agg,
            x='period',
            y='pct_obs',
            hue='bin',
            marker='o',
            ax=ax1
        )
        ax1.axhline(
            min_pop,
            color='grey',
            linestyle='--',
            linewidth=1,
            label='Seuil 5 %'
        )
        ax1.set_ylabel("Share of population")
        ax1.set_xlabel("Period")
        ax1.yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_title("Evolution of population shares by class with PSI")

        ax2 = ax1.twinx()
        ax2.plot(
            psi_df['period'],
            psi_df['psi'],
            color='black',
            linestyle='--',
            marker='s',
            label='PSI'
        )

        ax2.axhline(0.10, color='orange', linestyle=':')
        ax2.axhline(0.25, color='red', linestyle=':')
        ax2.set_ylabel("PSI")

        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(
            lines + lines2,
            labels + labels2,
            bbox_to_anchor=(1.02, 1),
            loc='upper left'
        )

        plt.tight_layout()
        plt.show()


        # --------------------
        # PLOT DEFAULT RATE
        # --------------------
        plt.figure(figsize=(13, 5))
        sns.lineplot(
            data=agg[agg['n_obs'] >= min_obs],
            x='period',
            y='default_rate',
            hue='bin',
            marker='o'
        )
        plt.gca().yaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
        plt.axhline(df['target'].mean(), color='black', linestyle='--', label='DR global')
        plt.title("Evolution of default rate by class")
        plt.ylabel("Default rate (%)")
        plt.xlabel("Period")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

    def plot_categorical_distribution(self,var,mask=None):
        if mask is not None:
            X = self.X[mask]
        else:
            X = self.X
        table = pd.crosstab(X[var], X[self.cible_col], normalize='index') * 100
        table = table.reset_index().melt(id_vars=var, var_name=self.cible_col, value_name='percentage')
        cramers_v,_ = self.v_cramer_t_tschuprow(X[var])

        plt.figure(figsize=(8,5))
        ax = sns.barplot(x=var, y='percentage', hue=self.cible_col, data=table, palette='Set2')
        plt.title("Default distribution (%) by categorical variable")
        # Ajouter les valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=8)
        plt.suptitle(f"V Cramer: {cramers_v:.3f}", y=1.02, fontsize=10)  
        plt.show()
    
    def v_cramer_t_tschuprow(self,var_quant):
        ct = pd.crosstab(var_quant, self.X[self.cible_col])
        n = ct.values.sum()

        chi2, p, dof, expected = chi2_contingency(ct, correction=False)
        r, k = ct.shape

        # V de Cramer
        denom_v = n * (min(r - 1, k - 1))
        cramers_v = np.sqrt(chi2 / denom_v) if denom_v > 0 else np.nan

        # T de Tschuprow
        denom_t = n * np.sqrt((r - 1) * (k - 1))
        tschuprows_t = np.sqrt(chi2 / denom_t) if denom_t > 0 else np.nan
        return cramers_v, tschuprows_t

    def discretise_with_manual_thresholds(
        self,
        var_quant,
        thresholds,
        labels=None,
        include_lowest=True,
        missing_label='Missing'
    ):
        """
        Discrétisation avec seuils fournis manuellement
        (IRB compliant – train/test/OOT)
        """

        thresholds = [-np.inf] + thresholds + [np.inf]

        if sorted(thresholds) != thresholds:
            raise ValueError("Les seuils doivent être strictement croissants")

        missing_mask = self.X[var_quant].isna()

        binned = pd.cut(
            self.X.loc[~missing_mask, var_quant],
            bins=thresholds,
            labels=labels,
            include_lowest=include_lowest,
            right=False
        )

        result = pd.Series(index=self.X.index, dtype='object')
        result.loc[~missing_mask] = binned.astype(str)
        result.loc[missing_mask] = missing_label
        self.X[var_quant] = result
        return result

    
    def merge_modalities(
        self,
        col,
        mapping,
    ):
        """
        Fusionne des modalités d'une variable catégorielle
        à partir d'un dictionnaire ancien -> nouveau
        
        Parameters
        ----------
        X : pd.DataFrame
            DataFrame contenant la variable
        col : str
            Nom de la variable catégorielle
        mapping : dict
            {ancienne_modalité: nouvelle_modalité}
        ordered : bool
            Indique si la variable est ordonnée

        Returns
        -------
        pd.Series
            Variable catégorielle fusionnée
        """

        # Mapping des valeurs
        merged = self.X[col].astype('str')
        merged = merged.apply(lambda x: mapping.get(x, x))

        # Contrôle des modalités non mappées
        if merged.isna().any():
            unmapped = self.X.loc[merged.isna(), col].unique()
            raise ValueError(f"Modalités non mappées détectées: {unmapped}")

        self.X[col] = merged.astype('category')



    def extract_binning_thresholds(self, X_binned):
        """
        Extrait les seuils de discrétisation depuis des catégories
        de type string + classe 'Missing'
        """

        thresholds = {}

        for var in X_binned.columns:
            vars = X_binned[var].astype('category')           

            categories = vars.cat.categories.astype(str)

            has_missing = 'Missing' in categories

            bounds = set()

            for cat in categories:
                if cat == 'Missing':
                    continue

                # Parsing du type: [-inf, 1.05) / [1.05, 3.05) / [3.05, inf)
                match = re.match(r'[\[\(]([^,]+),\s*([^\]\)]+)[\]\)]', cat)
                if match:
                    left, right = match.groups()
                    if left != '-inf':
                        bounds.add(float(left))
                    if right != 'inf':
                        bounds.add(float(right))

            thresholds[var] = {
                'cuts': sorted(bounds),
                'has_missing': has_missing
            }

        return thresholds


    def apply_binning_thresholds(
        self,
        X_binned,
        X_new,
        suffix=''
    ):
        """
        Applique les seuils de discrétisation à un nouveau DataFrame
        avec gestion explicite des valeurs manquantes
        """

        X_binned_new = pd.DataFrame(index=X_new.index)
        thresholds = self.extract_binning_thresholds(X_binned)

        for var, info in thresholds.items():

            cuts = [-np.inf] + info['cuts'] + [np.inf]
            vars = X_new[var]

            # Discrétisation des valeurs non manquantes
            X_binned_new[var + suffix] = pd.cut(
                vars,
                bins=cuts,
                include_lowest=True,
                right=False
            )
            #print(X_binned_new.loc[:, var + suffix].head(10))
            X_binned_new[var + suffix] = X_binned_new[var + suffix].astype('category')
            
            X_binned_new[var + suffix] = (
                    X_binned_new[var + suffix]
                    .cat.add_categories(['Missing'])
                    .fillna('Missing')
                )
            #print(X_binned_new.loc[:, var + suffix].head(10))
        return X_binned_new

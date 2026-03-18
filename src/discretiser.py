import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import seaborn as sns

tree = DecisionTreeClassifier(
            criterion='entropy',
            min_samples_leaf= 0.05,
            max_depth=2,
            random_state=42
        )

class Discretiser:

    def __init__(self, X_train, y_train, tree = tree, dico=None, date_col='obs_year', description_col='description'):
        self.tree = tree
        self.X_train = X_train
        self.y_train = y_train
        self.dico = dico
        self.date_col = date_col
        self.description_col = description_col


    


    def fit_cart(self, var,plot=False):
        X = np.array(self.X_train[var]).reshape(-1, 1)
        y = np.array(self.y_train)
        self.tree.fit(X,y)
        if plot:
            self.plot_tree_cart(var)
    def plot_tree_cart(self,var):
        plt.figure(figsize=(20,10))
        plot_tree(self.tree, feature_names=[var], class_names=['0','1'], filled=True, rounded=True)
        plt.show()

    def discretiser_variable(self, var_quant, labels=None, include_lowest=True):
        self.fit_cart(var_quant)
        data = self.X_train.copy()

        missing_mask = data[var_quant].isna()

        seuils = (
            sorted(set(self.tree.tree_.threshold[self.tree.tree_.threshold > -2]))
        )
        if -np.inf not in seuils:
            seuils = [-np.inf] + seuils
        if np.inf not in seuils:
            seuils = seuils + [np.inf]
        

        var_discrete = pd.cut(
            data.loc[~missing_mask, var_quant],
            bins=seuils,
            labels=labels,
            include_lowest=include_lowest,
            right=False
        )

        result = pd.Series(index=data.index, dtype='object')
        result.loc[~missing_mask] = var_discrete.astype(str)
        result.loc[missing_mask] = 'Missing'

        return result

    
    def v_cramer_t_tschuprow(self,var_quant):
        ct = pd.crosstab(var_quant, self.y_train)
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


    def plot_categorical_distribution(self,var_quant):
        X = pd.DataFrame({'categorie': var_quant, 'cible': self.y_train})
        table = pd.crosstab(X["categorie"], X['cible'], normalize='index') * 100
        table = table.reset_index().melt(id_vars='categorie', var_name='cible', value_name='pourcentage')
        cramers_v,_ = self.v_cramer_t_tschuprow(var_quant)

        plt.figure(figsize=(8,5))
        ax = sns.barplot(x='categorie', y='pourcentage', hue='cible', data=table, palette='Set2')
        plt.title("Distribution (%) de la cible selon la variable catégorielle")

        # Ajouter les valeurs sur les barres
        for container in ax.containers:
            ax.bar_label(container, fmt="%.1f%%", label_type="edge", fontsize=8)
        plt.suptitle(f"V de Cramer: {cramers_v:.3f}", y=1.02, fontsize=10)  
        plt.show()
    
    def plot_boxplot_with_strip(self,var_quant):
        X = pd.DataFrame({'quanti': var_quant, 'cible': self.y_train})
        plt.figure(figsize=(8,5))
        sns.boxplot(x='cible', y='quanti', data=X, showfliers=False, palette='Set3')
        sns.stripplot(x='cible', y='quanti', data=X, hue='cible', alpha=0.4, jitter=True)
        plt.title("Boxplot avec nuage de points")
        plt.show()

    def check_monotonicity(self, var_binned):
        df = pd.DataFrame({
            'bin': var_binned,
            'target': self.y_train
        })
        dr = df.groupby('bin')['target'].mean().values
        return np.all(np.diff(dr) >= 0) or np.all(np.diff(dr) <= 0)
   
    def compute_woe_iv(self, var_binned, eps=1e-6):
        df = pd.DataFrame({
            'bin': var_binned,
            'target': self.y_train
        })

        grouped = df.groupby('bin')['target'].agg(
            total='count',
            bads='sum'
        )
        grouped['goods'] = grouped['total'] - grouped['bads']

        total_goods = grouped['goods'].sum()
        total_bads = grouped['bads'].sum()

        grouped['dist_goods'] = grouped['goods'] / total_goods
        grouped['dist_bads'] = grouped['bads'] / total_bads

        # WOE avec protection division par zéro
        grouped['woe'] = np.log(
            (grouped['dist_goods'] + eps) /
            (grouped['dist_bads'] + eps)
        )

        grouped['iv_bin'] = (
            (grouped['dist_goods'] - grouped['dist_bads']) *
            grouped['woe']
        )

        iv = grouped['iv_bin'].sum()

        return iv, grouped.reset_index()


    def discretiser_all_variables(self, quantitative_cols):
        discretized_vars = {}
        results = {}

        for var in quantitative_cols:
            discretized_var = self.discretiser_variable(var)
            discretized_vars[var] = discretized_var

            cramers_v, tschuprows_t = self.v_cramer_t_tschuprow(discretized_var)
            iv, _ = self.compute_woe_iv(discretized_var)

            results[var] = {
                'variable': var,
                'cramers_v': cramers_v,
                'tschuprow_t': tschuprows_t,
                'iv': iv
            }

        corr = pd.DataFrame(results).T
        if self.dico is not None:
            corr = corr.join(self.dico[[self.description_col]], how='left', on='variable')

        corr = corr.sort_values(by='iv', ascending=False)
        return pd.DataFrame(discretized_vars), corr

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
        date_col,
        cible,
        freq='Y',
        min_obs=30
    ):
        """
        Visualise l'évolution dans le temps :
        - des volumes par bin
        - des taux de défaut par bin

        Parameters
        ----------
        var_binned : pd.Series
            Variable discrétisée (bins)
        date_col : pd.Series
            Variable date
        freq : str
            'Y' = annuel, 'Q' = trimestriel, 'M' = mensuel
        min_obs : int
            Seuil minimum d'observations pour afficher le DR
        """
        # --------------------
        # AGGREGATION DES DONNÉES
        # --------------------

        df = pd.DataFrame({
            'bin': var_binned,
            'target': self.y_train,
            'date': pd.to_datetime(date_col)
        })

        df['period'] = df['date'].dt.to_period(freq).astype(str)

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
        # PLOT VOLUME + PSI
        # --------------------
        fig, ax1 = plt.subplots(figsize=(13, 5))

        sns.lineplot(
            data=agg,
            x='period',
            y='n_obs',
            hue='bin',
            marker='o',
            ax=ax1
        )

        ax1.set_ylabel("Nombre d'observations")
        ax1.set_xlabel("Période")
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_title("Évolution des volumes par classe avec PSI")

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
        plt.axhline(df['target'].mean(), color='black', linestyle='--', label='DR global')
        plt.title("Évolution du taux de défaut par classe")
        plt.ylabel("Taux de défaut")
        plt.xlabel("Période")
        plt.xticks(rotation=45)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.show()


    def discretise_with_manual_thresholds(
        self,
        X,
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

        data = X.copy()
        thresholds = [-np.inf] + thresholds + [np.inf]

        if sorted(thresholds) != thresholds:
            raise ValueError("Les seuils doivent être strictement croissants")

        missing_mask = data[var_quant].isna()

        binned = pd.cut(
            data.loc[~missing_mask, var_quant],
            bins=thresholds,
            labels=labels,
            include_lowest=include_lowest,
            right=False
        )

        result = pd.Series(index=data.index, dtype='object')
        result.loc[~missing_mask] = binned.astype(str)
        result.loc[missing_mask] = missing_label

        return result

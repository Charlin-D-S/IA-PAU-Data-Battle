
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from itertools import combinations
import seaborn as sns
from matplotlib.ticker import FuncFormatter


from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.metrics import brier_score_loss

from sklearn.calibration import calibration_curve

from sklearn.tree import DecisionTreeClassifier, plot_tree


class Analyser:

    def __init__(self,X,probas_col='probas',points_col='points',target_col='default_t_plus_1',CHR_col='CHR'):
        self.X = X
        self.probas_col = probas_col
        self.points_col = points_col
        self.target_col = target_col
        self.y_true = self.X[self.target_col].astype(int).values
        self.y_score = self.X[self.probas_col].astype(float).values
        self.CHR_col = CHR_col

    def plot_ROC_AUC(self, plot=True, _return=False):
        y_true = self.y_true
        y_score = self.y_score
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        auc = roc_auc_score(y_true, y_score)
        # Interprétation de l'AUC :
        if auc >= 0.90:
            interpretation = "Excellent — séparation très forte; rare pour problèmes difficiles."
        elif auc >= 0.80:
            interpretation = "Très bon — bonne capacité discriminative pour la plupart des usages."
        elif auc >= 0.70:
            interpretation = "Acceptable / Correct — utile mais améliorable, vérifier calibration et utilité métier."
        elif auc >= 0.60:
            interpretation = "Faible — discrimination limitée; probablement insuffisant en production sans contraintes métier fortes."
        else:
            interpretation = "Mauvais / proche du hasard — peu de valeur discriminative; reconsidérer features ou modèle."

        gini = 2 * auc - 1
        print(f"GINI : {gini:.4f} --> {interpretation}")
        if plot:
            plt.figure(figsize=(8,8))
            plt.plot(fpr, tpr, color="C0", lw=2, label=f"ROC (AUC = {auc:.2%})")
            plt.plot([0,1],[0,1], color="grey", lw=1, linestyle="--", label="Alea")
            plt.fill_between(fpr, tpr, alpha=0.15, color="C0")
            plt.xlim([-0.01, 1.01])
            plt.ylim([-0.01, 1.01])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("Courbe ROC")
            plt.legend(loc="lower right")
            plt.grid(alpha=0.3)
            plt.show()
        
        if _return:
            return fpr, tpr, thresholds, auc
        
    def plot_PR_AUC(self, plot=True, _return=False):
        y_true = self.y_true
        y_score = self.y_score
        precision, recall, thresholds = precision_recall_curve(y_true, y_score)
        if plot:

            plt.figure(figsize=(7,6))
            plt.plot(recall, precision, color="C0", lw=2, label=f"PR curve (AP = {average_precision_score(y_true, y_score):.2%})")
            plt.hlines(y_true.mean(), 0, 1, colors="grey", linestyles="--", label=f"Baseline (prevalence = {y_true.mean():.2%})")
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title("Courbe Precision-Recall")
            plt.legend(loc="lower left")
            plt.grid(alpha=0.3)
            plt.show()

        auc_pr = average_precision_score(y_true, y_score)
        print(f"AUC PR: {auc_pr:.2%}")
        prevalence = y_true.mean()
        performance_ratio = auc_pr / prevalence if prevalence > 0 else np.nan
        print(f"Ratio de performance: {performance_ratio:.2f}")

        if performance_ratio <= 1:
            interpretation = "Le modèle performe en dessous de la prévalence; il n'apporte pas de valeur ajoutée."
        elif performance_ratio <= 3:
            interpretation = "Amélioration utile selon le contexte opérationnel."
        else:
            interpretation = "Fort gain relatif pour des actions ciblées."
        print(f"Interprétation: {interpretation}")
        if _return:
            return precision, recall, thresholds, auc_pr,performance_ratio
        
    def ks_stat(self):
        df = pd.DataFrame({"y_true": self.y_true, "y_score": self.y_score})
        df = df.sort_values('y_score').reset_index(drop=True)
        
        total_pos = (df['y_true'] == 1).sum()
        total_neg = (df['y_true'] == 0).sum()    
        
        df["cum_pos"] = (df['y_true'] == 1).cumsum() / total_pos
        df["cum_neg"] = (df['y_true'] == 0).cumsum() / total_neg
        
        df['ks'] = np.abs(df["cum_pos"]-df["cum_neg"])
        
        ks_index = df['ks'].idxmax()
        ks_value = df.loc[ks_index, 'ks']
        ks_point = df.loc[ks_index, "y_score"]
        
        return ks_value, ks_point, df
    

    def plot_ks(self):
        ks_value, ks_point, df = self.ks_stat()

        plt.plot(df["y_score"], df['cum_pos'], label="CDF positif")
        plt.plot(df["y_score"], df['cum_neg'], label='CDF négatif')
        
        plt.plot(df["y_score"], df["ks"], label='Différence cumulée')
        
        # Point KS
        ks_index = df['ks'].idxmax()
        plt.scatter(df.loc[ks_index, "y_score"], df.loc[ks_index, "ks"], color='red')
        plt.text(
            df.loc[ks_index, "y_score"],
            df.loc[ks_index, "ks"],
            f" KS={ks_value:.1%}\nScore={ks_point:.2f}",
            verticalalignment="bottom"
        )
        
        plt.title('KS plot')
        plt.xlabel('Score')
        plt.ylabel("Valeus cumulées")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        if ks_value <= 0.4:
            print("Discrimination très faible (KS <= 40%)")
        elif ks_value < 0.7:
            print("Discrimination correcte (40% < KS < 70%)")
        else:
            print("Discrimination forte (KS > 70%)")

    
    def pAUC_ROC(self,percent, segment='top'):
        y_score = self.y_score
        y_true = self.y_true
        segment = segment.lower()
        if segment not in ['top', 'bottom']:
            raise ValueError("segment doit valoir 'top' ou 'bottom'")
            
        if not (0 <= percent <= 1):
            raise ValueError("percent doit etre un pourcentage entre 0 et 1")
            
            
        df = pd.DataFrame({"y_true": y_true, "y_score": y_score})
        df = df.sort_values('y_score', ascending=(segment=='bottom')).reset_index(drop=True)
        
        n = len(df)
        k = int(np.ceil(percent*n))
        df_seg = df.iloc[:k]
        
        if df_seg['y_true'].nunique() < 2:
            return df_seg, np.nan
        
        return df_seg, roc_auc_score(df_seg['y_true'], df_seg['y_score'])
    
    def plot_ROC_AUC_global_partial(self, percents, segment='top'):
        fpr_full, tpr_full, _, auc_full = self.plot_ROC_AUC(plot=False, _return=True)
        
        plt.figure(figsize=(10, 6))
        plt.plot( fpr_full, tpr_full, label=f'ROC global (AUC = {auc_full:.2%})')

        
        for percent in percents:
        
            df_seg, auc_seg = self.pAUC_ROC(percent, segment='top')
            fpr_seg, tpr_seg, _ = roc_curve(df_seg['y_true'], df_seg['y_score'])
        
            plt.plot( fpr_seg,  tpr_seg,  label=f'ROC {segment} {percent:.2%} (AUC = {auc_seg:.2%})')   
        
        
        plt.plot([0,1],[0,1], color="grey", lw=1, linestyle="--", label="Alea")
        plt.xlim([-0.01, 1.01])
        plt.ylim([-0.01, 1.01])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"Courbe ROC globale et tronquée")
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        
        plt.legend()
        plt.show()

    def reliability_diagram(self, n_bins=20, CHR='', _cible='', _proba_theorique = ''):
        _df=self.X
        if CHR=='':
            prob_true, prob_pred = calibration_curve(self.y_true, self.y_score, n_bins=n_bins)
            
        else:
            if _df is None or _cible == '' or _proba_theorique == '':
                raise ValueError()
                
            calibration_df = _df.groupby(self.CHR_col).agg(
                y_true_mean = (_cible, 'mean'),
                pred_proba_mean=(_proba_theorique, 'mean'),
                n=('cible', 'count')
            ).reset_index()

            prob_true, prob_pred = calibration_df['y_true_mean'], calibration_df['pred_proba_mean']
            
            
        if CHR=='':
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), sharex=True, constrained_layout=True)
        else:
            fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, figsize=(7, 7), constrained_layout=True)

            
        ax1.plot(prob_pred, prob_true, marker='o', linewidth=2, label=f"Model (Brier={brier_score_loss(self.y_true, self.y_score):.2%})")
        ax1.plot([0,1],[0,1], linestyle='--', color='gray', label='Perfect calibration')
        ax1.set_xlabel("Mean predicted probability")
        ax1.set_ylabel("Fraction of positives")
        ax1.set_title("Reliability diagram (calibration curve)")
        ax1.legend(loc="best")
        ax1.grid(alpha=0.3)
        
        if CHR=='':
            ax2.hist(self.y_score, bins=n_bins, color='C0', edgecolor='k', alpha=0.7)
        else:
            counts = _df[self.CHR_col].value_counts().sort_index()
            ax2.bar(counts.index.astype(str), counts.values, color='C0', edgecolor='k', alpha=0.7)
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Count")
        ax2.set_title("Histogram of predicted probabilities")
        ax2.grid(alpha=0.2)

        N = len(self.y_score) if len(self.y_score) > 0 else 1
        ax_perc = ax2.twinx()

        primary_yticks = ax2.get_yticks()
        secondary_yticks = primary_yticks / N 

        ax_perc.set_yticks(primary_yticks)  # positionner aux mêmes valeurs numériques que l'axe gauche
        ax_perc.set_ylim(ax2.get_ylim())  # garder mêmes limites pour alignement visuel
        ax_perc.set_ylabel("Share of total (%)")
        ax_perc.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{100.0 * (y / N):.0f}%"))
        
        plt.show()
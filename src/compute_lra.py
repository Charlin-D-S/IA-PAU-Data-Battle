# ============================================================
# IMPORTS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm  # barre de progression pour les simulations


# ============================================================
# BREAKS DE CLASSES (bornes de PD pour rating)
# ============================================================

breaks = [
    np.float64(0.01035882738806523),
    0.03022480943039526,
    0.05755801819497361,
    0.10678483451203975,
    0.3687523924761468,
    0.5334739877176452
]


# ============================================================
# CLASSE PRINCIPALE : COMPUTELRA
# Objectif :
# Simuler des trajectoires de LRA via Monte Carlo
# pour quantifier l'incertitude due aux données manquantes
# ============================================================

class ComputeLRA:

    def __init__(
        self,
        df_with_missing,  # dataset contenant des valeurs manquantes
        df_clean,         # dataset complet (vérité terrain)
        cols_active,      # colonnes actives pour la simulation
        imputer,          # objet d'imputation (Monte Carlo, MICE, etc.)
        predictor,        # modèle de scoring
        lras=None,        # LRA déjà simulées (optionnel)
        quantile_step=200,
        breaks=breaks
    ):

        # ================================
        # STOCKAGE DES DONNÉES
        # ================================
        self.df_with_missing = df_with_missing
        self.df_clean = df_clean
        self.cols_active = cols_active

        # Séparation observations complètes / incomplètes
        self.df_without_missing = (
            self.df_with_missing[self.df_with_missing['nb_missings'] == 0]
            [self.cols_active]
            .copy()
        )

        self.df_to_impute = (
            self.df_with_missing[self.df_with_missing['nb_missings'] > 0]
            [self.cols_active]
            .copy()
        )

        self.imputer = imputer
        self.breaks = breaks
        self.predictor = predictor

        # Copie destinée à recevoir les imputations
        self.df_imputed = self.df_to_impute.copy()

        self.lras = lras
        self.quantile_step = quantile_step

        # LRA centrale observée (avec missings)
        self.lra_central = self.get_central_lra()

        # LRA de référence (dataset complet)
        self.lra_full = self.get_full_lra()

        # Trajectoire des quantiles simulés
        self.lra_quantiles = self.get_quantile_dist()

        # Variables explicatives utilisées par le modèle
        self.vars = predictor.VAR

    # ============================================================
    # DISTRIBUTION DES QUANTILES AU FIL DES SIMULATIONS
    # ============================================================

    def get_quantile_dist(self):

        quantiles = self.lra_central
        n = 1
        index = [0]

        # Si simulations déjà existantes
        if self.lras is not None:
            n = self.lras.shape[0]

            for i in list(range(self.quantile_step, n, self.quantile_step)):
                q = np.quantile(self.lras.values[:i], 0.99, axis=0)
                quantiles = np.vstack([quantiles, q])
                index.append(i)

            # Quantile final
            quantiles = np.vstack([
                quantiles,
                np.quantile(self.lras.values, 0.99, axis=0)
            ])
            index.append(n)

        quantiles = pd.DataFrame(
            quantiles,
            columns=[f"Class_{c}" for c in ['AA', 'A', 'BB', 'B', 'C']]
        )
        quantiles['index'] = index

        return quantiles

    # ============================================================
    # LRA CENTRALE (avec données manquantes)
    # ============================================================

    def get_central_lra(self, print_value=False):

        lra = (
            self.df_with_missing
            .groupby(['classe'], observed=True)['default_t_plus_1']
            .mean()
            .sort_index()
        )

        if print_value:
            print("LRA centrale observée par classe :")
            print(lra)

        return lra

    # ============================================================
    # LRA RÉFÉRENCE (dataset complet)
    # ============================================================

    def get_full_lra(self, print_value=False):

        lra_full = (
            self.df_clean
            .groupby(['classe'], observed=True)['default_t_plus_1']
            .mean()
            .sort_index()
        )

        if print_value:
            print("LRA observée par classe, données complètes :")
            print(lra_full)

        return lra_full

    # ============================================================
    # IMPUTATION + RECALCUL DES PROBABILITÉS
    # ============================================================

    def impute_with_distribution(self):

        # Imputation des variables explicatives manquantes
        df = self.imputer.impute(self.df_to_impute[self.vars])

        # Recalcul des probabilités via modèle de scoring
        probas = self.predictor.predire_probas(df)

        self.df_imputed['probas'] = probas

        # Reclassement en rating classes
        self.df_imputed['classe'] = pd.cut(
            self.df_imputed['probas'],
            bins=self.breaks,
            labels=[1, 2, 3, 4, 5],
            include_lowest=True
        )

    # ============================================================
    # RECALCUL DU LRA APRÈS IMPUTATION
    # ============================================================

    def get_simulated_LRA(self):

        # Concaténation données complètes + imputées
        df_all = pd.concat(
            [self.df_without_missing, self.df_imputed],
            ignore_index=True
        )

        lra = (
            df_all
            .groupby(['classe'], observed=True)['default_t_plus_1']
            .mean()
            .sort_index()
        )

        return np.array(lra)

    # ============================================================
    # SAUVEGARDE
    # ============================================================

    def save_lras(self, save_path, lras):

        if save_path:
            try:
                lras.to_csv(save_path, index=False)
            except Exception as e:
                print(f'An error occured : \n {e}')

    # ============================================================
    # TEST DE CONVERGENCE DU QUANTILE 99%
    # ============================================================

    def check_convergence(self, lras, eps):

        q = np.quantile(lras, 0.99, axis=0)

        n = self.lra_quantiles.shape[0]
        past_q = self.lra_quantiles.iloc[n-1, :5].values

        # Nombre de classes convergées
        som = np.sum(np.abs(past_q - q) < eps)

        m = lras.shape[0]

        q_df = pd.DataFrame({
            "Class_AA": [q[0]],
            "Class_A": [q[1]],
            "Class_BB": [q[2]],
            "Class_B": [q[3]],
            "Class_C": [q[4]],
            "index": [m-1]
        })

        self.lra_quantiles = pd.concat(
            [self.lra_quantiles, q_df],
            ignore_index=True
        )

        # Convergence si toutes les classes stabilisées
        if som >= 5:
            return True, np.abs(past_q - q) < eps
        else:
            return False, np.abs(past_q - q) < eps

    # ============================================================
    # SIMULATION MONTE CARLO PRINCIPALE
    # ============================================================

    def simulate_scores(
        self,
        n_sim,
        verbose=True,
        eps=1e-4,
        check_convergence=True,
        save_path=None
    ):

        # Initialisation
        if self.lras is None:
            lras = self.lra_central.values
        else:
            lras = self.lras.values

        for i in tqdm(range(n_sim), desc="Simulations"):

            # Imputation
            self.impute_with_distribution()

            # Recalcul LRA
            lra = self.get_simulated_LRA()

            lras = np.vstack([lras, lra])

            # Vérification convergence périodique
            if verbose and (i + 1) % self.quantile_step == 0:
                converge, som = self.check_convergence(lras, eps)
                print(f'After {i+1} simulations, Convergence ? {som}')
                if converge and check_convergence:
                    break

        self.lras = pd.DataFrame(
            lras,
            columns=[f"Class_{c}" for c in ['AA', 'A', 'BB', 'B', 'C']]
        )

        self.save_lras(save_path, self.lras)

    # ============================================================
    # CALCUL MoC PAR CLASSE
    # ============================================================

    def moc_par_classe(self, alpha=0.99, percent=True):

        moc = pd.Series(dtype=float)
        df_LRA = self.lras
        q_ref = self.lra_central

        for i, col in enumerate(df_LRA.columns, start=1):
            q_alpha = df_LRA[col].quantile(alpha)
            moc.loc[i] = (q_alpha - q_ref.loc[i])

        if percent:
            moc = moc / q_ref

        return pd.Series(moc, name=f"MoC_{alpha*100}%")

        
    #######################################################################
    def plot_quantile_convergence(self):
        plt.figure(figsize=(12, 8))
        

        for i, col in enumerate([f"Class_{c}" for c in ['AA','A','BB','B','C']], 1):
            plt.subplot(2, 3, i)

            # series = self.lras[col].values
            path = self.lra_quantiles.tail(-1)

            plt.plot(path['index'], path[col]*100, linewidth=2)

            # Ligne horizontale = quantile final
            final_q = path.loc[path.shape[0]-1,col]*100
            plt.axhline(final_q, linestyle="--", linewidth=1, label="Final Q99",color = 'red')

            # # Ligne horizontale = LRA centrale
            # central = self.lra_central[i]*100
            # plt.axhline(central, linestyle="--", linewidth=1, label="Central LRA")

            plt.title(f"{col} – Q_99 convergence")
            plt.xlabel("Number of simulations")
            plt.ylabel("Quantile (%)")
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()    
    
####################################################################################
    def plot_lras(self):    
        plt.figure(figsize=(12, 8))
        lras = self.lras.tail(-1)
        for i, col in enumerate(self.lras.columns, 1):
            plt.subplot(2, 3, i)
            
            # Données en pourcentage
            data = lras[col] * 100
            
            # Statistiques
            mediane = np.median(data)
            std = np.std(data)
            quantile_99 = np.percentile(data, 99)

            central = self.lra_central[i] * 100
            full = self.lra_full[i] * 100
            # Histogramme
            plt.hist(data, bins=50, edgecolor="black", alpha=0.7)

            # Ligne LRA centrale
            plt.axvline(
                full,
                linestyle="--",
                linewidth=1,
                label="Reference LRA",
                color = 'green'

            )

            # Ligne quantile final
            plt.axvline(
                quantile_99,
                linestyle="-",
                linewidth=1,
                color = 'red',
                label="Q99 (final)"
            )

            # Texte statistique
            stats_text = (
                f'Median: {mediane:.2f}\n'
                f'Std: {std:.2f}\n'
                f'Q99: {quantile_99:.2f}\n'
                f'Central: {central:.2f}\n'
                f'Reference: {full:.2f}'
                
            )

            plt.text(
                0.02, 0.98,
                stats_text,
                transform=plt.gca().transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            )

            plt.title(col)
            plt.legend()
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()
    
    

#################################################################################
    def plot_lras_quantile(self, alphas = None):
        if not alphas:
            alphas = [0.95, 0.99, 0.9999]
        quantiles_df = pd.DataFrame(
            {
                alpha: self.lras.quantile(alpha)
                for alpha in alphas
            }
        )

        # index plus propre pour l'affichage
        quantiles_df.index = range(1, len(quantiles_df) + 1)
        quantiles_df[0] = self.lra_central.values
        quantiles_df[-1] = self.lra_full.values
        quantiles_df = quantiles_df[[-1,0] + alphas]

        plt.figure(figsize=(8, 5))

        for alpha in [-1,0] + alphas:
                    plt.plot(
                        quantiles_df.index,
                quantiles_df[alpha],
                marker="o",
                label= 'Central LRA' if alpha == 0 else ' Reference LRA' if alpha == -1 else f"Quantile {alpha*100}%"
            )

        plt.xlabel("Rating class")
        plt.ylabel("LRA")
        plt.title("Rating class LRA")
        plt.xticks(quantiles_df.index)
        plt.grid(True)
        plt.legend()

        plt.show()



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# # cols_active = VAR + [target,'nb_missings','probas','points','CHR','classe','LRA','beta_scores']
# # df_with_missing['beta_scores'] = np.log(1 / df_with_missing['probas'] - 1) - pred.const

# # df_without_missing = df_with_missing[df_with_missing['nb_missings'] == 0][cols_active].copy()
# # df_imputed = df_with_missing[df_with_missing['nb_missings'] > 0][cols_active].copy()


# # distribution = {
# #     'payment_regularity_score' : MultinomialDist1,
# #     'credit_utilization' : KDEDistribution1,
# #     'total_outstanding_to_income' : KDEDistribution2
# # }
# breaks = [np.float64(0.01035882738806523),
#             0.03022480943039526,
#             0.05755801819497361,
#             0.10678483451203975,
#             0.3687523924761468,
#             0.5334739877176452]

# class ComputeLRA:
#     def __init__(self,df_with_missing,df_clean, cols_active,imputer,
#                  predictor,lras=None, quantile_step = 200,
#                  breaks = breaks):
#         self.df_with_missing = df_with_missing
#         self.df_clean = df_clean
#         self.cols_active = cols_active
#         self.df_without_missing = self.df_with_missing[self.df_with_missing['nb_missings'] == 0][self.cols_active].copy()
#         self.df_to_impute = self.df_with_missing[self.df_with_missing['nb_missings'] > 0][self.cols_active].copy()
#         self.imputer = imputer
#         #self.distribution = imputer.distribution
#         self.breaks = breaks
#         self.predictor = predictor
#         self.df_imputed = self.df_to_impute.copy()
#         self.lras = lras
#         self.quantile_step = quantile_step
#         self.lra_central = self.get_central_lra()
#         self.lra_full = self.get_full_lra()
#         self.lra_quantiles= self.get_quantile_dist()
#         self.vars = predictor.VAR
#     #######################################################################
#     def get_quantile_dist(self):
        
#         quantiles = self.lra_central
#         n=1
#         index =[0]
#         if self.lras is not None :
#             n = self.lras.shape[0]
#             for i in list(range(self.quantile_step,n,self.quantile_step)):
#                 q = np.quantile(self.lras.values[:i],0.99,axis=0)
#                 quantiles = np.vstack([quantiles, q])
#                 index.append(i)
#             quantiles = np.vstack([quantiles, np.quantile(self.lras.values,0.99,axis=0)])
#             index.append(n)
#         quantiles = pd.DataFrame(quantiles, columns=[f"Class_{c}" for c in ['AA','A','BB','B','C']])
#         quantiles['index'] = index
#         #quantiles = quantiles.tail(-1)
#         return quantiles
    
#     def get_central_lra(self,print_value = False):
#         lra = self.df_with_missing.groupby(['classe'], observed=True)['default_t_plus_1'].mean().sort_index()
#         if print_value:
#             print("LRA centrale observée par classe :")
#             print(lra)
#         return lra
#     #########################################################################
#     def get_full_lra(self,print_value = False):
#         lra_full = self.df_clean.groupby(['classe'], observed=True)['default_t_plus_1'].mean().sort_index()
#         if print_value:
#             print("LRA observée par classe, données complètes :")
#             print(lra_full)
#         return lra_full

#     #############################################################################

#     def impute_with_distribution(self):
#         df = self.imputer.impute(self.df_to_impute[self.vars])
#         probas = self.predictor.predire_probas(df)
#         self.df_imputed['probas'] = probas
#         self.df_imputed['classe'] = pd.cut(
#             self.df_imputed['probas'],
#             bins=self.breaks,
#             labels=[1, 2, 3, 4, 5],
#             include_lowest=True
#             )
#     #############################################################################
    
#     ################################################################################
#     def get_simulated_LRA(self):
#         df_all = pd.concat([self.df_without_missing, self.df_imputed], ignore_index=True)
#         lra = (
#                     df_all
#                     .groupby(['classe'], observed=True)['default_t_plus_1']
#                     .mean()
#                 )
#         lra = lra.sort_index()
#         return np.array(lra)
    
#     def save_lras(self,save_path,lras):
#         if save_path:
#             try:    
#                 lras.to_csv(save_path,index = False)
#             except Exception as e:
#                 print(f'An error occured : \n {e}')
#     #############################################################################
#     def check_convergence(self, lras, eps):
#         #if len(self.lra_quantiles) !=0 :
#         q = np.quantile(lras,0.99,axis=0)
#         n = self.lra_quantiles.shape[0]
#         past_q = self.lra_quantiles.iloc[n-1,:5].values
#         som = np.sum(np.abs(past_q - q)< eps)
#         m = lras.shape[0]
#         q_df = pd.DataFrame({ "Class_AA" : [q[0]] , 
#                             "Class_A":[q[1]],
#                             "Class_BB":[q[2]], 
#                             "Class_B":[q[3]],
#                             "Class_C":[q[4]],
#                             "index": [m-1]})

#         self.lra_quantiles = pd.concat([self.lra_quantiles,q_df],ignore_index=True)
#         if som >=5:
#             return True,np.abs(past_q - q)< eps
#         else:
#             return False,np.abs(past_q - q)< eps
#     def simulate_scores(
#         self,
#         n_sim,
#         verbose=True,
#         eps = 1e-4,
#         check_convergence = True,
#         save_path = None#'..\data\lras_simulee_monte_carlo.csv'
#     ):
#         """
#         Simule les scores conditionnels via Monte Carlo en imputant les variables explicatives manquantes
#         à partir de leur distribution conditionnelle estimée (KDE, multinomial, etc.)
#         """
#         if self.lras is None:
#             lras = self.lra_central.values
#         else:
#             lras = self.lras.values

#         for i in tqdm(range(n_sim), desc="Simulations"):
                        
#             self.impute_with_distribution()
#             lra = self.get_simulated_LRA()
#             lras = np.vstack([lras, lra])

#             if verbose and (i + 1) % self.quantile_step == 0:
#                 #print(f"  - Simulation {i + 1}/{n_sim}")
#                 converge,som = self.check_convergence(lras, eps)
#                 print(f'After {i+1} new simulations, Convergence ? {som} !!!!!!!!!')
#                 if converge and check_convergence:
#                     break
#         self.lras = pd.DataFrame(lras, columns=[f"Class_{c}" for c in ['AA','A','BB','B','C']])
        
#         self.save_lras(save_path, self.lras)
#         if (i + 1) % self.quantile_step != 0:
#             converge,som = self.check_convergence(lras, eps)
#         if converge:
#             print(f'Convergence of all Risk Class successfully after {i+1} new simulations: {som} !!!!!!!!!')
#         else:
#             print(f' NO Convergence of all Risk Class after {i+1} new simulations: {som} !!!!!!!!!')
        
    ##########################################################################
    # def moc_par_classe(self, alpha=0.99, percent = True):
    #     """
    #     Calcule la MoC par classe au niveau alpha
    #     """
    #     moc = pd.Series()
    #     df_LRA = self.lras
    #     q_ref = self.lra_central
    #     for i, col in enumerate(df_LRA.columns, start=1):
    #         q_alpha = df_LRA[col].quantile(alpha)
    #         moc.loc[i] = (q_alpha - q_ref.loc[i])
    #     if percent:
    #         moc = moc / q_ref
    #     return pd.Series(moc, name=f"MoC_{alpha*100}%")

        



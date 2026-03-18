import re
import numpy as np
import pandas as pd

VAR = ['min_SLDCRDMMS_SUM_Mm6_Mm1', 
       'IND_DP_MAX', 
       'MAX_NBRJJR_SS_Mm3_Mm0', 
       'client_haut_risque', 
       'MNTECSAVO', 
       'ANCIENNETE']

# dumies_order = {'min_SLDCRDMMS_SUM_Mm6_Mm1': ['5924+', '445-5924', '<445'],
#                 'IND_DP_MAX': [0, 1],
#                 'MAX_NBRJJR_SS_Mm3_Mm0': ['<1', '1+'],
#                 'client_haut_risque': [0, 1],
#                 'MNTECSAVO': ['278+', '<278'],
#                 'ANCIENNETE': ['294+', '<294']
#                 }

impute_map = {
        "min_SLDCRDMMS_SUM_Mm6_Mm1" : 400,
        "IND_DP_MAX" : 1,
        "MAX_NBRJJR_SS_Mm3_Mm0" : 2,
        'ANCIENNETE' : 200,
        'MNTECSAVO': 300,
        'client_haut_risque' :1,
    }


class Predictor:
    def __init__(self,df_summary,const, VAR=VAR,impute_map=impute_map): #,dumies_order=dumies_order
        self.VAR = VAR
        #self.dumies_order=dumies_order
        self.impute_map=impute_map
        self.df_summary = df_summary
        self.probas_rules = self.build_rules_from_df_summary('coef')
        self.point1000_rules = self.build_rules_from_df_summary('points/1000')
        self.const = const
        

    def compute_var(self, df):
                        # --- 2. Variables ratio _RES
        def compute_res_var(var):
            if "_RES" in var:
                base_var = var.replace("_RES", "")
                df[var] = (
                    df[base_var] / df["ENGAGEMENT_SUM"].replace(0, np.nan)
                )
        for var in self.VAR:

            # --- 1. Variables croisées avec #
            if "#" in var:
                left, right = var.split('#')

                left_var, mod1 = left.split('&')
                right_var, mod2 = right.split('&')
                compute_res_var(left_var)
                compute_res_var(right_var)
                cond1, _ = self.parse_modalite(mod1)
                cond2, _ = self.parse_modalite(mod2)

                def compute_cross(row):
                    v1 = row[left_var]
                    v2 = row[right_var]

                    # gestion valeurs manquantes
                    if pd.isna(v1) or pd.isna(v2):
                        return np.nan   # OU 0 si tu veux un comportement différent

                    return int(cond1(v1) or cond2(v2))

                df[var] = df.apply(compute_cross, axis=1)

            else :
                compute_res_var(var)
        return df

    def get_var(self):
        def res(a):
            if '_RES' in a:
                if 'ENGAGEMENT_SUM' not in vars:
                    vars.append('ENGAGEMENT_SUM')
                return a.replace('_RES','')
            return a
        vars = []
        for var in self.VAR:
            if '#'in var:
                left = var.split('#')[0]
                right = var.split('#')[1]
                a = res(left.split('&')[0])
                b = res(right.split('&')[0])
                vars = vars+[a,b]
            else : 
                vars.append(var)
        return vars

    def clean_df(self,df):
        vars = self.get_var()
        df = df[vars]
        df = df.applymap(
        lambda x: (
            str(x)
            .replace('\u202f', '')   # espace insécable
            .replace(' ', '')        # espace normal
            .replace(',', '.')       # virgule -> point
            .replace('?', '')        # ✅ supprime les '?'
            if isinstance(x, str) else x
        ))
        df = df.apply(pd.to_numeric, errors='coerce')
        if 'IND_DP_MAX' in vars : 
            df['IND_DP_MAX'] = (df['IND_DP_MAX'] != 0).astype(int).astype('category')
        return df
        

    def impute_with_map(self,df):
        """
        Applique l'imputation sur un DataFrame selon les valeurs d'imputation
        fournies dans impute_map.
        """
        df2 = df.copy()
        for col, val in self.impute_map.items():
            if col in df2.columns:
                df2[col] = df2[col].fillna(val)
        return df2
    
 

    def get_dummies(self,df):
        categ = self.dumies_order
        X = df.copy()
        
        for col in self.VAR:
            print(f"Processing column: {col}")
            X[col] = X[col].astype('category')
            
            # Utiliser l'ordre des catégories appris sur le train
            new_categories = categ[col]
            X[col] = X[col].cat.reorder_categories(new_categories, ordered=False)  # <--- plus de inplace

        
        # Créer les dummies en supprimant la catégorie de base (la moins risquée)
        X_dummies = pd.get_dummies(X, drop_first=True,prefix_sep='&')
        
        return X_dummies*1

    def prepare_data(self,df):
        df = self.clean_df(df)
        df = self.compute_var(df)
        df = self.impute_with_map(df)
        #df = self.get_dummies(df)
        return df
      
    def predire_probas(self,df):

        score_probas = self.apply_scoring(df,self.probas_rules) + self.const
        return 1/(1+ np.exp(-score_probas))

    def predire_points1000(self,df):

        score_probas = self.apply_scoring(df,self.point1000_rules)
        return score_probas

    
    def parse_modalite(self,mod):
        mod = str(mod).strip()

        # Cas binaire exact
        if re.fullmatch(r"\d+(\.\d+)?", mod):
            val = float(mod)
            return lambda x: x == val,f'x == {val}'

        # Cas intervalle a-b
        if "-" in mod and not (mod.startswith("<") or mod.startswith(">=")):
            a, b = mod.split("-")
            return lambda x, a=float(a), b=float(b): (x >= a) and (x < b),f'(x >= {a}) and (x < {b})'

        # Cas x+
        if mod.endswith("+"):
            thr = float(mod[:-1])
            return lambda x, thr=thr: x >= thr,f'x >= {thr}'
        # Cas >=x
        if mod.startswith(">="):
            thr = float(mod[2:])
            return lambda x, thr=thr: x >= thr,f'x >= {thr}'
        # Cas NAN
        if mod == 'NAN':
            return lambda x: bool(np.isnan(x)),'NAN'
        # Cas NOT_NAN
        if mod == 'NOT_NAN':
            return lambda x: not bool(np.isnan(x)),'NOT NAN'

        # Cas <x
        if mod.startswith("<"):
            thr = float(mod[1:])
            return lambda x, thr=thr: x < thr,f'x < {thr}'

        raise ValueError(f"Modalité inconnue : {mod}")

    def build_rules_from_df_summary(self,point_col = 'coef'):
        rules = []

        for _, row in self.df_summary.iterrows():
            var = row["variable"]
            mod = row["modalite"]
            pts = float(row[point_col])

            condition,formula = self.parse_modalite(mod)

            rules.append({
                "variable": var,
                "condition": condition,
                "points": pts,
                'formula': formula,
                "source": "df_summary"
            })

        return [dic for dic in rules if dic['points']>0]

    def merge_rules(self,auto_rules, manual_rules=None):
        if manual_rules is None:
            return auto_rules
        return auto_rules + manual_rules

    def build_scoring_function(self,rules):
        def compute_score(row):
            score = 0
            for rule in rules:
                var = rule["variable"]
                cond = rule["condition"]
                pts = rule["points"]

                try:
                    if cond(row[var]):
                        score += pts
                except Exception as e:
                    raise e
            return score

        return compute_score


    def apply_scoring(self,df, rules):
        # Construire la fonction de scoring
        scorer = self.build_scoring_function(rules)
        # Appliquer
        scores = df.apply(scorer, axis=1)
        return scores


import re
import numpy as np
import pandas as pd

VAR = [
    'payment_regularity_score',
    'credit_score',
    'credit_utilization',
    'early_warning_flag',
    'monthly_expenses',
    'total_outstanding_to_income'
    ]

impute_map = {
        "monthly_expenses" : 0,
        #"early_warning_flag" : 2,
        'payment_regularity_score': 0.8,
        'credit_utilization' :0.3,
        'credit_score':540,
        'total_outstanding_to_income':0,
    }


class Predictor:
    def __init__(self,df_summary,const, VAR=VAR,impute_map=impute_map): 
        self.VAR = VAR
        self.impute_map=impute_map
        self.df_summary = df_summary
        self.probas_rules = self.build_rules_from_df_summary('coef')
        self.point1000_rules = self.build_rules_from_df_summary('points/1000')
        self.const = const
        

    def clean_df(self,df):
        vars = self.VAR
        df = df.loc[:,vars]
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
    

    def prepare_data(self,df):
        df = self.clean_df(df)
        #df = self.compute_var(df)
        df = self.impute_with_map(df)
        #df = self.get_dummies(df)
        return df
      
    def predire_probas(self,df):

        score_probas = self.apply_scoring(df,self.probas_rules) + self.const
        return 1/(1+ np.exp(-score_probas))

    def predire_points1000(self,df):

        score_probas = self.apply_scoring(df,self.point1000_rules)
        return score_probas


    def parse_modalite(self, mod):
        """
        Parse une modalité issue d'une scorecard IRB
        et retourne :
        - une fonction booléenne f(x)
        - une représentation textuelle
        """

        mod = str(mod).strip()

        # -------------------------
        # Missing
        # -------------------------
        if mod.lower() == 'missing':
            return (
                lambda x: pd.isna(x),
                'x is NA'
            )

        # -------------------------
        # Valeur discrète (0.0, 1.0, etc.)
        # -------------------------
        if re.fullmatch(r"-?\d+(\.\d+)?", mod):
            val = float(mod)
            return (
                lambda x, v=val: x == v,
                f'x == {val}'
            )

        # -------------------------
        # Intervalles [a, b)
        # y compris [-inf, a) et [a, inf)
        # -------------------------
        match = re.fullmatch(
            r"\[\s*([-\w\.]+)\s*,\s*([-\w\.]+)\s*\)",
            mod
        )

        if match:
            left, right = match.groups()

            # borne gauche
            if left == '-inf':
                left_val = -np.inf
            else:
                left_val = float(left)

            # borne droite
            if right == 'inf':
                right_val = np.inf
            else:
                right_val = float(right)

            # Construction de la règle
            return (
                lambda x, l=left_val, r=right_val: (
                    (x >= l) and (x < r)
                ),
                f'{left_val} <= x < {right_val}'
            )

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


import numpy as np
import pandas as pd


class Summarize:

    def __init__(self,X_train,X_test,cible_col='default_t_plus_1'):
        self.X_train = X_train
        self.X_test = X_test
        self.n_train = self.X_train.shape[0]
        self.cible_col = cible_col
    def summarize_model(self,logit_model): 
        params = logit_model.params
        var_set = { col.split('&')[0] for col in params.index if col!='const'}
        self.X_train['count']=1
        
        summary = {}
        for col in var_set : 
            summary[col] = {}
            
            coefs = params[[i for i in params.index if i.startswith(col)]]
            summary[col]['coef_max'] = coefs.max()

            taux_cibles = self.X_train.groupby(col)[self.cible_col].mean().sort_values()*100
            taux_pops = self.X_train.groupby(col)['count'].sum()/self.n_train*100

            for i in range(len(taux_cibles)) : 
                index = taux_cibles.index[i]
                summary[col][index] ={}
                cle = col +"&"+ str(index)
                if cle in params.index : 
                    coef = params[cle] #pvalues
                    pvalue = logit_model.pvalues[cle] 
                else:
                    coef = 0
                    pvalue = np.nan
                taux_cible = taux_cibles[index]
                if i!=len(taux_cibles)-1:
                    ecart_relatif = (taux_cibles[taux_cibles.index[i+1]]/taux_cible-1) *100
                else:
                    ecart_relatif = np.nan

                summary[col][index]['taux_cible'] = taux_cible
                summary[col][index]['taux_pop'] = taux_pops[index]
                summary[col][index]['coef'] = coef
                summary[col][index]['pvalue'] = pvalue
                summary[col][index]['ecart_relatif'] = ecart_relatif

        sum_max_coefs = sum([ summary[col]['coef_max']  for col in var_set ])#for index in summary[col] ])
        for col in var_set :
            max = summary[col]['coef_max']
            max_contrib =0
            for index in summary[col] :
                if index != 'coef_max':
                    coef = summary[col][index]['coef']
                    x = 1000*(max-coef)/sum_max_coefs
                    if x>=max_contrib:
                        max_contrib = x
                    summary[col][index]['points_1000'] = x
            summary[col]['contribution'] = max_contrib/10

    def retrieve_var(self,col):
        terme = '&'+col.split('&')[-1]
        index = col.rfind(terme)
        if index !=1:
            return col[:index] + col[index+len(terme):]
        return col
    def build_summary_dataframe(self,logit_model):
        target_col=self.cible_col
        params = logit_model.params
        pvals = logit_model.pvalues

        var_set = {self.retrieve_var(col) for col in params.index if col!='const'}

        X_train = self.X_train.copy()
        X_train["_count"] = 1

        summary_rows = []
        var_max_coefs = {}

        # ---- 1) EXTRACTION DES INFOS ----
        for var in var_set:

            coefs = params[[i for i in params.index if i.startswith(var)]]
            var_max_coefs[var] = coefs.max()

            taux_cibles = X_train.groupby(var)[target_col].mean().sort_values() * 100
            taux_pops = X_train.groupby(var)['_count'].sum() / self.n_train * 100

            modalités = taux_cibles.index.tolist()

            for i, modal in enumerate(modalités):

                key = f"{var}&{modal}"

                coef = params[key] if key in params.index else 0
                pvalue = pvals[key] if key in pvals.index else np.nan

                taux_cible = taux_cibles[modal]
                taux_pop = taux_pops[modal]

                if i < len(modalités) - 1:
                    taux_next = taux_cibles[modalités[i+1]]
                    ecart_rel = (taux_next / taux_cible - 1) * 100 if taux_cible > 0 else np.nan
                else:
                    ecart_rel = np.nan

                summary_rows.append({
                    "variable": var,
                    "modalite": modal,
                    "taux_pop (%)": taux_pop,
                    "taux_cible (%)": taux_cible,
                    "coef": coef,
                    "pvalue": pvalue,
                    "ecart_relatif (%)": ecart_rel
                })

        df = pd.DataFrame(summary_rows)

        # ---- 2) Points /1000 ----
        total_max = sum(var_max_coefs.values())

        df["points/1000"] = df.apply(
            lambda r: 1000 * (var_max_coefs[r["variable"]] - r["coef"]) / total_max,
            axis=1
        )

        # ---- 3) Contribution echelle ----
        contributions = (
            df.groupby("variable")["points/1000"]
            .max() / 10
        )

        df["contrib_echelle (%)"] = df["variable"].map(contributions)

        # ---- 3) Contribution score ----
        moy_pts = df.groupby('variable')['points/1000'].mean()

        df['var'] = df.apply(lambda r: (r['taux_pop (%)']/100) * (moy_pts[r["variable"]] - r["points/1000"])**2,
                                                axis=1)
        ecart_type = np.sqrt(df.groupby('variable')['var'].sum())
        ecart_type_sum = np.sum(ecart_type)
        df['contrib_score (%)'] = df.apply(lambda r: 100 *ecart_type[r["variable"]]/ ecart_type_sum,
                                        axis=1)

        df.drop('var',axis=1,inplace=True)

        # ---- 5) ARRONDIR TOUTES LES COLONNES NUMÉRIQUES ----
        # num_cols = df.select_dtypes(include=[np.number]).columns
        # df[num_cols] = df[num_cols].round(2)

        return df


    def display_summary_with_histograms(self,df_summary, excel_path=None):

        df = df_summary.copy()
        # df["pvalue"] = df["pvalue"].fillna('--')
        # df["ecart_relatif (%)"] = df["ecart_relatif (%)"].fillna('--')

        # 🔹 Colonnes numériques arrondies à 2 décimales
        num_cols = df.select_dtypes(include="number").columns
        #df[num_cols] = df[num_cols].round(2)

        # 🎨 Couleurs adaptées au mode sombre
        # - Bleu foncé pour taux_pop
        # - Orange foncé pour taux_cible
        # - Bordures plus foncées pour meilleure visibilité
        bar_props_pop = {
            "color": "#2980B9",        # bleu profond (dark friendly)
            "border-color": "#1B4F72"  # bordure bleu très foncé
        }
        bar_props_cible = {
            "color": "#BA0B0B",        # orange foncé
            "border-color": "#7E5109"  # bordure marron très foncé
        }

        # 🔹 Style global
        styled = (
            df.style
            .bar(
                subset=["taux_pop (%)"],
                color=bar_props_pop["color"],
                #border_color=bar_props_pop["border-color"]
            )
            .bar(
                subset=["taux_cible (%)"],
                color=bar_props_cible["color"],
                #border_color=bar_props_cible["border-color"]
            )
            .format("{:.2f}", subset=num_cols)
            .set_properties(**{
                "text-align": "center",
                "white-space": "nowrap",
                "color": "white",               # texte blanc pour dark mode
                "background-color": "#1E1E1E"  # fond sombre VS Code
            })
            .hide(axis="index")
        )

        # 🔹 Export Excel
        if excel_path is not None:
            styled.to_excel(excel_path, engine="openpyxl")
            print(f"➡️ Export effectué : {excel_path}")

        return styled

    def display_summary_with_histograms2(self, df_summary, excel_path=None):

        df = df_summary.copy()

        # 🔹 Arrondi dès la copie (2 décimales sur colonnes numériques)
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].round(2)

        # 🔹 Valeurs manquantes
        df["pvalue"] = df["pvalue"].fillna("--")
        df["ecart_relatif (%)"] = df["ecart_relatif (%)"].fillna("--")

        # 🔹 Fond clair (style rapport académique)
        bg_color = "#FAFAFA"         # blanc très léger
        text_color = "#1A1A1A"       # gris très foncé (lisible impression)
        header_color = "#E8E8E8"     # gris clair pour entête

        # 🔹 Barres colorées pour les taux
        bar_pop_color = "#3498DB"     # bleu doux
        bar_cible_color = "#E74C3C"   # rouge atténué

        # --- Fonction de styling pour une variable donnée ---
        def style_group(df_group):
            return (
                df_group.style
                .bar(subset=["taux_pop (%)"], color=bar_pop_color)
                .bar(subset=["taux_cible (%)"], color=bar_cible_color)
                .format("{:.2f}", subset=num_cols)
                .set_properties(
                    **{
                        "text-align": "center",
                        "white-space": "nowrap",
                        "background-color": bg_color,
                        "color": text_color,
                    }
                )
                .set_table_styles(
                    [
                        {
                            "selector": "th",
                            "props": [
                                ("background-color", header_color),
                                ("color", text_color),
                                ("font-weight", "bold"),
                                ("text-align", "center"),
                            ],
                        }
                    ]
                )
                .hide(axis="index")
            )

        # 🔹 Regroupement par variable → dictionnaire de DataFrames stylés
        styled_tables = {}

        for var, group in df.groupby("variable"):
            styled_tables[var] = style_group(group)

        # 🔹 Export Excel optionnel
        if excel_path is not None:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                for var, group in df.groupby("variable"):
                    group.to_excel(writer, sheet_name=str(var), index=False)
            print(f"➡️ Export effectué : {excel_path}")

        return styled_tables

    def display_summary_with_histograms3(self, df_summary, excel_path=None):

        df = df_summary.copy()

        # --- Colonnes NUMÉRIQUES détectées proprement ---
        num_cols = df.select_dtypes(include="number").columns.tolist()

        # --- Arrondi uniquement sur valeurs numériques ---
        df[num_cols] = df[num_cols].round(2)

        # --- Colonnes textuelles où l'on peut mettre "--" ---
        df["pvalue"] = df["pvalue"].fillna("--")
        df["ecart_relatif (%)"] = df["ecart_relatif (%)"].fillna("--")

        # --- Fond clair ---
        bg_color = "#FAFAFA"
        alt_bg = "#F0F0F0"
        text_color = "#1A1A1A"
        header_color = "#E6E6E6"

        bar_pop_color = "#3498DB"
        bar_cible_color = "#E74C3C"

        # --- Bandes alternées par variable ---
        df["__bg__"] = (
            df["variable"].ne(df["variable"].shift()).cumsum() % 2
        )

        def highlight_groups(row):
            color = alt_bg if row["__bg__"] == 1 else bg_color
            return [
                f"background-color: {color}; color: {text_color};"
            ] * len(row)

        # --- Style final ---
        styled = (
            df.style
            .apply(highlight_groups, axis=1)
            .bar(subset=["taux_pop (%)"], color=bar_pop_color)
            .bar(subset=["taux_cible (%)"], color=bar_cible_color)
            .format("{:.2f}", subset=num_cols)  # ✔ uniquement numériques
            .set_properties(
                **{
                    "text-align": "center",
                    "white-space": "nowrap",
                    "color": text_color,
                }
            )
            .set_table_styles(
                [
                    {
                        "selector": "th",
                        "props": [
                            ("background-color", header_color),
                            ("color", text_color),
                            ("font-weight", "bold"),
                            ("text-align", "center"),
                        ],
                    }
                ]
            )
            .hide(axis="index")
        )

        # --- Export Excel optionnel ---
        if excel_path is not None:
            styled.to_excel(excel_path, engine="openpyxl")
            print(f"➡️ Export effectué : {excel_path}")

        return styled

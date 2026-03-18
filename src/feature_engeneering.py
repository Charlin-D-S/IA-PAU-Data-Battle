from math import ceil
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, _tree, plot_tree
import matplotlib.pyplot as plt


class FeatureEngineering:

    def __init__(self, X_train, y_train,X_test, tree=None):
        self.tree = tree or DecisionTreeClassifier(
            criterion='gini',
            min_samples_leaf=0.05,
            max_depth=2,
            random_state=42
        )
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.rule_map = None

    def fit_cart(self, vars_list, plot=False):
        self.vars_list = vars_list
        self.tree.fit(self.X_train[vars_list], self.y_train)

        if plot:
            self.plot_tree()

        # 🔥 Extraction des variables utilisées dans les splits
        tree_ = self.tree.tree_
        used_features_idx = tree_.feature[tree_.feature >= 0]  # ignore les feuilles (-2)
        used_features = sorted({self.vars_list[i] for i in used_features_idx})

        self.split_variables_ = "_AND_".join(used_features)  # stocker dans l'objet si tu veux
        return used_features


    def plot_tree(self):
        plt.figure(figsize=(8, 6))
        plot_tree(self.tree, feature_names=self.vars_list, filled=True, rounded=True)
        plt.show()

    def _extract_rules(self, node, conditions):
        tree = self.tree.tree_

        if tree.children_left[node] == _tree.TREE_LEAF:
            return {node: conditions}

        rules = {}
        try :
            feature = self.vars_list[tree.feature[node]]
            threshold = ceil(tree.threshold[node])  # ✅ arrondi ici

            # ✅ même sens que sklearn : <= à gauche, > à droite
            cond_left  = conditions + [f"{feature} < {threshold}"]
            cond_right = conditions + [f"{feature} >= {threshold}"]

            rules.update(self._extract_rules(tree.children_left[node], cond_left))
            rules.update(self._extract_rules(tree.children_right[node], cond_right))
        except Exception as e:
            return {}

        return rules

    def build_leaf_variable(self, plot=False):

        if plot:
            self.plot_tree()

        rules = self._extract_rules(0, [])
        # def get_leaf_id(row):
        #     for leaf_id, conds in rules.items():
        #         if all(eval(cond, {"__builtins__": None}, row.to_dict()) for cond in conds):
        #             return f"leaf_{leaf_id}"

        #     raise ValueError(f"Aucune feuille trouvée pour : {row.to_dict()}")  # Sécurité

        #leaf_series = self.X_train.apply(get_leaf_id, axis=1)
        rule_map = {f"leaf_{k}": " and ".join(v) for k, v in rules.items() if v }
        self.rule_map = rule_map
        return rule_map
    def train_apply_rule_map(self):
        def match_row(row):
            for leaf, rule in self.rule_map.items():
                # évaluer la règle sur la ligne
                if eval(rule, {"__builtins__": None}, row.to_dict()):
                    return self.rule_map[leaf]
            return None  # ou raise erreur si tu préfères

        return self.X_train.apply(match_row, axis=1)
    
    def test_apply_rule_map(self):
        def match_row(row):
            for leaf, rule in self.rule_map.items():
                # évaluer la règle sur la ligne
                if eval(rule, {"__builtins__": None}, row.to_dict()):
                    return self.rule_map[leaf]
            return None  # ou raise erreur si tu préfères

        return self.X_test.apply(match_row, axis=1)
        

    def croiser(self,VAR):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        var_list = VAR
        used = []
        while len(var_list)>=2 : 
            try :
                used_features = self.fit_cart(vars_list=var_list)
            except Exception as e:
                print(f"❌ Erreur lors du fit de l'arbre sur la taille {len(var_list)} :", e)
                break
            sub = used_features[1:]
            sb = [col for col in sub if col in used]
            if sb : 
                for v in sub:
                    if v in used:
                        var_list.remove(v)
                        used.remove(v)
                        break
                
            else : 
                try : 
                    self.build_leaf_variable()
                    if self.rule_map :
                        col = self.split_variables_
                        print(col)
                        df_train[col] = self.train_apply_rule_map()
                        df_test[col] = self.test_apply_rule_map()
                        var_list.remove(used_features[0])
                        used = used_features[1:].copy()
                    else :
                        return df_train, df_test
                except Exception as e:
                    print(f"❌ Erreur lors de la creation de la variable {col} avec {self.rule_map} :", e)
                    break
            print(f'Nombre de variables restantes : {len(var_list)}')   
        return df_train, df_test
    def croiser_step(self,VAR):
        df_train = pd.DataFrame()
        df_test = pd.DataFrame()
        var_list = VAR
        while len(var_list)>=2 : 
            try :
                used_features = self.fit_cart(vars_list=var_list)
            except Exception as e:
                print(f"❌ Erreur lors du fit de l'arbre sur la taille {len(var_list)} :", e)
                break                
            try : 
                self.build_leaf_variable()
                if self.rule_map :
                    col = self.split_variables_
                    print(col)
                    df_train[col] = self.train_apply_rule_map()
                    df_test[col] = self.test_apply_rule_map()
                    for col in used_features : 
                        var_list.remove(col)
                else :
                    return df_train, df_test
            except Exception as e:
                print(f"❌ Erreur lors de la creation de la variable {col} :", e)
                break
            print(f'Nombre de variables restantes : {len(var_list)}')  
        return df_train, df_test

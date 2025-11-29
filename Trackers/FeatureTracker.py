import pandas as pd
from sklearn.preprocessing import StandardScaler
import data 
import logisticRegression
import numpy as np
import Model

class FeatureTracker:
    def __init__(self, df):
        self.features = pd.DataFrame()
        self.removed_features = pd.DataFrame()
        self.cols_to_scale = []
        self.to_remove_cols = []
        self.df = df
        self.inter_terms = []
    
    def set_df(self, df):
        self.df = df.copy()
    
    def getDf(self):
        return self.df.copy()
    
    def getFeature(self, name):
        if name in self.features.columns: 
            isToScale = False
            for v in self.cols_to_scale: 
                if v == name: 
                    isToScale = True
                    break
            return self.features[name].copy(), isToScale
        elif name in self.to_remove_cols:
            isToScale = False
            for v in self.cols_to_scale: 
                if v == name: 
                    isToScale = True
                    break
            return self.removed_features[name].copy(), isToScale
    
    def get_cols_to_scale(self):
        return self.cols_to_scale
    
    def get_features_cols(self, with_inter_terms=False):
        if with_inter_terms: return self.features.columns 
        else: return list(set(self.features.columns) - set(self.inter_terms))

    def isNotInDfList(self):
        colInDf = self.df.columns.values
        colInFeatures = self.features.columns.values
        notInDfList = list(set(colInFeatures) - set(colInDf))
        return notInDfList
    
    def add(self, name, s:pd.Series, toScale=False):
        if name not in self.features.columns: 
            if s.isnull().any():
                s.fillna(0, inplace=True)
            if np.isinf(s).any():
                s.replace([np.inf, -np.inf], 0, inplace=True)
            self.features[name] = s.copy()
            if toScale: self.cols_to_scale.append(name)
        if name in self.to_remove_cols:
            self.to_remove_cols.remove(name)

    def remove(self, name, save=True):
        if name not in self.to_remove_cols:
            self.to_remove_cols.append(name)
        if name in self.features.columns: 
            if save: self.removed_features[name] = self.features[name].copy()
        elif name in self.df.columns:
            if save: self.removed_features[name] = self.df[name].copy()
            
    def remove_list(self, cols):
        for c in cols:
            self.remove(c)

        #if name in self.varsToScale: self.varsToScale.remove(name)
    
    def restore(self, name, isToScale=False):
        if name in self.removed_features:
            self.features[name] = self.removed_features.pop(name)
            self.to_remove_cols.remove(name)
            if isToScale and name not in self.cols_to_scale: self.cols_to_scale.append(name)
    
    def restore_list(self, cols):
        for c in cols: self.restore(c)

    def return_split_train_eval(self, X_other=None, toNpy=False, notToRemove=[]):
        if X_other is not None: X = self.flush_to_df(X_other=X_other, notToRemove=notToRemove)
        else: X = self.df
        X_train, X_val, y_train, y_val = data.get_split_train_eval_data(X)

        X_train_cols = X_train.columns
        for var in self.cols_to_scale:
            scaler = StandardScaler()
            if var in X_train_cols:
                scaler.fit(X_train[[var]])
                X_train[[var]] = scaler.transform(X_train[[var]])
                X_val[[var]] = scaler.transform(X_val[[var]])
        
        if toNpy:
            X_train = X_train.to_numpy()
            X_val = X_val.to_numpy()
            y_train = y_train.to_numpy()
            y_val = y_val.to_numpy()

        return X_train, y_train, X_val, y_val
        
    def print_features(self):
        print(self.features)

    def create_interaction_terms(self, var1, var2, terms:str, isVar1OneHot=False):
        items = terms.replace(" ", "").split(",")
        pairs = []
        new_cols ={}
        for item in items:
            a,b = item.split("x")
            pairs.append((int(a), int(b)))
        for a,b in pairs:
            col_name = f"{var1}_{a}_x_{var2}_{b}"
            if isVar1OneHot:
                new_cols[col_name] = (self.df[f'{var1}_{a}'] & (self.df[var2] == b)).astype(int) 
            else: 
                new_cols[col_name] = ((self.df[var1] == a) & (self.df[var2] == b)).astype(int)
        if new_cols:
            self.features = pd.concat([self.features, pd.DataFrame(new_cols, index=self.features.index)], axis=1)
            self.inter_terms.extend(new_cols.keys())

    def flush_to_df(self, X_other=None, notToRemove=[], returnDf=True, removeTargets=False):
        notHisFeature = []
        notInDfAlready = self.isNotInDfList()

        if X_other is not None: 
            X = X_other
            for c in self.df.columns:
                if c not in X_other.columns: notHisFeature.append(c)
            if notHisFeature: notInDfAlready = list(set(notInDfAlready) - set(notHisFeature))
        else: 
            X = self.df

        if notInDfAlready:
            X = pd.concat([X, self.features[notInDfAlready]], axis=1)
            self.features = self.features.copy()
        # else:
        #     print("every feature is in DF already")
        if len(self.to_remove_cols) != 0:
            for r in self.to_remove_cols:
                if r in X.columns and r not in notToRemove: 
                    X.drop(r, axis=1, inplace=True)
                if r in self.features.columns:
                    self.features.drop(r, axis=1, inplace=True)
        self.df = X.copy()
        if returnDf: 
            if removeTargets: X.drop('CARAVAN', axis=1, inplace=True)
            return X
        
    def save_state(self):
        state_tracker = FeatureTracker(self.df)
        state_tracker.cols_to_scale = self.cols_to_scale.copy()
        state_tracker.features = self.features.copy()
        state_tracker.to_remove_cols = self.to_remove_cols.copy()
        state_tracker.removed_features = self.removed_features.copy()
        state_tracker.flush_to_df(returnDf=False)
        return state_tracker

    def test_current(self, learning_rate=0.01, epochs=1000, class_weight=13, returnModel=True):
        X_train_np, y_train_np, X_val_np, y_val_np = self.return_split_train_eval(toNpy=True)
        model = Model.create_model(
            X_train_np, y_train_np, X_val_np, y_val_np, 
            learning_rate=learning_rate, extra_weight=class_weight,
            iterations=epochs, threshold_method='F1'
        )
        model.print_stats(X_val_np, y_val_np)
        if returnModel: return model
    
    def feature_comparator(self, X, baseExtraCols, colsToTest=None, cols_to_remove=None, add_inter_terms=True, learning_rate=0.01, epochs=1000, class_weight=13):
        featureTester = FeatureTracker(X)
        if cols_to_remove is None: cols_to_remove = list(self.to_remove_cols)

        for c in baseExtraCols:
            colToAdd, isToScale = self.getFeature(c)
            if c in cols_to_remove: cols_to_remove.remove(c)
            featureTester.add(c, colToAdd, toScale=isToScale)
        featureTester.to_remove_cols = cols_to_remove

        X = featureTester.flush_to_df()
        #print(featureTester.get_features_cols())

        X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(toNpy=True)
        print(f'--------------------| Test avec les variables de base |--------------------')
        #print(len(X.columns))

        model = Model.create_model(
            X_train_np, y_train_np, X_val_np, y_val_np, 
            learning_rate=learning_rate, extra_weight=class_weight,
            iterations=epochs, threshold_method='F1'
        )

        model.print_stats(X_val_np, y_val_np)
        print()            

        if colsToTest is not None:
            for c in colsToTest: 
                if c in cols_to_remove: 
                    cols_to_remove.remove(c)
                    featureTester.to_remove_cols = cols_to_remove
                colToAdd, isToScale = self.getFeature(c)
                featureTester.add(c, colToAdd, toScale=isToScale)
                X = featureTester.flush_to_df()

                X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(toNpy=True)
                
                print(f'----- Test avec la variable {c} -----')
                #print(len(X.columns))
                
                model = Model.create_model(
                    X_train_np, y_train_np, X_val_np, y_val_np, 
                    learning_rate=learning_rate, extra_weight=class_weight,
                    iterations=epochs, threshold_method='F1'
                )

                model.print_stats(X_val_np, y_val_np)
                print()
                featureTester.remove(c)

        if add_inter_terms: 
            for c in self.inter_terms:
                colToAdd, isToScale = self.getFeature(c)
                featureTester.add(c, colToAdd, toScale=isToScale)
            X = featureTester.flush_to_df()

            X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(toNpy=True)

            print(f'--------------------| Test avec interactions terms |--------------------')
            #print(len(X.columns))
            model = Model.create_model(
                X_train_np, y_train_np, X_val_np, y_val_np, 
                learning_rate=learning_rate, extra_weight=class_weight,
                iterations=epochs, threshold_method='F1'
            )

            model.print_stats(X_val_np, y_val_np)
            print()
            
            if colsToTest is not None:
                for c in colsToTest: 
                    colToAdd, isToScale = self.getFeature(c)
                    featureTester.restore(c, isToScale=isToScale)
                    X = featureTester.flush_to_df()

                    X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(toNpy=True)
                    
                    print(f'----- Test de la variable {c} -----')
                    #print(len(X.columns))
                    
                    model = Model.create_model(
                        X_train_np, y_train_np, X_val_np, y_val_np, 
                        learning_rate=learning_rate, extra_weight=class_weight,
                        iterations=epochs, threshold_method='F1'
                    )

                    model.print_stats(X_val_np, y_val_np)
                    print()
                    featureTester.remove(c)

#  def add(self, name, values, tag=None):
#         """Add a new variable or interaction with optional tag"""
#         self.features[name] = (values, tag)
#         self.buffer[name] = values

#     def flush_to_df(self, df):
#         if self.buffer:
#             new_cols = pd.DataFrame(self.buffer, index=df.index)
#             df.loc[:, new_cols.columns] = new_cols
#             self.buffer = {}

#     def add_to_removed(self, df, name=None, tag=None):
#         """
#         Remove a feature by name OR tag and store its column in removed_features.
#         """
#         # --- Remove by name ---
#         if name is not None:
#             # Remove from feature tracker
#             if name in self.features:
#                 del self.features[name]

#             # Remove from dataframe
#             if name in df.columns:
#                 self.removed_features[name] = df.pop(name)

#             return
#         # =============== REMOVE BY TAG ===============
#         if tag is not None:
#             to_remove = []

#             # meta = (values, tag_value)
#             for fname, meta in self.features.items():
#                 _, tag_value = meta
#                 if tag_value == tag:
#                     to_remove.append(fname)

#             for fname in to_remove:
#                 del self.features[fname]

#                 if fname in df.columns:
#                     self.removed_features[fname] = df.pop(fname)

#             return

#         # =============== NO INPUT ===============
#         raise ValueError("You must provide either 'name' or 'tag'.")

           
    
#     def get(self, name):
#         """Retrieve a variable by name"""
#         return self.features.get(name, (None, None))[0]
    
#     def remove(self, name=None, tag=None):
#         """Remove a variable by name and save it in removed_features"""
#         if name is not None and name in self.features:
#             self.removed_features[name] = self.features.pop(name)

#         if tag is not None:
#             to_remove = [fname for fname, (_, ftag) in self.features.items() if ftag == tag]

#             if not to_remove:
#                 print(f"No features found with tag '{tag}'.")
#                 return

#             for fname in to_remove:
#                 self.removed_features[fname] = self.features.pop(fname)

#             return

    
#     def restore(self, name):
#         """Restore a removed variable"""
#         if name in self.removed_features:
#             self.features[name] = self.removed_features.pop(name)
#             self.buffer[name] = self.get(name)
#         else:
#             print(f"Feature '{name}' not found in removed features.")
    
#     def list_features(self, tag=None):
#         """List feature names, optionally filtered by tag"""
#         if tag is None:
#             return list(self.features.keys())
#         else:
#             return [k for k, (_, t) in self.features.items() if t == tag]
    
#     def list_removed_features(self):
#         """List removed feature names"""
#         return list(self.removed_features.keys())
    
#     def all_features(self, tag=None):
#         """Return a DataFrame of all features, optionally filtered by tag"""
#         if tag is None:
#             return pd.DataFrame({k: v for k, (v, _) in self.features.items()})
#         else:
#             return pd.DataFrame({k: v for k, (v, t) in self.features.items() if t == tag})

#     def test_feature(self, X_base, scaler):
#         X = X_base.copy()
#         #scaler = StandardScaler()
#         for name, (values, tag) in self.features.items():
#             print(f"\n--- TEST DE LA CARACTÉRISTIQUE : {name} ---")
        
#             # 1. Ajouter la nouvelle caractéristique (feature)
#             # Assurez-vous que les indices correspondent entre X et values
#             X[name] = values # Si values est une Series/numpy array
            
#             # 2. Split le Data
#             # Assumons que y_base est inclus dans le split si besoin, sinon on split X et y séparément
#             X_train, X_val, y_train, y_val = data.get_split_train_eval_data(X, toNpy=False)
            
#             # 3. Mise à l'Échelle (Scaling) de la nouvelle Feature
#             # IMPORTANT : Scaler uniquement la nouvelle colonne en se basant sur le train
            
#             # Ajuster et Transformer sur X_train
#             X_train[name] = scaler.fit_transform(X_train[[name]])
            
#             # Transformer uniquement sur X_val
#             X_val[name] = scaler.transform(X_val[[name]])
            
#             # Convertir en NumPy pour l'entraînement (si votre fonction l'exige)
#             X_train_np = X_train.values
#             X_val_np = X_val.values
#             y_train_np = y_train.values
#             y_val_np = y_val.values
            
#             # 4. Entraînement du Modèle
#             # Utiliser les hyperparamètres optimisés du modèle de base
#             w, b = logisticRegression.logistic_regression(
#                 X_train_np, y_train_np, X_val_np, y_val_np, 
#                 learning_rate=0.01, 
#                 extra_weight=13.1, 
#                 iterations=1000 # ou le nombre que vous utilisez
#             )
            
#             # 5. Optimisation du Seuil et Statistiques
#             # Trouve le meilleur seuil pour ce modèle spécifique
#             f1_score_threshold, score_f1 = logisticRegression.f1_score_threshold(X_val_np, y_val_np, w, b)
            
#             print(f"Meilleur F1-Score sur Validation avec {name}: {score_f1:.4f}")
#             print(f"Seuil optimal: {f1_score_threshold:.4f}")
            
#             # Imprimer les stats complètes (Accuracy, AUC, Matrice de Confusion)
#             logisticRegression.print_model_stats(X_val_np, y_val_np, w, b, f1_score_threshold)

#             # 6. Nettoyage : Enlever la colonne pour l'itération suivante
#             # On enlève la colonne de la copie X pour recommencer avec seulement X_base à l'itération suivante
#             X = X.drop(columns=[name])
            


   
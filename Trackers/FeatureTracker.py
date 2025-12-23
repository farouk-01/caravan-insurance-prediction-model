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
        elif name in self.df.columns:
            return self.df[name].copy(), False
    
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
        if not isinstance(name,list):
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
        if name in self.removed_features: self.features[name] = self.removed_features.pop(name)
        if name in self.to_remove_cols: self.to_remove_cols.remove(name)
        if isToScale and name not in self.cols_to_scale: self.cols_to_scale.append(name)
    
    def restore_list(self, cols):
        for c in cols: self.restore(c)

    def return_split_train_eval(self, X_other=None, toNpy=False, notToRemove=[], to_scale=True):
        if X_other is not None: X = self.flush_to_df(X_other=X_other, notToRemove=notToRemove)
        else: X = self.df
        X_train, X_val, y_train, y_val = data.get_split_train_eval_data(X)

        X_train_cols = X_train.columns
        if to_scale:
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

    def get_trained_model(self, learning_rate=0.01, epochs=1000, class_weight=1, set_threshold_to=0.1, threshold_method=None, print_stats=True, returnModel=True, print_metrics=False, **kwargs):
        X_train_np, y_train_np, X_val_np, y_val_np = self.return_split_train_eval(toNpy=True)
        if threshold_method is not None: set_threshold_to=None
        model = Model.create_model(
            X_train_np, y_train_np, X_val_np, y_val_np, 
            learning_rate=learning_rate, extra_weight=class_weight,
            iterations=epochs,
            threshold_method=threshold_method,
            set_threshold_to=set_threshold_to,
            **kwargs
        )
        if print_stats: model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
        if returnModel: return model
    
    def feature_comparator(self, X, baseExtraCols=None, colsToTest=None, cols_to_remove=None, add_inter_terms=True,
                            learning_rate=0.01, epochs=1000, class_weight=1, print_metrics=True, **kwargs):
        featureTester = FeatureTracker(X)
        featureTester.cols_to_scale = self.cols_to_scale.copy()
        if cols_to_remove is None: cols_to_remove = list(self.to_remove_cols)

        if baseExtraCols is not None:
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
            iterations=epochs, threshold_method='F1', **kwargs
        )

        model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
        print()            

        if colsToTest is not None:
            for c in colsToTest: 
                if c in cols_to_remove: 
                    cols_to_remove.remove(c)
                colToAdd, isToScale = self.getFeature(c)
                featureTester.add(c, colToAdd, toScale=isToScale)
                X = featureTester.flush_to_df()

                X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(toNpy=True)
                
                print(f'|----- Test avec la variable {c} -----|')
                #print(len(X.columns))
                
                model = Model.create_model(
                    X_train_np, y_train_np, X_val_np, y_val_np, 
                    learning_rate=learning_rate, extra_weight=class_weight,
                    iterations=epochs, threshold_method='F1', **kwargs
                )

                model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
                print()
                featureTester.remove(c)

        if add_inter_terms and len(self.inter_terms) != 0: 
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

            model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
            print()
            
            if colsToTest is not None:
                for c in colsToTest: 
                    colToAdd, isToScale = self.getFeature(c)
                    featureTester.restore(c, isToScale=isToScale)
                    X = featureTester.flush_to_df()

                    X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(toNpy=True)
                    
                    print(f'|----- Test de la variable {c} -----|')
                    #print(len(X.columns))
                    
                    model = Model.create_model(
                        X_train_np, y_train_np, X_val_np, y_val_np, 
                        learning_rate=learning_rate, extra_weight=class_weight,
                        iterations=epochs, threshold_method='F1'
                    )

                    model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
                    print()
                    featureTester.remove(c)
import pandas as pd
from sklearn.preprocessing import StandardScaler
import data 
import logisticRegression
import numpy as np
import Model
import copy

class FeatureTracker:
    def __init__(self, df):
        self.features = pd.DataFrame()
        self.removed_features = pd.DataFrame()
        self.cols_to_scale = []
        self.to_remove_cols = []
        self.df = df.copy()
        self.inter_terms = []
    
    def set_df(self, df):
        self.df = df.copy()

    def refresh_tracker(self):
        self.features = self.features.copy()
        self.removed_features = self.removed_features.copy()
    
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
    
    def get_cols_to_scale_in_df(self):
        return self.features.columns.intersection(self.get_cols_to_scale()).to_list()

    def get_num_cat_cols(self):
        num_cols = self.get_cols_to_scale_in_df()
        X = self.flush_to_df(removeTargets=True)
        cat_cols = list(set(X.columns.values) - set(num_cols))
        return num_cols, cat_cols

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
            if toScale and name not in self.cols_to_scale: self.cols_to_scale.append(name)
        if name in self.to_remove_cols:
            self.to_remove_cols.remove(name)

    def delete(self, name):
        if name in self.features.columns:
            self.features.pop(name)
        if name in self.cols_to_scale:
            self.cols_to_scale.remove(name)
        if name in self.to_remove_cols:
            self.to_remove_cols.remove(name)
        if name in self.removed_features.columns:
            self.removed_features.pop(name)
        if name in self.df.columns:
            self.df.pop(name)

    def delete_list(self, cols):
        for c in cols: self.delete(c)

    def remove(self, name):
        if not isinstance(name,list):
            if name not in self.to_remove_cols:
                self.to_remove_cols.append(name)
            if name in self.features.columns: 
                self.removed_features[name] = self.features[name].copy()
            elif name in self.df.columns: #inutile je crois
                self.removed_features[name] = self.df[name].copy()
        
    def remove_list(self, cols):
        for c in cols:
            self.remove(c)

        #if name in self.varsToScale: self.varsToScale.remove(name)
    
    def restore(self, name, is_to_scale=False):
        if name in self.removed_features: self.features[name] = self.removed_features.pop(name)
        if name in self.to_remove_cols: self.to_remove_cols.remove(name)
        if is_to_scale and name not in self.cols_to_scale: self.cols_to_scale.append(name)
    
    def restore_list(self, cols):
        for c in cols: self.restore(c)

    def return_split_train_eval(self, X_other=None, to_np=False, not_to_remove=[], to_scale=True):
        if X_other is not None: X = self.flush_to_df(X_other=X_other, notToRemove=not_to_remove)
        else: X = self.df
        X_train, X_val, y_train, y_val = data.get_split_train_eval_data(X)

        X_train_cols = X_train.columns
        if to_scale:
            data_info = data.DataInfo.get_instance()
            cols_in_df_to_scale = [c for c in self.cols_to_scale if c in X_train_cols]
            if cols_in_df_to_scale:
                X_train = X_train.copy()
                X_val = X_val.copy()
                X_train = data_info.fit_transform(X_train, cols_in_df_to_scale)
                X_val = data_info.transform(X_val, cols_in_df_to_scale)
            # for var in self.cols_to_scale:
            #     if var in X_train_cols:
            #         scaler.fit(X_train[[var]])
            #         X_train[[var]] = scaler.transform(X_train[[var]])
            #         X_val[[var]] = scaler.transform(X_val[[var]])
        
        if to_np:
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

    def get_trained_model(self, learning_rate=0.01, iterations=1000, class_weight=1, set_threshold_to=0.1, threshold_method=None, print_stats=True, returnModel=True, print_metrics=False, **kwargs):
        X_train_np, y_train_np, X_val_np, y_val_np = self.return_split_train_eval(to_np=True)
        if threshold_method is not None: set_threshold_to=None
        model = Model.create_model(
            X_train_np, y_train_np, X_val_np, y_val_np, 
            learning_rate=learning_rate, class_weight=class_weight,
            iterations=iterations,
            threshold_method=threshold_method,
            set_threshold_to=set_threshold_to,
            **kwargs
        )
        if print_stats: model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
        if returnModel: return model
    
    def feature_comparator(self, X, base_extra_cols=None, cols_to_test=None, cols_to_remove=None, add_inter_terms=True,
                           test_on_train=False,
                            learning_rate=0.01, epochs=1000, class_weight=1, print_metrics=True, **kwargs):
        self.remove_list(cols_to_test)
        featureTester = FeatureTracker(X)

        featureTester.cols_to_scale = self.cols_to_scale.copy()

        if cols_to_remove is None: cols_to_remove = list(self.to_remove_cols)

        if base_extra_cols is not None:
            for c in base_extra_cols:
                colToAdd, isToScale = self.getFeature(c)
                if c in cols_to_remove: cols_to_remove.remove(c)
                featureTester.add(c, colToAdd, toScale=isToScale)
        
        featureTester.to_remove_cols = cols_to_remove
        
        X = featureTester.flush_to_df()
        

        X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(to_np=True)
        print(f'|----- Variable de base -----|')

        #threshold_method='F1' removed
        model = Model.create_model(
            X_train_np, y_train_np, X_val_np, y_val_np, 
            learning_rate=learning_rate, class_weight=class_weight,
            iterations=epochs, **kwargs
        )

        if test_on_train : model.print_stats(X_train_np, y_train_np, print_metrics=print_metrics)
        else : model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
        print()

        if cols_to_test is not None:
            for c in cols_to_test: 
                if c in cols_to_remove: 
                    cols_to_remove.remove(c)
                colToAdd, isToScale = self.getFeature(c)
                featureTester.add(c, colToAdd, toScale=isToScale)
                X = featureTester.flush_to_df()

                X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(to_np=True)
                
                print(f'|----- {c} -----|')
                #print(len(X.columns))
                
                model = Model.create_model(
                    X_train_np, y_train_np, X_val_np, y_val_np, 
                    learning_rate=learning_rate, class_weight=class_weight,
                    iterations=epochs, threshold_method='F1', **kwargs
                )

                if test_on_train : model.print_stats(X_train_np, y_train_np, print_metrics=print_metrics)
                else : model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
                print()
                featureTester.remove(c)

        if add_inter_terms and len(self.inter_terms) != 0: 
            for c in self.inter_terms:
                colToAdd, isToScale = self.getFeature(c)
                featureTester.add(c, colToAdd, toScale=isToScale)
            X = featureTester.flush_to_df()

            X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(to_np=True)

            print(f'|----- Test avec interactions terms -----|')
            #print(len(X.columns))
            model = Model.create_model(
                X_train_np, y_train_np, X_val_np, y_val_np, 
                learning_rate=learning_rate, class_weight=class_weight,
                iterations=epochs, threshold_method='F1'
            )

            if test_on_train : model.print_stats(X_train_np, y_train_np, print_metrics=print_metrics)
            else : model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
            print()
            
            if cols_to_test is not None:
                for c in cols_to_test: 
                    colToAdd, isToScale = self.getFeature(c)
                    featureTester.restore(c, is_to_scale=isToScale)
                    X = featureTester.flush_to_df()

                    X_train_np, y_train_np, X_val_np, y_val_np = featureTester.return_split_train_eval(to_np=True)
                    
                    print(f'|----- {c} -----|')
                    #print(len(X.columns))
                    
                    model = Model.create_model(
                        X_train_np, y_train_np, X_val_np, y_val_np, 
                        learning_rate=learning_rate, class_weight=class_weight,
                        iterations=epochs, threshold_method='F1'
                    )

                    if test_on_train : model.print_stats(X_train_np, y_train_np, print_metrics=print_metrics)
                    else : model.print_stats(X_val_np, y_val_np, print_metrics=print_metrics)
                    print()
                    featureTester.remove(c)
    
    def make_eval_tracker(self):
        df_eval = data.get_eval_data()
        categorical_non_ordinales = ['MOSTYPE', 'MOSHOOFD']
        df_eval = pd.get_dummies(df_eval, columns=categorical_non_ordinales, prefix=categorical_non_ordinales, dtype=int, drop_first=True)

        feature_eval_tracker = FeatureTracker(df_eval)
        feature_eval_tracker.to_remove_cols = self.to_remove_cols.copy()
        feature_eval_tracker.cols_to_scale = self.cols_to_scale.copy()

        income_brackets_midpoints = {
            'MINKM30': 15000,    
            'MINK3045': 37500,   
            'MINK4575': 60000,   
            'MINK7512': 98500,   
            'MINK123M': 180000 
        }

        income_cols = list(income_brackets_midpoints.keys())
        weighted_somme = 0
        for col in income_cols:
            weighted_somme += df_eval[col] * income_brackets_midpoints[col]
        total = df_eval[income_cols].sum(axis=1)

        for c in income_cols:
            feature_eval_tracker.remove(c)
        feature_eval_tracker.remove('MINKGEM')

        feature_eval_tracker.add('avg_area_income',  weighted_somme / total, toScale=True)
        feature_eval_tracker.add('PPERSAUTx_is_PBRAND_3_4',((df_eval["PPERSAUT"])*(df_eval['PBRAND'].isin([3,4]))))
        feature_eval_tracker.add('PPERSAUTxMHKOOP_geq_6',((df_eval["PPERSAUT"])*(df_eval['MHKOOP'] >= 6)))
        feature_eval_tracker.add('PPERSAUTxis_low_no_religion_area', df_eval['PPERSAUT']*((df_eval['MGODGE'] <= 3).astype(int)))
        feature_eval_tracker.add('is_PPERSAUT_6xMFALLEEN', ((df_eval['PPERSAUT'] == 6) * (df_eval['MFALLEEN'])).astype(int))
        feature_eval_tracker.add('PPERSAUT_6xMKOOPKLA', ((df_eval["PPERSAUT"] == 6)*(df_eval['MHKOOP'])))

        return feature_eval_tracker
    
    def make_split_eval_data(self):
        eval_tracker = self.make_eval_tracker()
        X = eval_tracker.flush_to_df()
        cols_to_scale = eval_tracker.cols_to_scale
        data_info = data.DataInfo.get_instance()
        cols_in_df_to_scale = [c for c in cols_to_scale if c in X.columns]
        if cols_in_df_to_scale:
            X = X.copy()
            X = data_info.transform(X, cols_in_df_to_scale)
        y = data.get_eval_targets()
        return X, y

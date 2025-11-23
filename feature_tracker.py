import pandas as pd

class FeatureTracker:
    def __init__(self):
        # Dictionary to hold all features and their tags
        self.features = {}  # {feature_name: (values, tag)}
        self.removed_features = {}  # {feature_name: (values, tag)}
        self.buffer = {}
    
    def add(self, name, values, tag=None):
        """Add a new variable or interaction with optional tag"""
        self.features[name] = (values, tag)
        self.buffer[name] = values

    def flush_to_df(self, df):
        if self.buffer:
            new_cols = pd.DataFrame(self.buffer, index=df.index)
            df.loc[:, new_cols.columns] = new_cols
            self.buffer = {}

    def add_to_removed(self, df, name=None, tag=None):
        """
        Remove a feature by name OR tag and store its column in removed_features.
        """
        # --- Remove by name ---
        if name is not None:
            # Remove from feature tracker
            if name in self.features:
                del self.features[name]

            # Remove from dataframe
            if name in df.columns:
                self.removed_features[name] = df.pop(name)

            return
        # =============== REMOVE BY TAG ===============
        if tag is not None:
            to_remove = []

            # meta = (values, tag_value)
            for fname, meta in self.features.items():
                _, tag_value = meta
                if tag_value == tag:
                    to_remove.append(fname)

            for fname in to_remove:
                del self.features[fname]

                if fname in df.columns:
                    self.removed_features[fname] = df.pop(fname)

            return

        # =============== NO INPUT ===============
        raise ValueError("You must provide either 'name' or 'tag'.")

           
    
    def get(self, name):
        """Retrieve a variable by name"""
        return self.features.get(name, (None, None))[0]
    
    def remove(self, name=None, tag=None):
        """Remove a variable by name and save it in removed_features"""
        if name is not None and name in self.features:
            self.removed_features[name] = self.features.pop(name)

        if tag is not None:
            to_remove = [fname for fname, (_, ftag) in self.features.items() if ftag == tag]

            if not to_remove:
                print(f"No features found with tag '{tag}'.")
                return

            for fname in to_remove:
                self.removed_features[fname] = self.features.pop(fname)

            return

    
    def restore(self, name):
        """Restore a removed variable"""
        if name in self.removed_features:
            self.features[name] = self.removed_features.pop(name)
            self.buffer[name] = self.get(name)
        else:
            print(f"Feature '{name}' not found in removed features.")
    
    def list_features(self, tag=None):
        """List feature names, optionally filtered by tag"""
        if tag is None:
            return list(self.features.keys())
        else:
            return [k for k, (_, t) in self.features.items() if t == tag]
    
    def list_removed_features(self):
        """List removed feature names"""
        return list(self.removed_features.keys())
    
    def all_features(self, tag=None):
        """Return a DataFrame of all features, optionally filtered by tag"""
        if tag is None:
            return pd.DataFrame({k: v for k, (v, _) in self.features.items()})
        else:
            return pd.DataFrame({k: v for k, (v, t) in self.features.items() if t == tag})

        


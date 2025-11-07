class ModelTracker:
    def __init__(self):
        self.models = {}
        self.results = []
        self.counter = 1
        self.preferred = None

    def add(self, model_obj, name=None, label=None, set_preferred=False):
        if name is None:
            name = f"v{self.counter}"
            self.counter += 1
        self.models[name] = model_obj
        self.results.append({
            'model': name,
            'F1': model_obj.score_f1,
            'threshold': model_obj.threshold,
            'improvement': model_obj.improvement
        })
        if set_preferred:
            self.preferred = name
        print(f"Model {name} saved\n")
    
    def to_df(self):
        import pandas as pd
        return pd.DataFrame(self.results)
    
    def get(self, name):
        return self.models.get(name)
    
    def get_most_recent(self):
        most_recent_name = self.results[-1]['model']
        return self.models[most_recent_name]
    
    def get_second_most_recent(self):
        second_recent_name = self.results[-2]['model']
        return self.models[second_recent_name]
    
    def remove_most_recent(self):
        last_name = self.results.pop(-1)['model']
        removed_model = self.models.pop(last_name)
        print(f"Removed most recent model: {last_name}")
        return removed_model
    
    def set_preferred(self, name):
        if name in self.models:
            self.preferred = name
            print(f"Model {name} is now preferred.")
        else:
            print(f"Model {name} not found.")
    
    def get_preferred(self):
        if self.preferred is None:
            print("No preferred model set.")
            return None
        return self.models[self.preferred]
    
    def get_by_name(self, name):
        return self.models[name]
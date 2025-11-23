import data

class X_tracker:
    def __init__(self):
        self.datasets = {}

    def add(self, name, X_with_targets):
        self.datasets[name] = X_with_targets
        print(f"Dataset {name} saved.\n")
    
    def get(self, name, split=False, toNpy=False):
        if split:
            return data.get_split_train_eval_data(self.datasets.get(name), toNpy=toNpy)
        else:
            return self.datasets.get(name).copy()
    
    def remove(self, name):
        if name in self.datasets:
            del self.datasets[name]
            print(f"Dataset {name} removed.")
        else:
            print(f"Dataset {name} not found.")

        
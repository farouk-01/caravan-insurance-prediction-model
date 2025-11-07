import logisticRegression

class Create:
    def __init__(self, w, b, threshold=None, improvement=""):
        self.w = w
        self.b = b
        self.threshold = threshold
        self.improvement = improvement
    
    def __repr__(self):
        return (f'Improvement: {self.improvement}')
    
    def get_y_pred(self, X):
        y_pred = logisticRegression.predict(X, self.w, self.b, self.threshold)
        return y_pred
    
    def print_stats(self, X, y):
        logisticRegression.print_model_stats(X, y, self.w, self.b, self.threshold)



import logisticRegression

class Model:
    def __init__(self, w, b, threshold=None, score_f1=None, score_auc=None, improvement=""):
        self.w = w
        self.b = b
        self.threshold = threshold
        self.score_f1 = score_f1
        self.score_auc = score_auc
        self.improvement = improvement
    
    def __repr__(self):
        return (f'Improvement: {self.improvement}')
    
    def get_y_pred(self, X):
        y_pred = logisticRegression.predict(X, self.w, self.b, self.threshold)
        return y_pred
    
    def print_stats(self, X, y):
        logisticRegression.print_model_stats(X, y, self.w, self.b, self.threshold, self.score_f1)

def create_model(X, y, learning_rate=0.005, extra_weight=1, improvement="", threshold_method="F1", to_print=False):
    w, b = logisticRegression.logistic_regression(X, y, learning_rate=learning_rate, extra_weight=extra_weight, to_print=to_print)
    if threshold_method == "F1":
        threshold, score_f1 = logisticRegression.f1_score_test(X, y, w, b)
    elif threshold_method == "Youden":
        threshold = logisticRegression.get_youden_threshold(X, y, w, b)
    model = Model(w, b, threshold, improvement=improvement, score_f1=score_f1)
    return model


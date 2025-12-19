import logisticRegression
import copy

class Model:
    def __init__(self, w, b, threshold=0.5, score_f1=None, score_auc=None, improvement=""):
        self.w = w
        self.b = b
        self.threshold = threshold
        self.improvement = improvement
    
    def __repr__(self):
        return (f'Improvement: {self.improvement}')
    
    def get_y_pred(self, X):
        y_pred = logisticRegression.predict(X, self.w, self.b, self.threshold)
        return y_pred
    
    def print_stats(self, X, y, threshold=None):
        if (self.improvement != ""):
            print(self.improvement)
        if threshold is None:
            t = self.threshold
        else:
            t = threshold
        logisticRegression.print_model_stats(X, y, self.w, self.b, t)

    def copy(self):
        return copy.deepcopy(self)

def create_model(X_train, y_train,X_val, y_val, learning_rate=0.001, iterations=1000, extra_weight=1, improvement="", 
                 threshold_method=None, set_threshold_to=None, l2_reg=False, l1_reg=False, lambda_const=None, to_print=False, score_f1 = None):
    w, b = logisticRegression.logistic_regression(X_train, y_train, X_val, y_val, learning_rate=learning_rate, iterations=iterations, extra_weight=extra_weight, l2_reg=l2_reg, l1_reg=l1_reg, lambda_const=lambda_const, to_print=to_print)
    if set_threshold_to is not None: threshold=set_threshold_to
    elif threshold_method == "F1":
        threshold, score_f1 = logisticRegression.f1_score_threshold(X_val, y_val, w, b)
    elif threshold_method == "Youden":
        threshold = logisticRegression.get_youden_threshold(X_val, y_val, w, b)
    elif threshold_method is None:
        threshold = 0.5
    model = Model(w, b, threshold, improvement=improvement, score_f1=score_f1)
    return model
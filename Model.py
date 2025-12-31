import logisticRegression
import copy
from Trackers import FeatureTracker
from sklearn.metrics import confusion_matrix

class Model:
    def __init__(self, w, b, threshold=0.5, name="",
                 score_f1=None, score_auc=None, improvement="",
                 cols=None, X_train=None, y_train=None, X_val=None, y_val=None):
        self.w = w
        self.b = b
        self.threshold = threshold
        self.name = name
        self.improvement = improvement

        self.cols = cols
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.X_eval = None
        self.y_eval = None
    
    def __repr__(self):
        if self.X_val is None or self.y_val is None:
            return "Aucune validation data"
        stats = logisticRegression.get_model_stats(
            self.X_val, self.y_val,
            self.w, self.b,
            self.threshold
        )
        return (
            f"Threshold   : {stats['threshold']:.4f}\n"
            f"{stats["confusion_matrix"]}"
        )
    def get_y_pred(self, X):
        y_pred = logisticRegression.predict(X, self.w, self.b, self.threshold)
        return y_pred
    
    def print_eval_stats(self, threshold=None, print_metrics=False):
        if self.X_eval is None or self.y_eval is None: raise ValueError("pls set eval set")

        if threshold is None: threshold = self.threshold

        self.print_stats(self.X_eval, self.y_eval, threshold=threshold, print_metrics=print_metrics)

    def print_train_stats(self, threshold=None, print_metrics=False):
        if self.X_train is None or self.y_train is None: raise ValueError("pls set train set")

        if threshold is None: threshold = self.threshold
        
        self.print_stats(self.X_train, self.y_train, threshold=threshold, print_metrics=print_metrics)

    def print_val_stats(self, threshold=None, print_metrics=False):
        if self.X_val is None or self.y_val is None: raise ValueError("pls set val set")

        if threshold is None: threshold = self.threshold
        
        self.print_stats(self.X_val, self.y_val, threshold=threshold, print_metrics=print_metrics)
    
    def print_stats(self, X, y, threshold=None, print_metrics=True):
        if (self.improvement != ""):
            print(self.improvement)
        if threshold is None:
            t = self.threshold
        else:
            t = threshold
        logisticRegression.print_model_stats(X, y, self.w, self.b, t, print_metrics)

    def copy(self):
        return copy.deepcopy(self)
    
    def predict_probas(self, X):
        return logisticRegression.predict_probas(X, self.w, self.b)
    
    def make_conf_matrix(self, X, y, threshold=None):
        if threshold is None: threshold = self.threshold
        y_pred = logisticRegression.predict(X, self.w, self.b, threshold)
        return confusion_matrix(y, y_pred)
    
    def get_conf_matrix(self, feature_tracker=None, of_train_set=False, threshold=None):
        if feature_tracker is None: raise ValueError('pls provide feature_tracker')

        if of_train_set : X, y, *_ = feature_tracker.return_split_train_eval()
        else: _, _, X, y = feature_tracker.return_split_train_eval()
        
        if threshold is None: threshold = self.threshold
        
        return self.make_conf_matrix(X, y, threshold=threshold)
    
def create_model(X_train, y_train,X_val, y_val, cols=None, learning_rate=0.001, iterations=1000, class_weight=1, name="", improvement="", 
                 threshold_method=None, threshold=None, l2_reg=False, l1_reg=False, lambda_const=None, to_print=False, score_f1 = None):
    w, b = logisticRegression.logistic_regression(
        X_train, y_train, X_val, y_val, 
        learning_rate=learning_rate, iterations=iterations, class_weight=class_weight, 
        l2_reg=l2_reg, l1_reg=l1_reg, lambda_const=lambda_const, 
        to_print=to_print
    )
    if threshold is not None: pass
    elif threshold_method == "F1":
        threshold, *_ = logisticRegression.f1_score_threshold(X_val, y_val, w, b)
    elif threshold_method == "Youden":
        threshold = logisticRegression.get_youden_threshold(X_val, y_val, w, b)
    else: threshold = 0.1
    model = Model(w, b, threshold, name=name, improvement=improvement, score_f1=score_f1, X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val, cols=cols)
    return model
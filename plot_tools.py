import numpy as np
import matplotlib.pyplot as plt
import logisticRegression as logReg
from sklearn.metrics import f1_score, precision_score, recall_score


def plot_threshold_metrics(model, X, y, step=0.01):
    thresholds = np.arange(0,1 + step ,step)
    f1_scores = []
    precision_scores = []
    recall_scores = []

    for t in thresholds:
        y_pred = logReg.predict(X, model.w, model.b, t)
        f1_scores.append(f1_score(y, y_pred))
        precision_scores.append(precision_score(y, y_pred, zero_division=0))
        recall_scores.append(recall_score(y, y_pred, zero_division=0))

    plt.figure(figsize=(8,5))
    plt.plot(thresholds, f1_scores, label="F1 Score", color="blue")
    plt.plot(thresholds, precision_scores, label="Precision", color="green")
    plt.plot(thresholds, recall_scores, label="Recall", color="red")
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    plt.title("Threshold vs F1 / Precision / Recall")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_weights_effects(X_train, y_train, X_val, y_val, threshold, ratio, weights_to_test, learning_rate):
    recalls_scores = []
    precision_scores = []
    f1_scores = []

    for weight in weights_to_test:
        w, b = logReg.logistic_regression(X_train, y_train, learning_rate, extra_weight=ratio*weight, to_print=False)
        y_pred = logReg.predict(X_val, w, b, threshold)
        recalls_scores.append(recall_score(y_val, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred))

    best_idx = np.argmax(f1_scores)
    best_weight = weights_to_test[best_idx]
    best_f1 = f1_scores[best_idx]

    plt.figure(figsize=(8,5))
    plt.plot(weights_to_test, recalls_scores, label='Recall', marker='o')
    plt.plot(weights_to_test, precision_scores, label='Precision', marker='o')
    plt.plot(weights_to_test, f1_scores, label='F1 Score', marker='o')
    plt.scatter(best_weight, best_f1, color='red', s=100, label=f'Best F1: {best_f1:.3f} at weight {best_weight}')
    plt.xlabel('Positive Class Weight')
    plt.ylabel('Score')
    plt.title('Effect of Positive Class Weight on Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()

def lr_grid_search(X_train, y_train, X_val, y_val, lrs, iterations=1000, toPlot=False, **kwargs):
    val_losses = []
    f1_scores = []
    for lr in lrs:
        w,b = logReg.logistic_regression(X_train, y_train, learning_rate=lr, iterations=iterations, to_print=False, **kwargs)
        val_pred = logReg.predict(X_val, w, b, 0.5)
        val_losses.append(logReg.cost_function(X_val, y_val, w, b))
        f1_scores.append(f1_score(y_val, val_pred, zero_division=0))
    if toPlot:
        fig, ax1 = plt.subplots()

        ax1.semilogx(lrs, val_losses, marker='o', color='tab:red', label='val loss')
        ax1.set_xlabel('learning rate')
        ax1.set_ylabel('val loss', color='tab:red')
        #ax1.invert_yaxis()
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        ax2.semilogx(lrs, f1_scores, marker='o', color='tab:blue', label='val F1')
        ax2.set_ylabel('val F1', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        ax1.grid(which='both', linestyle='--', linewidth=0.5)

        plt.title('Validation Loss & F1 vs Learning Rate')
        plt.xticks(lrs, [f"{lr:.3f}" for lr in lrs])
        plt.show()
    return val_losses, f1_scores
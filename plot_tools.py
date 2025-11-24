import numpy as np
import matplotlib.pyplot as plt
import logisticRegression as logReg
from sklearn.metrics import f1_score, precision_score, recall_score

def plot_weights_effects(X_train, y_train, X_val, y_val, threshold, weights_to_test, learning_rate=0.01, **kwargs):
    recalls_scores = []
    precision_scores = []
    f1_scores = []

    for weight in weights_to_test:
        w, b = logReg.logistic_regression(X_train, y_train, X_val, y_val, learning_rate, extra_weight=weight, to_print=False, **kwargs)
        y_pred = logReg.predict(X_val, w, b, threshold)
        recalls_scores.append(recall_score(y_val, y_pred, zero_division=0))
        precision_scores.append(precision_score(y_val, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_val, y_pred))

    best_idx_f1 = np.argmax(f1_scores)
    best_weight_f1 = weights_to_test[best_idx_f1]
    best_f1 = f1_scores[best_idx_f1]
    corresponding_recall = recalls_scores[best_idx_f1]

    best_idx_recall = np.argmax(recalls_scores)
    best_weight_recall = weights_to_test[best_idx_recall]
    best_recall = recalls_scores[best_idx_recall]
    corresponding_f1 = f1_scores[best_idx_recall]

    plt.figure(figsize=(8,5))
    plt.plot(weights_to_test, recalls_scores, label='Recall', marker='o')
    plt.plot(weights_to_test, precision_scores, label='Precision', marker='o')
    plt.plot(weights_to_test, f1_scores, label='F1 Score', marker='o')
    plt.scatter(best_weight_f1, best_f1, color='black', s=100, label=f'Best F1: {best_f1:.3f} with recall {corresponding_recall:.3f} at weight {best_weight_f1:.3f}')
    plt.scatter(best_weight_recall, best_recall, color='red', s=100, 
                label=f'Best Recall: {best_recall:.3f} with F1 {corresponding_f1:.3f} at weight {best_weight_recall:.3f}')    
    plt.xlabel('Positive Class Weight')
    plt.ylabel('Score')
    plt.title('Effect of Positive Class Weight on Metrics')
    plt.legend()
    plt.grid(True)
    plt.show()



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

def lr_grid_search(X_train, y_train, X_val, y_val, lrs, iterations=1000, plotF1=False, toPrint=False, toPlot=False, **kwargs):
    val_losses = []
    f1_scores = []
    cost_min = []
    cost_max = []
    for lr in lrs:
        w,b, train_costs, val_costs = logReg.logistic_regression(
            X_train, y_train, X_val, y_val,
            learning_rate=lr, 
            iterations=iterations, 
            return_costs=True,  
            to_print=toPrint,
            **kwargs
        )
        #attention c'est le validation test ici, donc pas de extra_weight
        val_losses.append(val_costs[-1])

        val_pred = logReg.predict(X_val, w, b, 0.5)
        f1_scores.append(f1_score(y_val, val_pred, zero_division=0))

        last_10cost = max(1, int(0.1 * iterations))
        cost_min.append(np.min(val_costs[-last_10cost:]))
        cost_max.append(np.max(val_costs[-last_10cost:]))

    best_idx = np.argmax(f1_scores)
    best_lr = lrs[best_idx]
    best_f1 = f1_scores[best_idx]
    if toPlot:
        fig, ax1 = plt.subplots(figsize=(12,8))

        ax1.semilogx(lrs, val_losses, marker='o', color='tab:red', label='val loss')
        ax1.fill_between(lrs, cost_min, cost_max, color='orange', alpha=0.5, label='Cost Oscillation')
        ax1.set_xlabel('learning rate')
        ax1.set_ylabel('val loss', color='tab:red')
        #ax1.invert_yaxis()
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        if plotF1:
            ax2.semilogx(lrs, f1_scores, marker='o', color='tab:blue', label='val F1')
            ax2.scatter(best_lr, best_f1, s=100, color='green', label=f'Best f1: {best_f1:.4f} at lr of {best_lr:.6f}')
        ax2.set_ylabel('val F1', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')


        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        plt.title('Validation Loss & F1 vs Learning Rate')
        #plt.xticks(lrs, [f"{lr:.3f}" for lr in lrs])
        plt.show()
    return val_losses, f1_scores

def plot_convergence(X_train, y_train, X_val, y_val, learning_rate, epochs_max, to_print=False, **kwargs):
    w, b, train_costs, val_costs = logReg.logistic_regression(
        X_train, y_train, X_val, y_val,
        learning_rate=learning_rate,
        iterations=epochs_max,
        return_costs=True,
        to_print=to_print,
        **kwargs 
    )
    iterations = range(1, epochs_max + 1)

    plt.figure(figsize=(10, 6))
    #plt.plot(iterations, train_costs, label='Training Loss', color='blue')
    plt.plot(iterations, val_costs, label='Validation Loss', color='red')
    
    plt.title(f'Convergence du Modele (LR={learning_rate:.6f})')
    plt.xlabel('Epoques/Iterations')
    plt.ylabel('Perte (Loss)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()


def train_val_accuracy_plot(X_train, y_train, X_val, y_val, learning_rate, epochs_max, **kwargs):
    w, b, train_costs, val_costs, w_history, b_history = logReg.logistic_regression(
        X_train, y_train, X_val, y_val,
        learning_rate=learning_rate,
        iterations=epochs_max,
        return_costs=True,
        return_weight_history=True,
        **kwargs 
    )
    
    num_epochs_run = len(w_history) 
    iterations_run = range(1, num_epochs_run + 1)
        
    train_accuracies = []
    val_accuracies = []

    for epoch_index in range(num_epochs_run):
        w_epoch = w_history[epoch_index]
        b_epoch = b_history[epoch_index] 
        
        y_train_pred = logReg.predict(X_train, w_epoch, b_epoch, 0.5)
        y_val_pred = logReg.predict(X_val, w_epoch, b_epoch, 0.5)
        
        train_accuracies.append(np.mean(y_train_pred == y_train))
        val_accuracies.append(np.mean(y_val_pred == y_val))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations_run, train_accuracies, label='Training Accuracy', color='blue') # Utiliser iterations_run
    plt.plot(iterations_run, val_accuracies, label='Validation Accuracy', color='orange') # Utiliser iterations_run
    
    plt.title(f'Accuracy vs Epochs (LR={learning_rate:.6f})')
    plt.xlabel('Epoques/Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def epochs_grid_search(X_train, y_train, X_val, y_val, epochs_list, learning_rate, **kwargs):
    val_losses = []
    f1_scores = []

    for epoch in epochs_list:
        w, b = logReg.logistic_regression(
            X_train, y_train,
            learning_rate=learning_rate,
            iterations=epoch,
            to_print=False,
            **kwargs
        )
        val_pred = logReg.predict(X_val, w, b, 0.5)
        val_losses.append(logReg.cost_function(X_val, y_val, w, b))
        f1_scores.append(f1_score(y_val, val_pred, zero_division=0))

    fig, ax1 = plt.subplots()

    ax1.plot(epochs_list, val_losses, marker='o', color='tab:red', label='val loss')
    ax1.set_xlabel('epochs')
    ax1.set_ylabel('val loss', color='tab:red')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    ax2 = ax1.twinx()
    ax2.plot(epochs_list, f1_scores, marker='o', color='tab:blue', label='val F1')
    ax2.set_ylabel('val F1', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:blue')

    ax1.grid(which='both', linestyle='--', linewidth=0.5)
    plt.title(f'Validation Loss & F1 vs Epochs (LR={learning_rate})')
    plt.show()

    return val_losses, f1_scores
import numpy as np
import matplotlib.pyplot as plt
import logisticRegression as logReg
from sklearn.metrics import f1_score, precision_score, recall_score
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import seaborn as sns

def plot_weights_effects(X_train, y_train, X_val, y_val, threshold, weights_to_test, learning_rate=0.01, **kwargs):
    recalls_scores = []
    precision_scores = []
    f1_scores = []

    for weight in weights_to_test:
        w, b = logReg.logistic_regression(X_train, y_train, X_val, y_val, learning_rate, class_weight=weight, to_print=False, **kwargs)
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



def plot_threshold_metrics(model, feature_tracker, step=0.01,  range_min=0, range_max=1, plot_only_recall=True):
    thresholds = np.arange(range_min, range_max + step ,step)
    f1_scores = []
    precision_scores = []
    recall_scores = []
    X_train, y_train, X, y = feature_tracker.return_split_train_eval()

    if plot_only_recall:
        for t in thresholds:
            y_pred = logReg.predict(X, model.w, model.b, t)
            recall_scores.append(recall_score(y, y_pred, zero_division=0))
        plt.figure(figsize=(8,5))
        plt.plot(thresholds, recall_scores, label="Recall", color="red")
        plt.title("Threshold vs Recall")
    else:
        for t in thresholds:
            y_pred = logReg.predict(X, model.w, model.b, t)
            f1_scores.append(f1_score(y, y_pred))
            #precision_scores.append(precision_score(y, y_pred, zero_division=0))
            recall_scores.append(recall_score(y, y_pred, zero_division=0))
        plt.figure(figsize=(8,5))
        plt.plot(thresholds, f1_scores, label="F1 Score", color="blue")
        #plt.plot(thresholds, precision_scores, label="Precision", color="green")
        plt.plot(thresholds, recall_scores, label="Recall", color="red")
        plt.title("Threshold vs F1 / Recall")

    
    plt.xlabel("Threshold")
    plt.ylabel("Score")
    
    plt.legend()
    plt.grid(True)
    plt.show()

def lr_grid_search(X_train, y_train, X_val, y_val, lrs, iterations=1000, plot_f1=False, to_print=False, to_plot=False, plot_recall=False, **kwargs):
    val_losses = []
    f1_scores = []
    recall_scores = []
    cost_min = []
    cost_max = []
    for lr in lrs:
        w,b, train_costs, val_costs = logReg.logistic_regression(
            X_train, y_train, X_val, y_val,
            learning_rate=lr, 
            iterations=iterations, 
            return_costs=True,  
            to_print=to_print,
            **kwargs
        )
        val_losses.append(val_costs[-1])

        val_pred = logReg.predict(X_val, w, b, 0.5)
        f1_scores.append(f1_score(y_val, val_pred, zero_division=0))
        recall_scores.append(recall_score(y_val, val_pred, zero_division=0))

        last_10cost = max(1, int(0.1 * iterations))
        cost_min.append(np.min(val_costs[-last_10cost:]))
        cost_max.append(np.max(val_costs[-last_10cost:]))

    best_idx = np.argmax(f1_scores)
    best_lr = lrs[best_idx]
    best_f1 = f1_scores[best_idx]
    best_recall = recall_scores[best_idx]

    if to_plot:
        fig, ax1 = plt.subplots()

        ax1.semilogx(lrs, val_losses, marker='o', color='tab:red', label='val loss')
        ax1.fill_between(lrs, cost_min, cost_max, color='orange', alpha=0.5, label='Cost Oscillation')
        ax1.set_xlabel('learning rate')
        ax1.set_ylabel('val loss', color='tab:red')
        ax1.tick_params(axis='y', labelcolor='tab:red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('score', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:blue')

        if plot_f1:
            ax2.semilogx(lrs, f1_scores, marker='o', color='tab:blue', label='val F1')
            ax2.scatter(best_lr, best_f1, s=100, color='green', label=f'Best f1: {best_f1:.4f} at lr of {best_lr:.6f}')

        if plot_recall:
            ax2.semilogx(lrs, recall_scores, marker='o', color='tab:purple', label='val Recall')
            ax2.scatter(best_lr, best_recall, s=80, color='tab:purple',
                        label=f'Recall au best f1: {best_recall:.4f}')



        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='best')
        ax1.grid(which='both', linestyle='--', linewidth=0.5)
        plt.title('Validation Loss & score vs Learning Rate')
        #plt.xticks(lrs, [f"{lr:.3f}" for lr in lrs])
        plt.show()
    return val_losses, f1_scores, recall_scores

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

def plot_hist_comparison( column, X_tp=None, X_fp=None, X_fn=None, X_tn=None, bins=20):
    plt.figure(figsize=(12, 7))
    if X_tp is not None:
        mean_tp = X_tp[column].mean()
        median_tp = X_tp[column].median()
        plt.axvline(mean_tp, color='green', linestyle='dashed', linewidth=2, 
                    label=f'Mean TP: {mean_tp:.2f}')
        plt.axvline(median_tp, color='green', linestyle=':', linewidth=2, 
                    label=f'Median TP: {median_tp:.2f}')
        plt.hist(X_tp[column], bins=bins, alpha=0.5, label='True Positives', color='green')
    if X_fp is not None:
        mean_fp = X_fp[column].mean()
        median_fp = X_fp[column].median()
        plt.axvline(mean_fp, color='orange', linestyle='dashed', linewidth=2, 
                    label=f'Mean FP: {mean_fp:.2f}')
        plt.axvline(median_fp, color='orange', linestyle=':', linewidth=2, 
                    label=f'Median FP: {median_fp:.2f}')
        plt.hist(X_fp[column], bins=bins, alpha=0.5, label='False Positives', color='orange')
    if X_tn is not None:
        mean_tn = X_tn[column].mean()
        median_tn = X_tn[column].median()
        plt.axvline(mean_tn, color='blue', linestyle='dashed', linewidth=2, 
                    label=f'Mean TN: {mean_tn:.2f}')
        plt.axvline(median_tn, color='blue', linestyle=':', linewidth=2, 
                    label=f'Median TN: {median_tn:.2f}')
        plt.hist(X_tn[column], bins=bins, alpha=0.5, label='True Negatives', color='blue')
    if X_fn is not None:
        mean_fn = X_fn[column].mean()
        median_fn = X_fn[column].median()
        plt.axvline(mean_fn, color='red', linestyle='dashed', linewidth=2, 
                    label=f'Mean FN: {mean_fn:.2f}')
        plt.axvline(median_fn, color='red', linestyle=':', linewidth=2, 
                    label=f'Median FN: {median_fn:.2f}')
        plt.hist(X_fn[column], bins=bins, alpha=0.5, label='False Negatives', color='red')

    plt.xlabel(column)
    plt.title(f'Histogram pour {column}')
    plt.legend()
    plt.grid(True)
    plt.show()

def get_df_groups(TP_FN=False, FP_FN=False):
    if TP_FN == FP_FN:
        raise ValueError("Choisir exactement un : TP_FN=True ou FP_FN=True")
    elif TP_FN:
        var1, var2 = "TP", "FN"
    else:
        var1, var2 = "FP", "FN"
    return var1, var2

def plot_PCA(cat_cols, num_cols, df_profiles, return_plot=False, TP_FN=False, FP_FN=False):
    var1, var2 = get_df_groups(TP_FN, FP_FN)
    df_plot = df_profiles[df_profiles["Group"].isin([var1,var2])].copy()

    scaler = StandardScaler()
    
    X_cat = pd.get_dummies(df_plot[cat_cols], drop_first=False)

    if num_cols is not None:
        X_num = df_plot[num_cols].astype(float)
        X_num = pd.DataFrame(
            scaler.fit_transform(X_num),
            index = df_plot.index,
            columns=num_cols
        )
    
        X = pd.concat([X_num, X_cat], axis=1)
    else:
        X = X_cat

    pca = PCA(n_components=2, random_state=0)
    Z = pca.fit_transform(X)

    loadings = pd.DataFrame(
        pca.components_.T, 
        index=X.columns,    
        columns=["PC1", "PC2"]
    )

    df_emb = pd.DataFrame(Z, columns=["PC1", "PC2"], index=df_plot.index)
    df_emb["Group"] = df_plot["Group"].values

    fig, ax = plt.subplots()

    sns.scatterplot(data=df_emb, x="PC1", y="PC2", hue="Group", hue_order=[var1, var2], alpha=0.7, s=25, ax=ax)
    ax.set_title("PCA")
    fig.tight_layout()

    if return_plot: return loadings, fig, ax
    else:
        plt.show()
        return loadings

def plot_LDA(categorie_cols, continue_cols, df_profiles, one_hot_cat=False, TP_FN=False, FP_FN=False, return_vars=False):
    var1, var2 = get_df_groups(TP_FN, FP_FN)
    df_plot = df_profiles[df_profiles["Group"].isin([var1, var2])].copy()

    scaler = StandardScaler()
    
    X_num = df_plot[continue_cols].astype(float)
    X_num = pd.DataFrame(
        scaler.fit_transform(X_num),
        index = df_plot.index,
        columns=continue_cols
    )

    if one_hot_cat: X_cat = pd.get_dummies(df_plot[categorie_cols], drop_first=False)
    else:
        X_cat = df_plot[categorie_cols].astype("Int64")
        X_cat = pd.DataFrame(
            scaler.fit_transform(X_cat),
            index=df_plot.index,
            columns=categorie_cols
        )
    X = pd.concat([X_num, X_cat], axis=1)
    y = df_plot["Group"].values

    lda = LinearDiscriminantAnalysis(n_components=1)
    Z = lda.fit_transform(X, y) 

    loadings = pd.DataFrame(
        {"LD1": lda.coef_.ravel()},
        index=X.columns
    ).sort_values("LD1", key=lambda s: s.abs(), ascending=False)

    df_emb = pd.DataFrame({"LD1": Z.ravel()}, index=df_plot.index)
    df_emb["Group"] = y

    if return_vars:
        return loadings, df_emb, lda, scaler, X.columns

    # plt.figure(figsize=(7, 3))
    sns.violinplot(data=df_emb, x="Group", y="LD1", order=[var1, var2], inner=None, cut=0)
    sns.stripplot(data=df_emb, x="Group", y="LD1", order=[var1, var2], alpha=0.5, size=3)
    plt.title("LDA (LD1)")
    plt.tight_layout()

    
    return loadings

def plot_LDA_all_groups(cols, df_profiles):
    var1, var2 = 'TP', 'FN'

    df_all = df_profiles[df_profiles["Group"].isin(['TP', 'FN', "FP", "TN"])].copy()
    X_all = pd.get_dummies(df_all[cols].astype("category"), drop_first=False)

    mask_train = df_all["Group"].isin([var1, var2])
    X_train = X_all.loc[mask_train]
    y_train = df_all.loc[mask_train, "Group"].values

    scaler = StandardScaler(with_mean=False)
    X_train_scaled = scaler.fit_transform(X_train)
    X_all_scaled = scaler.transform(X_all)

    lda = LinearDiscriminantAnalysis(n_components=1)
    lda.fit(X_train_scaled, y_train)

    Z_all = lda.transform(X_all_scaled).ravel()

    loadings = (
        pd.DataFrame({"LD1": lda.coef_.ravel()}, index=X_all.columns)
        .sort_values("LD1", key=lambda s: s.abs(), ascending=False)
    )

    df_emb = pd.DataFrame({
        "LD1": Z_all,
        "Group": df_all["Group"].values
    }, index=df_all.index)

    order = ['TP', 'FN', "FP", "TN"]

    sns.violinplot(data=df_emb, x="Group", y="LD1", order=order, inner=None, cut=0)
    sns.stripplot(data=df_emb, x="Group", y="LD1", order=order, alpha=0.5, size=3)

    plt.title("LDA (LD1) de {} vs {}".format(var1, var2))
    plt.tight_layout()

    return loadings

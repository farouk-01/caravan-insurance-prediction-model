from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import seaborn as sns
import numpy as np
import pandas as pd
import scipy.stats as st
import logisticRegression
import data

def make_mi_scores(X, y, discrete_features):
    #Beware not to use a regression scoring function with a classification problem, you will get useless results.
    # if isContinuous:
    #     mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    # else:
    mi_scores = mutual_info_classif(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    plt.figure(dpi=100, figsize=(10, len(scores) * 0.15)) 
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")
    plt.show()

def create_interaction_terms(df, var1, var2, terms:str, isVar1OneHot=False):
    items = terms.replace(" ", "").split(",")
    pairs = []
    new_cols ={}
    for item in items:
        a,b = item.split("x")
        pairs.append((int(a), int(b)))

    for a,b in pairs:
        col_name = f"{var1}_{a}_x_{var2}_{b}"
        if isVar1OneHot:
            new_cols[col_name] = (df[f'{var1}_{a}'] & (df[var2] == b)).astype(int) 
        else: 
            new_cols[col_name] = ((df[var1] == a) & (df[var2] == b)).astype(int)
    if new_cols:
        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df

def find_correlated_cols(df, threshold=0.95, toPlot=True):
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().abs()

    if toPlot:
        #plt.figure(figsize=(10,8))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, cbar=True)
        plt.title("Matrice de correlation")
        plt.show()

    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    strong_pairs = []
    for col in upper.columns:
        for row in upper.index:
            corr_value = upper.loc[row, col]
            if corr_value > threshold:
                strong_pairs.append((row, col))
    return strong_pairs

def fisher_info(X, w, b):
    z = X @ w + b
    p = logisticRegression.sigmoid(z)
    W = np.diag(p * (1-p))
    fi = X.T @ W @ X
    return fi

def ols(X, y):
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    return w

def regressionLineaire(X, y):
    w = ols(X, y)
    residual = y - X @ w
    return w, residual

def auxRegLin(X, eps, tol=1e-12):
    var_residual = np.sum(eps**2)
    xm = np.mean(X)
    var_X = np.sum((X - xm)**2)
    if var_X <= tol: return 0
    return (1 - var_residual/var_X)
    
def vif(X, cols, return_coef=False):
    vif_res = {}
    all_coef_dfs = []
    for c in cols:
        X_i = X[c].to_numpy()
        X_hat_i = X.drop(c, axis=1).to_numpy()
        X_hat = np.column_stack([np.ones(len(X_i)), X_hat_i])
        w, residu = regressionLineaire(X_hat, X_i)
        R2 = auxRegLin(X_i, residu)
        vif = np.inf if np.isclose(1 - R2, 0) else 1 / (1 - R2)
        vif_res[c] = vif
        if return_coef:
            coef_df = pd.DataFrame({
                'Target': [c] * len(w),
                'Variable': ['Intercept'] + list(X.drop(c, axis=1).columns),
                'coef': w
            })
            all_coef_dfs.append(coef_df)
            # print(coef_df.sort_values(by='coef', key=abs, ascending=False).head(10).to_markdown())
            # print()
    if return_coef:
        coef_df = pd.concat(all_coef_dfs, ignore_index=True)
        return coef_df
    return pd.DataFrame.from_dict(vif_res, orient='index', columns=['VIF']).sort_values(by='VIF', ascending=False)

def get_vif_coef(X_train, target, to_markdown=True):
    cols = X_train.columns
    df_vif = vif(X_train, cols, return_coef=True)
    if to_markdown: return df_vif[df_vif['Target'] == target].sort_values(by='coef', key=abs, ascending=False).head().to_markdown()
    return df_vif[df_vif['Target'] == target].sort_values(by='coef', key=abs, ascending=False).head()

def get_negative_variance_info(fi, X_cols, huge=1e10):
    cov_matrix = np.linalg.inv(fi)
    #std = np.sqrt(np.diag(cov_matrix))
    coef_df = pd.DataFrame({
        'feature': X_cols,
        'var': np.diag(cov_matrix),
    })

    print('negatif count in variance:               ', coef_df[coef_df['var'] < 0].shape[0])
    print('negatif gigantesque count in variance:   ', coef_df[coef_df['var'] <= -huge].shape[0])
    print('Gigantesque count in variance:           ', coef_df[coef_df['var'] >= huge].shape[0])

    huge_neg_coeff_cols = coef_df[coef_df['var'] < 0]['feature'].values
    huge_pos_coeff_cols = coef_df[coef_df['var'] >= huge]['feature'].values

    inf_coef_cols = np.concatenate((huge_neg_coeff_cols, huge_pos_coeff_cols), axis=0)
    return coef_df, inf_coef_cols


def get_constant_and_rare_cols(coef_df, X_train, y_train, huge=1e10):
    huge_neg_coeff_cols = coef_df[coef_df['var'] < 0]['feature'].values
    huge_pos_coeff_cols = coef_df[coef_df['var'] >= huge]['feature'].values

    inf_coef_cols = np.concatenate((huge_neg_coeff_cols, huge_pos_coeff_cols), axis=0)
    X_train_inf_coef = X_train[inf_coef_cols].copy()

    cols_with_zeros_targets = []
    cols_with_all_targets = []
    cols_with_rare_outcomes = {}

    for c in inf_coef_cols:
        crossTab = pd.crosstab(X_train_inf_coef[c], y_train)

        if 1 in crossTab.index:
            #print(f'\n{c} = 1')
            not_target_count = crossTab.loc[1, 0]
            target_count = crossTab.loc[1, 1]
            #print(f"    (CARAVAN=0) : {not_target_count}")

            if target_count == 0:
                #print(f'    Prédit toujours CARAVAN=0')
                cols_with_zeros_targets.append(c)
            elif not_target_count == 0:
                #print(f'    Prédit toujours CARAVAN=1')
                cols_with_all_targets.append(c)
            else:
            #print(f"    (CARAVAN=1): {target_count}")
                cols_with_rare_outcomes[c] = np.array([target_count, not_target_count])
        else:
            print('what?', c)
    return cols_with_zeros_targets, cols_with_all_targets, cols_with_rare_outcomes

def or_with_ic(model, X_train, cols, printFull=False):
    coeff = model.w 
    bias = model.b 
    odds_ratio = np.exp(coeff) 
    confiance = 0.95

    #X_train_np, X_val_np, y_train_np, y_val_np = data.get_split_train_eval_data(X, toNpy=True)

    fi = fisher_info(X_train, coeff, bias)
    cov_matrix = np.linalg.inv(fi)
    std = np.sqrt(np.diag(cov_matrix))

    surface = 1 - (1-confiance)/2
    z = st.norm.ppf(surface)

    bi = coeff - z * std
    bs = coeff + z * std

    bi_or = np.exp(bi)
    bs_or = np.exp(bs)
        
    coef_df = pd.DataFrame({
        'feature': cols,
        '$\\beta_n$': coeff,
        '$OR$': odds_ratio,
        '$bi_{OR}$': bi_or,
        '$bs_{OR}$': bs_or
    })

    coef_df = coef_df.round(4)

    mask_signif = (coef_df['$bi_{OR}$'] > 1) | (coef_df['$bs_{OR}$'] < 1)
    if printFull: return (coef_df.sort_values(by='$OR$', key=abs, ascending=False).to_markdown())
    return (coef_df[mask_signif].sort_values(by='$bi_{OR}$', ascending=False).to_markdown())
    

def split_val_results(X_val, y_val, w, b, threshold):
    X_val_np = X_val.to_numpy()
    y_pred = logisticRegression.predict(X_val_np, w, b, threshold)

    tp_indices = np.where((y_val == 1) & (y_pred == 1))[0]
    fp_indices = np.where((y_val == 0) & (y_pred == 1))[0]
    tn_indices = np.where((y_val == 0) & (y_pred == 0))[0]
    fn_indices = np.where((y_val == 1) & (y_pred == 0))[0]
    X_tp = X_val_np[tp_indices]
    X_fp = X_val_np[fp_indices]
    X_tn = X_val_np[tn_indices]
    X_fn = X_val_np[fn_indices]
    X_tp_df = pd.DataFrame(data=X_tp, columns=X_val.columns)
    X_fp_df = pd.DataFrame(data=X_fp, columns=X_val.columns)
    X_tn_df = pd.DataFrame(data=X_tn, columns=X_val.columns)
    X_fn_df = pd.DataFrame(data=X_fn, columns=X_val.columns)

    return X_tp_df, X_fp_df, X_tn_df, X_fn_df

def get_df_val_analysis(X_val, y_val, w, b, threshold, raw=False, quantile=None):
    X_tp_df, X_fp_df, X_tn_df, X_fn_df = split_val_results(X_val, y_val, w, b, threshold)
    
    if raw:
        tp = X_tp_df.copy(); tp["Group"] = "TP"
        fp = X_fp_df.copy(); fp["Group"] = "FP"
        tn = X_tn_df.copy(); tn["Group"] = "TN"
        fn = X_fn_df.copy(); fn["Group"] = "FN"

        df_profiles = pd.concat([tn, fp, fn, tp], axis=0, ignore_index=True)
        return df_profiles
    elif quantile is None:
        profile_tp = X_tp_df.mean()
        profile_fp = X_fp_df.mean()
        profile_tn = X_tn_df.mean()
        profile_fn = X_fn_df.mean()
    else:
        profile_tp = X_tp_df.quantile(quantile)
        profile_fp = X_fp_df.quantile(quantile)
        profile_tn = X_tn_df.quantile(quantile)
        profile_fn = X_fn_df.quantile(quantile)

    #var_to_check = ['total_contribution', 'MHKOOP', 'diff_educ_mid_bas', 'MRELSA', 'MFGEKIND' ,'MFWEKIND', 'MGODGE', 'MRELGE', 'MRELOV', 'MGODPR'] 
    #var_to_check = ['total_contribution', 'MHKOOP', 'diff_educ_mid_bas', 'MRELSA', 'MFGEKIND' ,'MFWEKIND', 'MGODGE', 'MRELGE', 'MRELOV', 'MGODPR', 'MGODRK'] 

    df_profiles = pd.DataFrame({
        'TN': profile_tn,
        'FP': profile_fp,
        'FN': profile_fn,
        'TP': profile_tp
    })
    df_profiles.index.name = 'Variable'
    return df_profiles

def print_df_analysis_html(df):
    def colorize_row(row):
        row_min = row.min()
        row_max = row.max()
        norm = mcolors.Normalize(vmin=row_min, vmax=row_max)
        cmap = cm.get_cmap("YlOrRd")
        
        return [f'background-color: rgba({int(rgba[0]*255)}, {int(rgba[1]*255)}, {int(rgba[2]*255)}, {rgba[3]:.2f}); color: black;'
                for val in row for rgba in [cmap(norm(val))]]

    # Apply row-wise coloring
    styler = df.style.apply(colorize_row, axis=1)
    styler.set_table_attributes('style="border-collapse:collapse; width:60%;"')

    # Display HTML
    html = styler.format("{:.2f}").to_html()
    return html
    #print(html)

def get_target_count_of_variables(X, y, vars, markdown=True):
    dfs = {
        f"{v} (CARAVAN=1)": X.loc[y == 1, v].value_counts()
        for v in vars
    }
    df = pd.concat(dfs, axis=1).fillna(0).astype(int)

    if markdown: return df.to_markdown()
    return df

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

def split_model_results(X, y, w, b, threshold):
    X_np = X.to_numpy()
    y_pred = logisticRegression.predict(X_np, w, b, threshold)

    tp_indices = np.where((y == 1) & (y_pred == 1))[0]
    fp_indices = np.where((y == 0) & (y_pred == 1))[0]
    tn_indices = np.where((y == 0) & (y_pred == 0))[0]
    fn_indices = np.where((y == 1) & (y_pred == 0))[0]

    def df_og_index(idx):
        df = pd.DataFrame(X_np[idx], columns=X.columns)
        df.index = X.index[idx]
        return df

    return df_og_index(tp_indices), df_og_index(fp_indices), df_og_index(tn_indices), df_og_index(fn_indices)

def get_df_model_analysis(X, y, w, b, threshold, raw=True, quantile=None):
    X_tp_df, X_fp_df, X_tn_df, X_fn_df = split_model_results(X, y, w, b, threshold)
    
    if raw:
        tp = X_tp_df.copy(); tp["Group"] = "TP"
        fp = X_fp_df.copy(); fp["Group"] = "FP"
        tn = X_tn_df.copy(); tn["Group"] = "TN"
        fn = X_fn_df.copy(); fn["Group"] = "FN"

        df_profiles = pd.concat([tn, fp, fn, tp], axis=0)
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

    df_profiles = pd.DataFrame({
        'TN': profile_tn,
        'FP': profile_fp,
        'FN': profile_fn,
        'TP': profile_tp
    })
    df_profiles.index.name = 'Variable'
    return df_profiles

def get_df_profiles(feature_tracker, model=None):
    if model is None: 
        feature_tracker.flush_to_df(returnDf=False, removeTargets=True)
        model = feature_tracker.get_trained_model(print_stats=False)
    threshold = model.threshold

    X_train, y_train, *_ = feature_tracker.return_split_train_eval()
    #X_train_unscaled, *_ = feature_tracker.return_split_train_eval(to_scale=False)
    df_profiles = get_df_model_analysis(X_train, y_train, model.w, model.b, threshold=threshold, raw=True)
    return df_profiles


def get_df_conf_matrix_split(feature_tracker, model=None):
    df_profiles = get_df_profiles(feature_tracker, model)

    df_tn = df_profiles[df_profiles["Group"] == "TN"].copy()
    df_fp = df_profiles[df_profiles["Group"] == "FP"].copy()
    df_fn = df_profiles[df_profiles["Group"] == "FN"].copy()
    df_tp = df_profiles[df_profiles["Group"] == "TP"].copy()

    return df_tn, df_fp, df_fn, df_tp

def get_df_conf_matrix_count_by_var(var, feature_tracker, model=None, TP_FN=True):
    df_tn, df_fp, df_fn, df_tp = get_df_conf_matrix_split(feature_tracker, model)

    if TP_FN: 
        var1 = 'TP'
        tab1 = df_tp[var].value_counts().rename('TP count')
        var2 = 'FN'
        tab2 = df_fn[var].value_counts().rename('FN count')

    tab = pd.concat([tab1, tab2], axis=1).astype(int).rename_axis(var).reset_index()

    total_row = pd.DataFrame([{
        var: 'Total',
        f'{var1} count': tab[f'{var1} count'].sum(),
        f'{var2} count': tab[f'{var2} count'].sum()
    }])
    
    tab = pd.concat([tab, total_row], ignore_index=True)
    #print(tab.to_markdown(index=False))
    return tab

def get_df_conf_matrix_contrib(feature_tracker, model=None):
    X = feature_tracker.flush_to_df(removeTargets=True)

    if model is None: model = feature_tracker.get_trained_model(print_stats=False)

    df_profiles = get_df_profiles(feature_tracker, model)
    contrib = df_profiles[X.columns].mul(model.w, axis=1)
    
    # if TP_FN: diff = contrib[df_profiles['Group'] == 'TP'].mean() - contrib[df_profiles['Group'] == 'FN'].mean()

    # diff.sort_values(key=abs, ascending=False).head()
    
    return contrib

def get_df_conf_matrix_contrib_analysis(var_filter, var_filter_value, var_name, feature_tracker, model=None, TP_FN=True):
    df_tn, df_fp, df_fn, df_tp = get_df_conf_matrix_split(feature_tracker, model)
    contrib = get_df_conf_matrix_contrib(feature_tracker, model=model)
    df_filtered_1 = df_tp[df_tp[var_filter] == var_filter_value].copy()
    df_filtered_2 = df_fn[df_fn[var_filter] == var_filter_value].copy()
    if TP_FN:
        tab = pd.DataFrame({
            'Group': [f'TP ({var_filter} = {var_filter_value})', f'FN ({var_filter} = {var_filter_value})'],
            f'Contribution moyenne ({var_name})': [
                contrib.loc[df_filtered_1.index, var_name].mean(),
                contrib.loc[df_filtered_2.index, var_name].mean()
            ],
            f'Mode ({var_name})': [
                df_filtered_1[var_name].mode().tolist(),
                df_filtered_2[var_name].mode().tolist()
            ]})
    return tab

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

    if markdown: return df.sort_index().to_markdown()
    return df

from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


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
    for item in items:
        a,b = item.split("x")
        pairs.append((int(a), int(b)))

    for a,b in pairs:
        col_name = f"{var1}_{a}_x_{var2}_{b}"
        if isVar1OneHot:
            df[col_name] = (df[f'{var1}_{a}'] & (df[var2] == b)).astype(int) 
        else: 
            df[col_name] = ((df[var1] == a) & (df[var2] == b)).astype(int)
    return df

def find_correlated_cols(df, threshold=0.95, toPlot=True):
    numeric_df = df.select_dtypes(include=np.number)
    corr_matrix = numeric_df.corr().abs()

    if toPlot:
        plt.figure(figsize=(12,10))
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
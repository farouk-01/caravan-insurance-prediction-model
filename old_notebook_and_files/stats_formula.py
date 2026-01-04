import pandas as pd

def esperence_filter(counts, crossTab, min_count=5):
    row_totals = counts.sum(axis=1)       
    n_cols = counts.shape[1]                  
    row_thresholds = row_totals / n_cols

    for row in counts.index:
        mask = (counts.loc[row] >= row_thresholds[row]) & (counts.loc[row] >= min_count)
        crossTab.loc[row] = crossTab.loc[row].where(mask, 0)
    return crossTab

def variation_crossTab(ct, df, var1, threshold=0.05):
    marginal = df.groupby(var1)['CARAVAN'].mean()
    diff = ct.sub(marginal, axis=0).where(ct != 0, 0)
    diff = diff.where(diff >= threshold, 0)
    return diff.round(4)
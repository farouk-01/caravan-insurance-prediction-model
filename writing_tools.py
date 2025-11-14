import pandas as pd
import numpy as np
import stats_formula

def crossTab_with_caravan_to_markdown(df, var1, isProb = False):
    ct = pd.crosstab(df['CARAVAN'], df[var1])
    prob = ct.div(ct.sum(axis=0), axis=1)

    counts_row = ['CARAVAN = 1'] + ct.loc[1].tolist()
    prob_row = [f'P(CARAVAN = 1\\|{var1})'] + [f"{v:.4f}" for v in prob.loc[1].tolist()]

    combined = pd.DataFrame(
        [counts_row, prob_row],
        columns=[var1] + ct.columns.tolist()
    )
    print(combined.to_markdown(index=False))

def crossTab_norm(var1, var2, df, min_count = 5):
        col1 = df[var1]
        col2 = df[var2]
        counts1 = pd.crosstab(col1, col2, df['CARAVAN'] == 1, aggfunc='sum')
        counts2 = pd.crosstab(col2, col1, df['CARAVAN'] == 1, aggfunc='sum')
        ct1 = pd.crosstab(col1, col2,  df['CARAVAN'], aggfunc='mean').fillna(0)
        ct2 = pd.crosstab(col2, col1,  df['CARAVAN'], aggfunc='mean').fillna(0)

        ct1 = stats_formula.esperence_filter(counts1, ct1)
        ct2 = stats_formula.esperence_filter(counts2, ct2)
        diff1 = stats_formula.variation_crossTab(ct1, df, var1)
        diff2 = stats_formula.variation_crossTab(ct2, df, var2)
        diff1_aligned = diff1.reindex_like(diff2).fillna(0)
        diff_combined = np.maximum(diff1_aligned, diff2)
        crossTab_norm = pd.DataFrame(diff_combined, index=diff1_aligned.index, columns=diff1_aligned.columns)
        #crossTab_norm = stats_formula.variation_crossTab(crossTab_norm, df, var1)
        
        #crossTab_norm = crossTab_norm.where(counts >= min_count, 0)

        crossTab_norm = crossTab_norm.loc[~(crossTab_norm==0).all(axis=1)]
        crossTab_norm = crossTab_norm.loc[:, ~(crossTab_norm==0).all(axis=0)]


        crossTab_norm = crossTab_norm.reset_index()
        crossTab_norm = crossTab_norm.rename(columns={var2: f"{var1}/{var2}"})
        return crossTab_norm.round(4)

def crossTab_norm_to_markdown(var1, var2, df):
    ct = crossTab_norm(var1, var2, df)
    print(ct.to_markdown(index=False))





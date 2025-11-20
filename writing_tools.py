import pandas as pd
import numpy as np
import stats_formula

def crossTab_with_caravan_to_markdown(df, var1, isFreq=False):
    ct = pd.crosstab(df[var1], df['CARAVAN'])
    prob = ct.div(ct.sum(axis=0), axis=1)

    if isFreq:
        freq_tbl = ct.copy()
        freq_tbl.insert(0, var1, freq_tbl.index)
        
        row_totals = freq_tbl[[1]].sum(axis=1)
        grand_total = row_totals.sum()

        freq_tbl[f"P({var1})"] = row_totals / grand_total

        freq_tbl.drop(columns=0, inplace=True)
        freq_tbl.sort_values(by=1, inplace=True, ascending=False)

        freq_tbl[f'cumul P({var1})'] = freq_tbl[f"P({var1})"].cumsum()

        freq_tbl.rename(columns={1: 'CARAVAN = 1'}, inplace=True)
        print(freq_tbl.to_markdown(index=False))
    else:

        counts_row = ['CARAVAN = 1'] + ct.loc[1].tolist()
        prob_row = [f'P(CARAVAN = 1\\|{var1})'] + [f"{v:.4f}" for v in prob.loc[1].tolist()]

        combined = pd.DataFrame(
            [counts_row, prob_row],
            columns=[var1] + ct.columns.tolist()
        )
        print(combined.to_markdown(index=False))

def crossTab_with_var2_to_markdown(df, var1, var2, isProb = False):
    ct = pd.crosstab(df[var1], df[var2])

    if isProb:
        ct = ct.div(ct.sum(axis=0), axis=1)
        formatted = ct.copy()
        for col in formatted.columns:
            formatted[col] = formatted[col].map(lambda v: f"{v:.4f}")
    else:
        formatted = ct.copy()

    formatted.index.name = f"{var1}/{var2}"

    print(formatted.to_markdown())

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
        # diff1 = ct1
        # diff2 = ct2

        diff2 = diff2.T

        common_index = diff1.index.intersection(diff2.index) 
        common_columns = diff1.columns.union(diff2.columns) 
        
        diff1_aligned = diff1.reindex(index=common_index, columns=common_columns, fill_value=0)
        diff2_aligned = diff2.reindex(index=common_index, columns=common_columns, fill_value=0)

        diff_combined = np.maximum(diff1_aligned, diff2_aligned)

        crossTab_norm = pd.DataFrame(diff_combined, index=diff1_aligned.index, columns=diff1_aligned.columns) 

        crossTab_norm = crossTab_norm.loc[~(crossTab_norm==0).all(axis=1)]
        crossTab_norm = crossTab_norm.loc[:, ~(crossTab_norm==0).all(axis=0)]


        crossTab_norm = crossTab_norm.reset_index()
        crossTab_norm = crossTab_norm.rename(columns={var1: f"{var1}/{var2}"})
        return crossTab_norm.round(4)

def crossTab_norm_to_markdown(var1, var2, df):
    ct = crossTab_norm(var1, var2, df)
    print(ct.to_markdown(index=False))





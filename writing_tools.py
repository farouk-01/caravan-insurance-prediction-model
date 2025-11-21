import pandas as pd
import numpy as np
import stats_formula

def crossTab_with_caravan_to_markdown(df, var1, wFreq=False, wFreqCumul=False, wCond=False, transpose=False):
    ct = pd.crosstab(df[var1], df['CARAVAN'])
    freq_tbl = ct.copy()
    freq_tbl.insert(0, var1, freq_tbl.index)
    freq_tbl.sort_values(by=1, inplace=True, ascending=False)

    if wFreq:
        row_totals = freq_tbl[[1]].sum(axis=1)
        grand_total = row_totals.sum()

        freq_tbl[f"P({var1})"] = row_totals / grand_total
        if wFreqCumul:
            freq_tbl[f'cumul P({var1})'] = freq_tbl[f"P({var1})"].cumsum()
    if wCond:
         freq_tbl[f'P(CARAVAN = 1 \\| {var1})'] = freq_tbl[1] / (freq_tbl[1] + freq_tbl[0])

    freq_tbl.drop(columns=0, inplace=True)
    freq_tbl.rename(columns={1: 'CARAVAN = 1'}, inplace=True)
    print(f'**Tableau de frequence de {var1} si CARAVAN=1**')
    if transpose:
         freq_tbl.set_index(var1, inplace=True)
         freq_tbl.columns.name = var1
         print(freq_tbl.T.to_markdown())
    else:
         print(freq_tbl.to_markdown(index=False))
    

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

def crossTab_norm(var1, var2, df, min_count=5, hasDict=False, dict1=None, dict2=None):
        col1 = df[var1]
        col2 = df[var2]
        counts1 = pd.crosstab(col1, col2, df['CARAVAN'] == 1, aggfunc='sum')
        counts2 = pd.crosstab(col2, col1, df['CARAVAN'] == 1, aggfunc='sum')
        ct1 = pd.crosstab(col1, col2,  df['CARAVAN'], aggfunc='mean').fillna(0)
        ct2 = pd.crosstab(col2, col1,  df['CARAVAN'], aggfunc='mean').fillna(0)

        ct1 = stats_formula.esperence_filter(counts1, ct1, min_count=min_count)
        ct2 = stats_formula.esperence_filter(counts2, ct2, min_count=min_count)
        diff1 = stats_formula.variation_crossTab(ct1, df, var1)
        diff2 = stats_formula.variation_crossTab(ct2, df, var2)

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

def crossTab_norm_to_markdown(var1, var2, df, min_count=5, hasDict=False, dict1 = None, dict2 = None):
    ct = crossTab_norm(var1, var2, df, min_count=min_count)
    interactions = []
    for i, row in ct.iterrows():
        for col in ct.columns[1:]:
            if row[col] > 0:
                interactions.append(f"{int(row[ct.columns[0]])}x{int(col)}")
    if hasDict:
        if dict1 is not None:
            ct[ct.columns[0]] = ct[ct.columns[0]].map(lambda x: dict1.get(x, x))
    print(ct.to_markdown(index=False))
    print(f"\n significant interactions: ({var1}x{var2})", ", ".join(interactions))
    print()





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

        #Bug: no matter l'ordre de var1 et var2, common_index et common_columns 
        #garde les mm valeurs, donc le tableau va tjrs afficher le mm ordre pour row/col
        #en attendant juste verifie que c'est le bon ordre
        #Index sera tjrs lui avec le plus petit nombre d'index
        common_index = ct1.index.intersection(ct2.index) 
        common_columns = ct1.columns.union(ct2.columns) 
        
        diff1_aligned = diff1.reindex(index=common_index, columns=common_columns, fill_value=0)
        diff2_aligned = diff2.reindex(index=common_index, columns=common_columns, fill_value=0)

        #Le bug en haut + sa peut confondre, par exemple si X = 0 et Z = 8
        #On pourrait obtenir 0x8 = 0 (X x Z), mais 8x0 (Z x X)= 0.043 
        #Je crois c'est correct parce que toute facon un interaction terms est seulement une multiplication
        #Donc X x Z = Z x X. La raison pourquoi j'ai des valeurs différentes est du au méthode
        #esperence_filter() qui filtre selon l'esperance attendu selon la ligne donc l'esperence peut changer 
        #de variable a une autre. Pareil pour variation_crossTab
        #Ensuite quand je fais np.maximum, je prend le plus élevé des deux termes 
        #pour ne pas manquer de terme significatif
        #C'est pas urgent à regler
        diff_combined = np.maximum(diff1_aligned, diff2_aligned)

        crossTab_norm = pd.DataFrame(diff_combined, index=diff1_aligned.index, columns=diff1_aligned.columns) 

        crossTab_norm = crossTab_norm.loc[~(crossTab_norm==0).all(axis=1)]
        crossTab_norm = crossTab_norm.loc[:, ~(crossTab_norm==0).all(axis=0)]


        crossTab_norm = crossTab_norm.reset_index()
        crossTab_norm = crossTab_norm.rename(columns={'index': f"{var1}/{var2}"})
        return crossTab_norm.round(4)

def crossTab_norm_to_markdown(var1, var2, df):
    ct = crossTab_norm(var1, var2, df)
    print(ct.to_markdown(index=False))





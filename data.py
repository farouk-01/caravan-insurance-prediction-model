import pandas as pd 
import re

def get_data():
    df = pd.read_table('insurance_data/ticdata2000.txt')

    col_names = [
        "MOSTYPE","MAANTHUI","MGEMOMV","MGEMLEEF","MOSHOOFD","MGODRK","MGODPR",
        "MGODOV","MGODGE","MRELGE","MRELSA","MRELOV","MFALLEEN","MFGEKIND",
        "MFWEKIND","MOPLHOOG","MOPLMIDD","MOPLLAAG","MBERHOOG","MBERZELF",
        "MBERBOER","MBERMIDD","MBERARBG","MBERARBO","MSKA","MSKB1","MSKB2",
        "MSKC","MSKD","MHHUUR","MHKOOP","MAUT1","MAUT2","MAUT0","MZFONDS",
        "MZPART","MINKM30","MINK3045","MINK4575","MINK7512","MINK123M",
        "MINKGEM","MKOOPKLA","PWAPART","PWABEDR","PWALAND","PPERSAUT","PBESAUT",
        "PMOTSCO","PVRAAUT","PAANHANG","PTRACTOR","PWERKT","PBROM","PLEVEN",
        "PPERSONG","PGEZONG","PWAOREG","PBRAND","PZEILPL","PPLEZIER","PFIETS",
        "PINBOED","PBYSTAND","AWAPART","AWABEDR","AWALAND","APERSAUT","ABESAUT",
        "AMOTSCO","AVRAAUT","AAANHANG","ATRACTOR","AWERKT","ABROM","ALEVEN",
        "APERSONG","AGEZONG","AWAOREG","ABRAND","AZEILPL","APLEZIER","AFIETS",
        "AINBOED","ABYSTAND", "CARAVAN"
    ]

    df.columns = col_names


    with open("insurance_data/dictionary.txt", "r") as f:
        text = f.read()

    var_discrete = {}
    col_descriptions = {}
    lines = text.splitlines()[3:89]
    for line in lines:
        words = line.split()
        var_name = words[1]
        description = " ".join(words[2:])
        if re.search(r'see L[0-4]', description): #alors var discrète
            var_discrete[var_name] = True
        else:
            var_discrete[var_name] = False
        col_descriptions[var_name] = description
    df.attrs['description'] = col_descriptions
    df.attrs['var_type'] = var_discrete

    age_map = {1: 25, 2: 35, 3: 45, 4: 55, 5: 65, 6: 75}
    df['MGEMLEEF'] = df['MGEMLEEF'].map(age_map)

    return df

def get_split_data(df):
    X = df.drop('CARAVAN', axis=1)
    y = df['CARAVAN']

    return X, y

def describe(col, df):
    return df.attrs['description'].get(col)

def top_index_and_values(top_n, df):
    top_var = df[df < 1].head(top_n)
    top_index = top_var.index
    top_values = top_var.values
    for (a, b), val in zip(top_index, top_values):
        print(f'{describe(a, df):<{50}} {a:<10} - {val:>5.4f}' )

def get_var_by_types(df):
    var_type = df.attrs['var_type']
    discrete_vars = [col for col, is_discrete in var_type.items() if is_discrete]
    continuous_vars = [col for col, is_discrete in var_type.items() if not is_discrete]
    return discrete_vars, continuous_vars

df = get_data()
print(df['MOSTYPE'].value_counts())
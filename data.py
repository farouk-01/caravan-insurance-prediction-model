import pandas as pd 
import numpy as np
import re
import glob 
import logisticRegression

def read_dictionnary(df):
    with open("insurance_data/dictionary.txt", "r") as f:
        text = f.read()

    var_ordinale = {}
    col_descriptions = {}
    lines = text.splitlines()[3:89]
    for line in lines:
        words = line.split()
        var_name = words[1]
        description = " ".join(words[2:])
        if  re.search(r'see L[134]', description): #ils sont ordinales
            var_ordinale[var_name] = True
        else:
            var_ordinale[var_name] = False
        col_descriptions[var_name] = description
    df.attrs['description'] = col_descriptions
    df.attrs['ordinale'] = var_ordinale

    return df

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


    df = read_dictionnary(df)
    return df

def get_test_data():
    df = pd.read_table('insurance_data/ticeval2000.txt')

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
        "AINBOED","ABYSTAND"
    ]

    df.columns = col_names

    df = read_dictionnary(df)
    return df

def add_interactions_terms(X, interactions):
    X_new = X.copy()
    for var1, var2 in interactions:
        inter_name = f"{var1}_x_{var2}"
        X_new[inter_name] = X_new[var1] * X_new[var2]
    return X_new

def get_test_data_with_terms():
    interactions_to_add_1 = [
        ("PPERSAUT", "APLEZIER"),
        ("PPERSAUT", "PPLEZIER"),
        ("PPERSAUT", "PBRAND"),
        ("APERSAUT", "PBRAND"),
        ("PPLEZIER", "MINKGEM"),
        ("PPLEZIER", "MKOOPKLA"),
        ("PPLEZIER", "MHKOOP")
    ]
    df = get_test_data()
    df = add_interactions_terms(df, interactions_to_add_1 )
    return df

def get_test_targets():
    df = pd.read_table('insurance_data/tictgts2000.txt')
    df.columns = ['CARAVAN']
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
    var_type = df.attrs['ordinale']
    ordinale_vars = [col for col, is_ordinale in var_type.items() if is_ordinale]
    discrete_vars = [col for col, is_ordinale in var_type.items() if not is_ordinale]
    return ordinale_vars, discrete_vars

def save_model_state(w,b,threshold,accuracy,conf_matrix, version, changes, why):
    model_state = {
        'w': w,
        'b': b,
        'threshold': threshold,
        'accuracy': accuracy, 
        'conf_matrix': conf_matrix,
        'version': version,
        'changes': changes,
        'why': why
    }
    np.save(f"model_version/model_state_dict_{version}.npy", model_state)

def import_models():
    models = {}
    for path in glob.glob('model_version/model_state_dict_*.npy'):
        models[path] = np.load(path, allow_pickle=True).item()
    return models

def compare_models():
    models = import_models()
    for name, m in models.items():
        print(f'version: {m['version']}')
        print(f'{m['changes']} car {m['why']}')
        print(m['accuracy'])
        print(m['conf_matrix'])
        print()

def edit_model_state(model_version, changes, why):
    state = np.load(f'model_version/model_state_dict_{model_version}.npy', allow_pickle=True).item()
    state['version'] = model_version
    state['changes'] = changes
    state['why'] = why
    np.save(f'model_version/model_state_dict_{model_version}.npy', state)

    

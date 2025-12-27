import pandas as pd 
import numpy as np
import re
import glob 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

class DataInfo:
    _instance = None
    _exist = False

    def __new__(cls, df):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, df):
        if self.__class__._exist:
            return
        self.df = df
        self.desc_dict = df.attrs['description']
        self.scaler = StandardScaler()
        self.__class__._exist = True

    @classmethod
    def get_instance(cls):
        if cls._instance is None or not cls._exist: raise Exception("Appelle DataInfo(df) avant")
        return cls._instance

    def fit_scaler(self, X, cols_to_scale):
        self.scaler.fit(X[cols_to_scale])
        return self

    def transform(self, X, cols_to_scale):
        X_scaled = X.copy()
        X_scaled[cols_to_scale] = self.scaler.transform(X_scaled[cols_to_scale])
        return X_scaled
    
    def fit_transform(self, X, cols_to_scale):
        self.fit_scaler(X, cols_to_scale)
        return self.transform(X, cols_to_scale)

    def get_desc_dict(self):
        return self.desc_dict
    
    def get_dict_of(self, vars=None):
        desc_dict = self.get_desc_dict()
        
        if vars is None: return desc_dict

        return {v: desc_dict[v] for v in vars if v in desc_dict}
    
    def _format_desc(self, desc):
        if desc is None: return None
        desc = re.sub(r"\s+", "_", desc.strip())
        return desc
    
    def _get_new_label(self, var, dict_of_var):
        desc = self._format_desc(dict_of_var.get(var))
        if desc is None: return var
        return f"{var}_{desc}" if desc else var

    def replace_by_name_desc(self, obj, vars=None):
        desc_dict = self.get_desc_dict() if vars is None else self.get_dict_of(vars)
        
        if isinstance(obj, pd.Series):
            if obj.index.name is not None: obj.rename_axis(self._get_new_label(obj.index.name, desc_dict), inplace=True)
            return obj.rename(index={i: self._get_new_label(i, desc_dict) for i in obj.index})
            
        elif isinstance(obj, pd.DataFrame):
            df = obj.copy()
            df = df.rename(columns={c: self._get_new_label(c, desc_dict) for c in df.columns})
            df = df.rename(index={i: self._get_new_label(i, desc_dict) for i in df.index})
            return df
        
        raise TypeError("obj doit etre Series ou DataFrame")

        


def read_dictionnary(df):
    with open("insurance_data/dictionary.txt", "r") as f:
        text = f.read()

    var_categorical = {}
    col_descriptions = {}
    lines = text.splitlines()[3:89]
    for line in lines:
        words = line.split()
        var_name = words[1]
        description = " ".join(words[2:])
        if  re.search(r'see L[02]', description): #ils sont ordinales
            var_categorical[var_name] = True
        else:
            var_categorical[var_name] = False
        col_descriptions[var_name] = description
    df.attrs['description'] = col_descriptions
    df.attrs['categorical'] = var_categorical

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
    data_info = DataInfo(df)
    return df

def get_split_train_eval_data(df, toNpy=False):
    X_train_full, y_train_full = get_split_data(df)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, random_state=42, stratify=y_train_full
    )
    if toNpy:
        X_train = X_train.to_numpy()
        X_val = X_val.to_numpy()
        y_train = y_train.to_numpy()
        y_val = y_val.to_numpy()
    return X_train, X_val, y_train, y_val

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
    X = df.copy()
    y = X.pop('CARAVAN')

    return X, y

def describe(col, df):
    return df.attrs['description'].get(col)

def top_index_and_values(top_n, df):
    top_var = df[df < 1].head(top_n)
    top_index = top_var.index
    top_values = top_var.values
    for (a, b), val in zip(top_index, top_values):
        print(f'{describe(a, df):<{50}} {a:<10} - {val:>5.4f}' )

def get_var_by_types(df): #todo change is_ordinale
    var_type = df.attrs['categorical']
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

def get_dict(level):
    # L0
    L0 = {
        1: "High Income, expensive child",
        2: "Very Important Provincials",
        3: "High status seniors",
        4: "Affluent senior apartments",
        5: "Mixed seniors",
        6: "Career and childcare",
        7: "Dinki's (double income no kids)",
        8: "Middle class families",
        9: "Modern, complete families",
        10: "Stable family",
        11: "Family starters",
        12: "Affluent young families",
        13: "Young all american family",
        14: "Junior cosmopolitan",
        15: "Senior cosmopolitans",
        16: "Students in apartments",
        17: "Fresh masters in the city",
        18: "Single youth",
        19: "Suburban youth",
        20: "Etnically diverse",
        21: "Young urban have-nots",
        22: "Mixed apartment dwellers",
        23: "Young and rising",
        24: "Young, low educated",
        25: "Young seniors in the city",
        26: "Own home elderly",
        27: "Seniors in apartments",
        28: "Residential elderly",
        29: "Porchless seniors: no front yard",
        30: "Religious elderly singles",
        31: "Low income catholics",
        32: "Mixed seniors",
        33: "Lower class large families",
        34: "Large family, employed child",
        35: "Village families",
        36: "Couples with teens 'Married with children'",
        37: "Mixed small town dwellers",
        38: "Traditional families",
        39: "Large religous families",
        40: "Large family farms",
        41: "Mixed rurals"
    }

    # L1
    L1 = {
        1: "20-30 years",
        2: "30-40 years",
        3: "40-50 years",
        4: "50-60 years",
        5: "60-70 years",
        6: "70-80 years"
    }

    # L2
    L2 = {
        1: "Successful hedonists",
        2: "Driven Growers",
        3: "Average Family",
        4: "Career Loners",
        5: "Living well",
        6: "Cruising Seniors",
        7: "Retired and Religeous",
        8: "Family with grown ups",
        9: "Conservative families",
        10: "Farmers"
    }

    # L3
    L3 = {
        0: "0%",
        1: "1 - 10%",
        2: "11 - 23%",
        3: "24 - 36%",
        4: "37 - 49%",
        5: "50 - 62%",
        6: "63 - 75%",
        7: "76 - 88%",
        8: "89 - 99%",
        9: "100%"
    }

    # L4
    L4 = {
        0: "f 0",
        1: "f 1 - 49",
        2: "f 50 - 99",
        3: "f 100 - 199",
        4: "f 200 - 499",
        5: "f 500 - 999",
        6: "f 1000 - 4999",
        7: "f 5000 - 9999",
        8: "f 10.000 - 19.999",
        9: "f 20.000 - ?"
    }

    level_dicts = {
        "L0": L0,
        "L1": L1,
        "L2": L2,
        "L3": L3,
        "L4": L4
    }

    return level_dicts.get(level.upper(), None)

def apply_Lx_to_index(idx, name, level):
    lx = get_dict(level)

    # 1. Match the pattern name_digit (e.g. MOSTYPE_8)
    match = re.search(rf"{name}_(\d+)", idx)
    if not match:
        return idx  # Nothing to replace
    
    code = int(match.group(1))
    label = lx.get(code, f"{name}_{code}")  # Fallback if key not found

    # 2. Replace ONLY the matched part, leave the rest untouched
    return idx[:match.start()] + label + idx[match.end():]

def prepare_catpca_df(df, cols, cat_cols):
    df_for_catpca = df.copy()
    dropped = [c for c in cols if df_for_catpca[c].nunique() <= 1]
    if dropped: 
        df_for_catpca.drop(columns=dropped, inplace=True)
        cat_cols = [c for c in cat_cols if c in df_for_catpca.columns]

    df_for_catpca[cat_cols] = df_for_catpca[cat_cols].astype('Int64')
    
    for c in cat_cols:
        if df_for_catpca[c].min() == 0:
            df_for_catpca[c] = df_for_catpca[c] + 1
    return df_for_catpca.copy()
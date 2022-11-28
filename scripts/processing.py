


def drop_columns(X,list_col):
    print("Ces colonnes vont être retirées:",list_col)
    X.drop(columns=list_col,inplace=True)
    return X


def get_dummies_var(X):
    list_dummies=[]
    for col in X.columns:
        if X[col].nunique()<=2:
            list_dummies.append(col)
    return list_dummies


def convert_modalities_to_quali(X, modalities_var):
    for i in modalities_var:
        if i in (X.select_dtypes(include=['int64','float64'])).columns.tolist():
            X[i] = X[i].astype(str) 
    return X


def comparer_na_variables_doublons(X, dict_doublons):

    colonnes_to_drop=[]
    print("COMPARAISON NA POUR COLONNES EN DOUBLE")

    for key,value in dict_doublons.items():
        print(key," VS ",value)

        print(X[key].isna().sum()," VS ", X[value].isna().sum())
        if X[key].isna().sum()== X[value].isna().sum():
            colonnes_to_drop.append(key)

        elif X[key].isna().sum()>= X[value].isna().sum():
            colonnes_to_drop.append(key)

        else:
            colonnes_to_drop.append(value)

    print("DONE")

    return colonnes_to_drop


# pour supp les valeurs manquantes importantes ( sauf ou y'a SUM )

def na_sup_20_fill_by_0_vs_autres(missing_rate_sup_20):

    variables_a_fill_0=[]
    variables_restantes=[]

    for var in missing_rate_sup_20: 
        if 'SUM' in var : 
            variables_a_fill_0.append(var)
        else: 
            variables_restantes.append(var)

    return variables_a_fill_0,variables_restantes



def fill_by_0(X, list_to_fill):
    print("IMPUTATION PAR 0")
    for col in list_to_fill:
        X[col]=X[col].fillna(0)
    print("OK")
    return X



def imputation_for_na(X):
    print("IMPUTATION PAR MODE OU MEDIANE")
    for col in X.columns:
        if X[col].dtype=="object":
            X[col]= X[col].fillna(X[col].value_counts().idxmax())
        else:
            X[col]=X[col].fillna(X[col].median())
    print("OK")
    return X



#On établi les 3 dictionnaires pour les variables socio-démographique


def proxys_processing(df,richesse,chomage,inondation):
    richesse.set_index("Unnamed: 0",inplace=True)
    chomage.set_index("Code",inplace=True)
    inondation.set_index("Commune",inplace=True)
    dictionary_chomage=chomage.to_dict()['MOYENNE']
    dictionary_hlm=richesse.to_dict()['Taux de logements sociaux']
    dictionary_prix_loyer=richesse.to_dict()['Loyer moyen par mètre carré de surface habitable (en €)']
    dictionary_innondation=inondation.to_dict()['Somme de nb_com_ddrm']
    departement_proxy=chomage.index.astype(str).tolist()
    codes_postaux_proxy=inondation.index.astype(str).tolist()
    df['DEPARTEMENT_CRI']=df['DEPARTEMENT_CRI'].astype(str).apply(lambda x: replace_if_not_in(x,departement_proxy,df["DEPARTEMENT_CRI"]))
    df["TAUX_HLM"]=df["DEPARTEMENT_CRI"].astype(str).map(dictionary_hlm)
    df["TAUX_CHOMAGE"]=df["DEPARTEMENT_CRI"].astype(str).map(dictionary_chomage)
    df["PRIX_LOYER"]=df["DEPARTEMENT_CRI"].map(dictionary_prix_loyer)
    df["TAUX_HLM"]=df["TAUX_HLM"].fillna(df["TAUX_HLM"].median())
    df["TAUX_CHOMAGE"]=df["TAUX_CHOMAGE"].fillna(df["TAUX_CHOMAGE"].median())
    df["PRIX_LOYER"]=df["PRIX_LOYER"].fillna(df["PRIX_LOYER"].median())


def replace_if_not_in(x,list_geo,col):
    if x not in list_geo:
        return col.value_counts().idxmax()
    else: 
        return x
def proxys_processing(df,richesse,chomage,inondation):
    richesse.set_index("Unnamed: 0",inplace=True)
    chomage.set_index("Code",inplace=True)
    inondation.set_index("Commune",inplace=True)
    dictionary_chomage=chomage.to_dict()['MOYENNE']
    dictionary_hlm=richesse.to_dict()['Taux de logements sociaux']
    dictionary_prix_loyer=richesse.to_dict()['Loyer moyen par mètre carré de surface habitable (en €)']
    dictionary_innondation=inondation.to_dict()['Somme de nb_com_ddrm']
    departement_proxy=chomage.index.astype(str).tolist()
    codes_postaux_proxy=inondation.index.astype(str).tolist()
    df['DEPARTEMENT_CRI']=df['DEPARTEMENT_CRI'].astype(str).apply(lambda x: replace_if_not_in(x,departement_proxy,df["DEPARTEMENT_CRI"]))
    df["TAUX_HLM"]=df["DEPARTEMENT_CRI"].astype(str).map(dictionary_hlm)
    df["TAUX_CHOMAGE"]=df["DEPARTEMENT_CRI"].astype(str).map(dictionary_chomage)
    df["PRIX_LOYER"]=df["DEPARTEMENT_CRI"].map(dictionary_prix_loyer)
    df["TAUX_HLM"]=df["TAUX_HLM"].fillna(df["TAUX_HLM"].median())
    df["TAUX_CHOMAGE"]=df["TAUX_CHOMAGE"].fillna(df["TAUX_CHOMAGE"].median())
    df["PRIX_LOYER"]=df["PRIX_LOYER"].fillna(df["PRIX_LOYER"].median())

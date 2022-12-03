import pandas as pd

"""   FONCTIONS GENERALES   """


def drop_columns(X:pd.DataFrame,columns: list) -> pd.DataFrame:

    """ retourne le dataframe avec colonnes supprimées
    Arguments: 
        - X_train: X_train comportant les variables quantitatives 
        - columns: liste des colonnes à dropper

    """  
    print("Ces colonnes vont être retirées:",columns)
    X.drop(columns=columns,inplace=True)
    return X



"""   TRAITEMENT DES TYPES   """


def get_dummies_var(X:pd.DataFrame) -> list:

    """ retourne la liste des variables binaires (or target)
    Argument: 
        - X

    """
    list_dummies=[]
    for col in X.columns:
        if X[col].nunique()<=2:
            list_dummies.append(col)
    return list_dummies



def convert_modalities_to_quali(X: pd.DataFrame, modalities_var: list) -> pd.DataFrame:

    """ retourne le dataframe avec variables catégorielles converties au bon type
    Arguments: 
        - X
        - modalities_var: variables catégorielles qui ne sont pas converties au bon type ==> numériques au lieu de str 

    """
    for i in modalities_var:
        if i in (X.select_dtypes(include=['int64','float64'])).columns.tolist():
            X[i] = X[i].astype(str) 
    return X



"""   TRAITEMENT DES VALEURS MANQUANTES   """


def compare_na_variables_duplicates(X:pd.DataFrame, dict_doublons:dict) -> list:

    """ retourne la liste des colonnes en doublons à supprimer ()
    Arguments: 
        - X
        - modalities_var: variables catégorielles qui ne sont pas converties au bon type ==> numériques au lieu de str 

    """
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



def na_sup_20_fill_by_0_vs_autres(missing_rate_sup_20: list) :

    """ retourne la liste des colonnes à fill par 0 et la liste des variables à drop (restantes)
    Argument: 
        - missing_rate_sup_20: liste des variables ayant un taux de na supérieurs à 20%

    """
    variables_a_fill_0=[]
    variables_restantes=[]

    for var in missing_rate_sup_20: 
        if ('SUM' in var) or ('COUT' in var): 
            variables_a_fill_0.append(var)
        else: 
            variables_restantes.append(var)

    return variables_a_fill_0,variables_restantes
    


def fill_by_0(X:pd.DataFrame, fill_0:list)-> pd.DataFrame:

    """ retourne le dataframe avec colonnes à imputer valeurs manquantes par 0 
    Arguments: 
        - X
        - fill_0: list des variables à imputer par 0

    """
    print("IMPUTATION PAR 0")
    for col in fill_0:
        X[col]=X[col].fillna(0)
    print("OK")
    return X



def imputation_for_na(X:pd.DataFrame)-> pd.DataFrame:

    """ retourne le dataframe avec le reste des variables imputées (par médiane ou mode)
    Argument: 
        - X

    """
    print("IMPUTATION PAR MODE OU MEDIANE")
    for col in X.columns:
        if X[col].dtype=="object":
            X[col]= X[col].fillna(X[col].value_counts().idxmax())
        else:
            X[col]=X[col].fillna(X[col].median())
    print("OK")
    return X



"""   TRAITEMENT DES VARIABLES EXTERNES   """


def proxys_processing(X:pd.DataFrame,dict_data_extern: dict)-> pd.DataFrame:
    
    """ retourne le dataframe avec les variables auxiliaires pre-processés
    Arguments:
        - X
        - richesse_data, chomage_data, inondation_data : dataframe des variables auxiliaires

    """
    
    chomage=dict_data_extern["chomage"]
    inondation=dict_data_extern["inondation"]
    richesse=dict_data_extern["richesse"]

    richesse.set_index("Unnamed: 0",inplace=True)
    chomage.set_index("Code",inplace=True)
    inondation.set_index("Commune",inplace=True)

    dictionary_chomage=chomage.to_dict()['MOYENNE']
    dictionary_hlm=richesse.to_dict()['Taux de logements sociaux']
    dictionary_prix_loyer=richesse.to_dict()['Loyer moyen par mètre carré de surface habitable (en €)']
    #dictionary_innondation=inondation.to_dict()['Somme de nb_com_ddrm'] Inondation pas utilisé 


    departement_proxy=chomage.index.astype(str).tolist()
    #codes_postaux_proxy=inondation.index.astype(str).tolist() Inondation pas utilisé ==> Pas besoin des codes postaux 


    X['DEPARTEMENT_CRI']=X['DEPARTEMENT_CRI'].astype(str).apply(lambda x: replace_if_not_in(x,departement_proxy,X["DEPARTEMENT_CRI"]))
    X["TAUX_HLM"]=X["DEPARTEMENT_CRI"].astype(str).map(dictionary_hlm)
    X["TAUX_CHOMAGE"]=X["DEPARTEMENT_CRI"].astype(str).map(dictionary_chomage)
    X["PRIX_LOYER"]=X["DEPARTEMENT_CRI"].map(dictionary_prix_loyer)
    X["TAUX_HLM"]=X["TAUX_HLM"].fillna(X["TAUX_HLM"].median())
    X["TAUX_CHOMAGE"]=X["TAUX_CHOMAGE"].fillna(X["TAUX_CHOMAGE"].median())
    X["PRIX_LOYER"]=X["PRIX_LOYER"].fillna(X["PRIX_LOYER"].median())



# Imputation par le mode si variable géographique n'est pas présente dans les colonnes du dataframe de base
def replace_if_not_in(x,list_geo,col):

    if x not in list_geo:
        return col.value_counts().idxmax()
    else: 
        return x



""" TRAITEMENT VARIABLE CROISEE """

def cross_variable_for_project_type(X_train:pd.DataFrame):

    """ retourne le dataframe avec les variables croisées ainsi que la liste des noms de ces nouvelles variables
    Argument:
        - X_train
    """
    list_costs=[
        "COUT_BIEN_FINANCE_BRP",
        "AUTRES_COUT_BRP"]

    cross_var_mount= get_cross_var(X_train[list_costs],X_train["COD_CPPOP_CRI"])
    new_col_from_cross=cross_var_mount.columns.tolist()
    print("Nouvelles variables créées:",new_col_from_cross)
    X_train=pd.concat([X_train.reset_index(drop=True),cross_var_mount.reset_index(drop=True)],axis=1)

    return X_train,new_col_from_cross



def get_cross_var(X_train_costs: pd.DataFrame,cppop_cri:pd.Series)-> pd.DataFrame:

    results=pd.DataFrame()
    results["montants"]=X_train_costs.sum(axis=1)
    results["type"]=cppop_cri
    table = pd.pivot_table(results,values="montants", index=results.index, columns="type", fill_value=0)

    return table



    
import pandas as pd
import optbinning
import json
from sklearn.preprocessing import LabelEncoder
import numpy as np






    

def selection_categorical_var_post_grouping(X_train:pd.DataFrame,y:pd.Series,seuil_diff_tx_moyen:np.float) -> list:
    """ retourne la liste des variables qui ne sont pas discriminantes post-regroupement (pas assez de modalités ou taux de différence entre modalité pas suffisant)
    Argument: 
        - X_train: x_train avec quali sélectionnées
        - y
        - seuil_diff_tx_moyen: taux de différence (ici 0.001)
    Rq: on a laissé apparente uniquement les variables sélectionnées à la fin de ce notebook 
    """  
    quali_no_discriminant=[]
    for col in X_train.columns.tolist():
        test=pd.concat([y.reset_index(drop=True),X_train[col].reset_index(drop=True)],axis=1)
        moyenne_defaut_per_mod=test.groupby([col])["defaut_36mois"].agg("mean").sort_values()
        if X_train[col].nunique()<2:
            quali_no_discriminant.append(col)
        elif moyenne_defaut_per_mod.diff().abs().min()<seuil_diff_tx_moyen:
            quali_no_discriminant.append(col)

    return quali_no_discriminant



    

def replace_encoding_by_real_labels(X: pd.DataFrame)-> pd.DataFrame:

    """ retourne le dataframe avec variables encodées avec leurs intitulés dans le lexique
    Argument: 
        - X
    Rq: on a laissé apparente uniquement les variables sélectionnées à la fin de ce notebook 
    """  

    COD_ETA_BIEN_CRI={"10":"neuf",
    "20":"vente en état futur d'achèvement",
    "30":"contrat construction maison individuelle",
    "40":"ancien_inf_10_ans",
    "60":"ancien_sup_10_ans",
    "50":"ancien_sup_10_ans",
    "70":"clé en main avec levée d'option"}


    CODTYP_CRT_TRAVAIL_CRI={"1":"cdi et professions libérales",
    "2":"cdi et professions libérales",
    "3":"cdd et intérim",
    "4":"fonctionnaire ou agent public",
    "5":"fonctionnaire ou agent public",
    "6":"fonctionnaire ou agent public",
    "7":"cdd et intérim","8":"chomage, retraités, inactifs",
    "9":"chomage, retraités, inactifs",
    "A":"fonctionnaire ou agent public",
    "B":"fonctionnaire ou agent public",
    "Y":"chomage, retraités, inactifs",
    "Z":"cdi et professions libérales"}

    COD_SITU_LOGT_CRI={"10.0":"propriétaire",
    "20.0":"propriétaire accédant",
    "30.0":"locataire hlm",
    "40.0":"locataire autre",
    "50.0":"locataire de fonction",
    "60.0":"occupant gratuit",
    "70.0":"logement parents",
    "nan":"autres",
    "900.°":"autres"}

    COD_CPPOP_CRI={"10":"ACQUISITION_SEULE",
    "20":"ACQUISITION_TRAVAUX",
    "70":"RACHAT_DE_PRET",
    "30":"TERRAIN_CONSTRUCTION",
    "40":"CONSTRUCTION_SEULE",
    "60":"TRAVAUX",
    "50":"TRAVAUX_CONSTRUCTION",
    "80":"PAIEMENT_SOULTE",
    "90":"PAIEMENT_SOULTE_TRAVAUX",
    "110":"RACHAT_DE_PRET",
    "130":"RACHAT_DE_PRET"}


    COD_TYPE_MARCHE_CRI={"M1":"M1",
    "M21":"M2",
    "M2":"M2"}


    # Replace with map function from dict values
    try:
        X["COD_ETA_BIEN_CRI"]=X["COD_ETA_BIEN_CRI"].map(COD_ETA_BIEN_CRI)
    except:
        print("Check COD_ETA_BIEN_CRI is selected")

    try:
        X["CODTYP_CRT_TRAVAIL_CRI"]=X["CODTYP_CRT_TRAVAIL_CRI"].map(CODTYP_CRT_TRAVAIL_CRI)
    except:
        print("Check CODTYP_CRT_TRAVAIL_CRI is selected")
    try:
        X["COD_SITU_LOGT_CRI"]=X["COD_SITU_LOGT_CRI"].map(COD_SITU_LOGT_CRI)
    except:
        print("Check COD_SITU_LOGT_CRI is selected")
    try:
        X["COD_TYPE_MARCHE_CRI"]=X["COD_TYPE_MARCHE_CRI"].map(COD_TYPE_MARCHE_CRI)
    except:
        print("Check COD_TYPE_MARCHE_CRI is selected")
    try:
        X["COD_CPPOP_CRI"]=X["COD_CPPOP_CRI"].map(COD_CPPOP_CRI)
    except:
        print("Check COD_CPPOP_CRI is selected")


    return X


def encoding_categorical_variables(X:pd.DataFrame,dict_encoding:dict,list_quali_var:list):

    for col in list_quali_var:
        X[col]=X[col].fillna(X[col].mode()[0])
        X[col].replace(dict_encoding[col], inplace=True)
        mod=X[col].mode()[0]
        X[col]=X[col].apply(lambda x: replace_by_mod_if_not_encoded(x,dict_encoding[col].values(),mod))
        X[col]=X[col].fillna(X[col].mode()[0])
        X[col]=X[col].astype("category")

def replace_by_mod_if_not_encoded(x,dict_encoding_values,mod):
    if x not in dict_encoding_values:
        x=mod
    else:
        x=x
    return x






def group_modalities_with_optbinning(X_train: pd.DataFrame,y_train:pd.Series,list_to_group= None,cat_cutoff=0.1,min_event_rate_diff=0.01,encoding=False,display_=True) -> dict:

    """ retourne le dictionnaire des encodings des variables selon le découpage de Optibinning
    Argument: 
        - X_train: X train avec variables qualitatives précedemment sélectionnées
        - y_train: y 
        - list_to_group
        - cat_cutoff: minimum effectif pour une modalité (5% typiquement)
        - min_event_rate_diff : taux de différence de la survenu de l'évenement minimum entre modalités souhaité
        - encoding: est-ce que on encode à ce moment ou non
        - display: display ou non les tableaux de binning
    """ 

    le = LabelEncoder()
    dict_all = dict(zip([], []))

    if list_to_group == None:

        for col in X_train.select_dtypes(include="O").columns:

            optb = optbinning.OptimalBinning(name=col,dtype="categorical",cat_cutoff=cat_cutoff,min_event_rate_diff=min_event_rate_diff)
            optb.fit(X_train[col],y_train)
            print("################################", col, "################################")
            print("STATUS :",optb.status)

            if display_==True:
                display(optb.binning_table.build())

            if encoding==True:
                binned_var=optb.transform(X_train[col],metric="bins")
                print("ENCODING...")
                temp_keys = X_train[col].values
                temp_values = le.fit_transform(binned_var)
                dict_temp = dict(zip(temp_keys, temp_values))
                dict_all[col] = dict_temp
 


    else:

        for col in list_to_group: 

            optb = optbinning.OptimalBinning(name=col,dtype="categorical",cat_cutoff=cat_cutoff,min_event_rate_diff=min_event_rate_diff)
            optb.fit(X_train[col],y_train)
            print("################################", col, "################################")
            print("STATUS :",optb.status)

            if display_==True:
                display(optb.binning_table.build())

            if encoding==True:
                binned_var=optb.transform(X_train[col],metric="bins")
                print("ENCODING...")
                temp_keys = X_train[col].values
                temp_values = le.fit_transform(binned_var)
                dict_temp = dict(zip(temp_keys, temp_values))
                dict_all[col] = dict_temp
                
    with open('doc/encoding_categorical_variables.json', 'w') as fp:
        json.dump(dict_all, fp,  cls=NpEncoder)

    return dict_all






# Classe pour encoder le Json avec des variables numériques 
class NpEncoder(json.JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)




"""  REGROUPEMENT DE MODALITES INITIAL """

""""
def liste_quali_a_regrouper(X_train:pd.DataFrame):

    liste_inf_5_modalites=[]
    liste_sup_5_modalites=[]
    df_quali=X_train.select_dtypes(include="object")

    for col in df_quali.columns:

        if df_quali[col].nunique() <= 5:
            liste_inf_5_modalites.append(col)

        else:
            liste_sup_5_modalites.append(col)

    return liste_inf_5_modalites,liste_sup_5_modalites



def regrouper_modalites(X,y,liste_var_to_group,drop=False):

    " Fonction qui regroupe les modalités qui ont un effectif inférieur à 5%, soit par le mode majoritaire, mais au mieux par la modalité au taux de défaut le plus proche
    retourne le dataframe
    Argument: 
        - X
        - y
        - liste_var_to_group: liste des variables catégorielles à regrouper

    liste_col_drop=[]

    for col in liste_var_to_group:

        print(col)
        dic_for_encode={}
        dict_modalites_count=dict(sorted(X[col].value_counts(normalize=True).to_dict().items(), key=lambda item:item[1], reverse=False))

        for key,value in dict_modalites_count.items():
            df_defaut_mod=pd.crosstab(X[col],y,normalize=True)

            if value<0.05:
                key_defaut=df_defaut_mod.loc[key,1]
                nearest_mod = df_defaut_mod.iloc[(df_defaut_mod[1]-key_defaut).abs().argsort()[:2]].index.tolist()[1]

                if len(X[X[col]==nearest_mod]) < 0.05*len(X):
                    dic_for_encode[key]=X[col].value_counts().idxmax()
                    print(key," est intégré dans la modalité ",X[col].value_counts().idxmax(), "(classe majoritaire)")

                else:
                    dic_for_encode[key]=nearest_mod
                    print(key," est intégré dans la modalité ", nearest_mod)


        if X[col].nunique()==1:
            print(f"La colonne {col} ne comporte plus qu'une modalité, elle n'est pas suffisamment discriminante")

            if drop==True:
                X.drop(columns=col,inplace=True)
                liste_col_drop=None
                
            else: 
                liste_col_drop.append(col)

    return X, liste_col_drop,dic_for_encode
""" 




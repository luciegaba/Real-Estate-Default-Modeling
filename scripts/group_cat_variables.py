import pandas as pd
import optbinning
from sklearn.preprocessing import LabelEncoder



def liste_quali_a_regrouper(X):
    liste_inf_5_modalites=[]
    liste_sup_5_modalites=[]
    df_quali=X.select_dtypes(include="object")
    for col in df_quali.columns:
        if df_quali[col].nunique() <= 5:
            liste_inf_5_modalites.append(col)
        else:
            liste_sup_5_modalites.append(col)
    return liste_inf_5_modalites,liste_sup_5_modalites




""" def replace_encoding_with_grouping(X_train):

    COD_USAGE_BIEN_CRI={"10":"residence principale", 
    "20": "residence secondaire", 
    "30":"residence secondaire",
    "40":"locatif",
    "50":"locatif",
    "60":"locatif"}



    QUA_INT_MAX_BRP={"2":"monsieur",
    "3":"madame",
    "4":"mademoiselle",
    "7":"entité",
    "5":"entité",
    "6":"entité"}	

    COD_ETA_BIEN_CRI={"10":"neuf",
    "20":"neuf",
    "30":"neuf",
    "40":"récent_inf_10_ans",
    "60":"ancien_sup_10_ans",
    "50":"ancien_sup_10_ans",
    "70":"autres",
    "110":"autres",
    "120":"autres",
    "130":"autres",
    "140":"autres",
    "150":"autres"}

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

    CODTYPE_PROJET_CRI={"10":"appartement",
    "20":"maison",
    "30":"maison",
    "70":"local mixte et professionnel",
    "80":"local mixte et professionnel",
    "90":"sci/scpi",
    "100":"sci/scpi",
    "130":"autres",
    "140":"autres",
    "50":"autres",
    "60":"autres",
    "40":"autres"}

    COD_SITU_LOGT_CRI={"10.0":"propriétaire",
    "20.0":"propriétaire",
    "30.0":"locataire",
    "40.0":"locataire",
    "50.0":"locataire",
    "60.0":"logement gratuit",
    "70.0":"logement gratuit",
    "nan":"autres",
    "900.°":"autres"}

    COD_SIT_FAM_EMPRUNTEUR_CRI={"1.0":"seul",
    "4.0":"seul",
    "6.0":"seul",
    "5.0":"seul",
    "2.0":"marié ou union",
    "3.0":"marié ou union",
    "nan":"seul"}
    COD_TYPE_MARCHE_CRI={"M1":"M1",
    "M21":"M2",
    "M2":"M2"}
    CSP_RGP_BRP={"1.0": "agriculteurs exploitants",
    "2.0":"artisans, commerçants et chefs d'entreprise",
    "3.0":"cadres et professions intellectuelles supérieures",
    "4.0":"professions Intermédiaires",
    "5.0":"employés",
    "6.0":"ouvriers",
    "7.0":"retraités",
    "8.0":"autres personnes sans activité professionnelle",
    "nan": "autres personnes sans activité professionnelle"}


    X_train["COD_USAGE_BIEN_CRI"]=X_train["COD_USAGE_BIEN_CRI"].map(COD_USAGE_BIEN_CRI)
    X_train["QUA_INT_MAX_BRP"]=X_train["QUA_INT_MAX_BRP"].map(QUA_INT_MAX_BRP)
    X_train["COD_ETA_BIEN_CRI"]=X_train["COD_ETA_BIEN_CRI"].map(COD_ETA_BIEN_CRI)
    X_train["CODTYP_CRT_TRAVAIL_CRI"]=X_train["CODTYP_CRT_TRAVAIL_CRI"].map(CODTYP_CRT_TRAVAIL_CRI)
    X_train["CODTYPE_PROJET_CRI"]=X_train["CODTYPE_PROJET_CRI"].map(CODTYPE_PROJET_CRI)
    X_train["COD_SITU_LOGT_CRI"]=X_train["COD_SITU_LOGT_CRI"].map(COD_SITU_LOGT_CRI)
    X_train["COD_SIT_FAM_EMPRUNTEUR_CRI"]=X_train["COD_SIT_FAM_EMPRUNTEUR_CRI"].map(COD_SIT_FAM_EMPRUNTEUR_CRI)
    X_train["COD_TYPE_MARCHE_CRI"]=X_train["COD_TYPE_MARCHE_CRI"].map(COD_TYPE_MARCHE_CRI)
    X_train["CSP_RGP_BRP"]=X_train["CSP_RGP_BRP"].map(CSP_RGP_BRP)
    return X_train"""



def regrouper_modalites(X,y,liste,drop=False):
    liste_col_drop=[]
    dic_for_all_encoding={}
    for col in liste:
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

        dic_for_all_encoding[col]=dic_for_encode
        if X[col].nunique()==1:
            print(f"La colonne {col} ne comporte plus qu'une modalité, elle n'est pas suffisamment discriminante")
            if drop==True:
                X.drop(columns=col,inplace=True)
                liste_col_drop=None
            else: 
                liste_col_drop.append(col)

    return X, liste_col_drop,dic_for_encode



    

def replace_encoding_by_real_mod(X):

    COD_USAGE_BIEN_CRI={"10":"residence principale", 
    "20": "residence secondaire", 
    "30":"residence de retraite",
    "40":"locatif principal",
    "50":"locatif secondaire",
    "60":"locatif professionnel"}



    QUA_INT_MAX_BRP={"2":"monsieur",
    "3":"madame",
    "4":"mademoiselle",
    "7":"entité",
    "5":"entité",
    "6":"entité"}	

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

    CODTYPE_PROJET_CRI={"10":"appartement",
    "20":"maison individuelle",
    "30":"terrain constructible",
    "70":"local professionnel",
    "80":"local mixte",
    "90":"sci/scpi",
    "100":"sci/scpi",
    "130":"bien non destiné au logement",
    "140":"péniche",
    "50":"garage, box, parking",
    "60":"annexe(s), piscine",
    "40":"terrain non constructible"}

    COD_SITU_LOGT_CRI={"10.0":"propriétaire",
    "20.0":"propriétaire accédant",
    "30.0":"locataire hlm",
    "40.0":"locataire autre",
    "50.0":"locataire de fonction",
    "60.0":"occupant gratuit",
    "70.0":"logement parents",
    "nan":"autres",
    "900.°":"autres"}

    COD_SIT_FAM_EMPRUNTEUR_CRI={"1.0":"célibataire",
    "4.0":"veuf",
    "6.0":"séparé",
    "5.0":"divorcé",
    "2.0":"marié",
    "3.0":"union libre",
    "nan":"célibataire"}
    COD_TYPE_MARCHE_CRI={"M1":"M1",
    "M21":"M2",
    "M2":"M2"}
    CSP_RGP_BRP={"1.0": "agriculteurs exploitants",
    "2.0":"artisans, commerçants et chefs d'entreprise",
    "3.0":"cadres et professions intellectuelles supérieures",
    "4.0":"professions Intermédiaires",
    "5.0":"employés",
    "6.0":"ouvriers",
    "7.0":"retraités",
    "8.0":"autres personnes sans activité professionnelle",
    "nan": "autres personnes sans activité professionnelle"}

    X["COD_USAGE_BIEN_CRI"]=X["COD_USAGE_BIEN_CRI"].map(COD_USAGE_BIEN_CRI)
    X["QUA_INT_MAX_BRP"]=X["QUA_INT_MAX_BRP"].map(QUA_INT_MAX_BRP)
    X["COD_ETA_BIEN_CRI"]=X["COD_ETA_BIEN_CRI"].map(COD_ETA_BIEN_CRI)
    X["CODTYP_CRT_TRAVAIL_CRI"]=X["CODTYP_CRT_TRAVAIL_CRI"].map(CODTYP_CRT_TRAVAIL_CRI)
    X["CODTYPE_PROJET_CRI"]=X["CODTYPE_PROJET_CRI"].map(CODTYPE_PROJET_CRI)
    X["COD_SITU_LOGT_CRI"]=X["COD_SITU_LOGT_CRI"].map(COD_SITU_LOGT_CRI)
    X["COD_SIT_FAM_EMPRUNTEUR_CRI"]=X["COD_SIT_FAM_EMPRUNTEUR_CRI"].map(COD_SIT_FAM_EMPRUNTEUR_CRI)
    X["COD_TYPE_MARCHE_CRI"]=X["COD_TYPE_MARCHE_CRI"].map(COD_TYPE_MARCHE_CRI)
    X["CSP_RGP_BRP"]=X["CSP_RGP_BRP"].map(CSP_RGP_BRP)
    return X




def encoding_col_and_cat(X,dict_encoding,list_quali_var):
    for col in list_quali_var:
        X[col]=X[col].fillna(X[col].mode()[0])
        X[col].replace(dict_encoding[col], inplace=True)
        X[col]=X[col].fillna(X[col].mode()[0])
        X[col]=X[col].astype("category")
 

def select_quali_variables(X_train_quali,y,seuil_diff_tx_moyen):
    quali_too_low_mod=[]
    quali_no_discriminant=[]
    quali_good=[]
    for col in X_train_quali.columns.tolist():
        test=pd.concat([y.reset_index(drop=True),X_train_quali[col].reset_index(drop=True)],axis=1)
        moyenne_defaut_per_mod=test.groupby([col])["defaut_36mois"].agg("mean").sort_values()
        if X_train_quali[col].nunique()<2:
            quali_too_low_mod.append(col)
        elif moyenne_defaut_per_mod.diff().abs().min()<seuil_diff_tx_moyen:
            quali_no_discriminant.append(col)
        else:
            quali_good.append(col)
    return quali_too_low_mod,quali_no_discriminant,quali_good

def group_modalities_with_optbinning(X_train,y_train,liste=None,cat_cutoff=0.1,min_event_rate_diff=0.01,encoding=False):
    le = LabelEncoder()
    dict_all = dict(zip([], []))
    if liste==None:
        for col in X_train.select_dtypes(include="O").columns:
            optb = optbinning.OptimalBinning(name=col,dtype="categorical",cat_cutoff=cat_cutoff,min_event_rate_diff=min_event_rate_diff)

            optb.fit(X_train[col],y_train)
            print("################################", col, "################################")
            print("STATUS :",optb.status)
            display(optb.binning_table.build())
            print(pd.Series(optb.transform(X_train[col],metric="bins")).value_counts())

            if encoding==True:
                binned_var=optb.transform(X_train[col],metric="bins")
                print("ENCODING...")
                temp_keys = X_train[col].values
                temp_values = le.fit_transform(binned_var)
                dict_temp = dict(zip(temp_keys, temp_values))
                dict_all[col] = dict_temp
 
        return dict_all
    else:
        for col in liste: 
            optb = optbinning.OptimalBinning(name=col,dtype="categorical",cat_cutoff=cat_cutoff,min_event_rate_diff=min_event_rate_diff)
            optb.fit(X_train[col],y_train)
            print("################################", col, "################################")
            print("STATUS :",optb.status)
            display(optb.binning_table.build())
            print(pd.Series(optb.transform(X_train[col],metric="bins")).value_counts())
            if encoding==True:
                binned_var=optb.transform(X_train[col],metric="bins")
                print(X_train[col].value_counts())
                print("ENCODING...")
                temp_keys = X_train[col].values
                temp_values = le.fit_transform(binned_var)
                dict_temp = dict(zip(temp_keys, temp_values))
                dict_all[col] = dict_temp

        return dict_all
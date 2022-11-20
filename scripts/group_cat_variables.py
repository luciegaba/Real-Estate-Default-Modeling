import pandas as pd

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




def replace_encoding_by_real_mod(X_train):

    COD_USAGE_BIEN_CRI={"10":"residence principale", 
    "20": "residence secondaire", 
    "30":"residence secondaire",
    "40":"locatif",
    "50":"locatif",
    "60":"locatif"}

    COD_CPPOP_CRI={"10":"achat",
    "20":"achat",
    "70":"achat",
    "30":"construction",
    "40":"construction",
    "60":"travaux",
    "50":"travaux",
    "80":"soulte",
    "90":"soulte",
    "110":"achat",
    "130":"achat"}

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
    X_train["COD_CPPOP_CRI"]=X_train["COD_CPPOP_CRI"].map(COD_CPPOP_CRI)
    X_train["QUA_INT_MAX_BRP"]=X_train["QUA_INT_MAX_BRP"].map(QUA_INT_MAX_BRP)
    X_train["COD_ETA_BIEN_CRI"]=X_train["COD_ETA_BIEN_CRI"].map(COD_ETA_BIEN_CRI)
    X_train["CODTYP_CRT_TRAVAIL_CRI"]=X_train["CODTYP_CRT_TRAVAIL_CRI"].map(CODTYP_CRT_TRAVAIL_CRI)
    X_train["CODTYPE_PROJET_CRI"]=X_train["CODTYPE_PROJET_CRI"].map(CODTYPE_PROJET_CRI)
    X_train["COD_SITU_LOGT_CRI"]=X_train["COD_SITU_LOGT_CRI"].map(COD_SITU_LOGT_CRI)
    X_train["COD_SIT_FAM_EMPRUNTEUR_CRI"]=X_train["COD_SIT_FAM_EMPRUNTEUR_CRI"].map(COD_SIT_FAM_EMPRUNTEUR_CRI)
    X_train["COD_TYPE_MARCHE_CRI"]=X_train["COD_TYPE_MARCHE_CRI"].map(COD_TYPE_MARCHE_CRI)
    X_train["CSP_RGP_BRP"]=X_train["CSP_RGP_BRP"].map(CSP_RGP_BRP)
    return X_train



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
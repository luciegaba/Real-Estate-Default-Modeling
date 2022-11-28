import pandas as pd
from scipy import stats
from scipy.stats import pointbiserialr
import scipy
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.model_selection  import train_test_split,GridSearchCV,StratifiedKFold
import warnings 
warnings.filterwarnings("ignore")
import sklearn


def pipeline(stab_data,dict_encoding):
    """en input on met le dataframe. La fonction devrait faire le preprocessing, la discrétisation quali, la discrétisation quanti 
    """
    #PREPROCESSING
    #Conversion format date
    date_columns=["DAT_VALIDE_ACCORD_CRI","date_debloc_avec_crd"]
    stab_data["date_debloc_avec_crd"]=pd.to_datetime(stab_data["date_debloc_avec_crd"],format="%Y%m")

    #je ne mets pas le split OOT 
    #je ne mets pas le split train, test
    
    colonnes_drop=['TOP_GARANTIE_CL',"ID", 'DAT_VALIDE_ACCORD_CRI']
    stab_data.drop(columns= colonnes_drop, inplace = True)
    
    ###
    print("warning :Variables dummies")
    ###
    list_dummies = ['TOP_CONNU_BRP', 'TOP_SCI_BRP', 'TOP_ETR_BRP', 'TOP_PRET_RELAIS_BRP', 'TOP_SURFINANCEMENT_BRP', 'top_exist_conso_revo_BRP', 'TOP_ASC_DESC_BRP', 'ROL_INT_MAX_BRP', 'IND_INCIDENT_BDF_CRI', 'IND_PRIMO_ACCEDT_CRI', 'TOP_BIEN_FR_CRI', 'TOP_NAT_FR_CRI', 'top_locatif', 'top_pret_int_ext', 'top_autre_pret_int', 'top_autre_pret_ext', 'top_pers_seule']
    convert_modalities_to_quali(stab_data,list_dummies)

    ###
    print("warning : modalities_var")
    ###
    modalities_var=["TYP_CNT_TRA_MAX_BRP",
                    "CODTYPE_PROJET_CRI",
                    "COD_ETA_BIEN_CRI",
                    "COD_USAGE_BIEN_CRI",
                    "STA_CLP_BRP",
                    "TYP_LOG_ACT_BRP",
                    "QUA_INT_MAX_BRP",
                    "ROL_INT_MAX_BRP",
                    "CODTYP_CRT_TRAVAIL_CRI",
                    "COD_SITU_LOGT_CRI",
                    "COD_SIT_FAM_EMPRUNTEUR_CRI",
                    "COD_TYPE_MARCHE_CRI",
                    "NAT_BIEN_FIN_BRP",
                    "SIT_FAM_INT_BRP",
                    "COD_CPPOP_CRI", 
                    "COD_CSP_BRP",
                    "CSP_RGP_BRP",
                    "ASU_BIEN_FIN_BRP",
                    "CODTYPE_PROJET_CRI",
                    "QUA_INT_1_BRP",
                    "CODPAY_NAT_EMPRUNTEUR_CRI",
                    "DEPARTEMENT_CRI",
                    "COD_POSTAL_BIEN_CRI"]  
    convert_modalities_to_quali(stab_data, modalities_var)

    ###
    print("warning : high_na")
    ###
    high_na = ['FINANCEMENT_PR_BRP', 'SUM_MNT_PRE_INTERNE_BRP', 'TX_APPORT_AGENCE_BRP', 'TX_FINANCEMENT_AGENCE_BRP', 'TX_APPORT_BRP', 'TX_FINANCEMENT_BRP', 'mnt_pret_ext_brp', 'SUM_LOYER_BRP', 'SUM_AUTRES_CHARG_BRP']
    drop_columns(stab_data,high_na)

    ###
    print("warning : colonnes_doublons_to_drop")
    ###
    colonnes_doublons_to_drop = ['NAT_BIEN_FIN_BRP', 'ASU_BIEN_FIN_BRP', 'TYP_CNT_TRA_MAX_BRP', 'SIT_FAM_INT_BRP', 'TYP_LOG_ACT_BRP']
    drop_columns(stab_data,colonnes_doublons_to_drop)

    ###
    print("warning : variables_a_fill_0 ET variables_na_to_drop")
    ###
    variables_a_fill_0 = ['SUM_RESS_IMMO_BRP', 'SUM_PATR_IMMO_BRP', 'SUM_MTENCBIE_IMMO_BRP', 'SUM_MNT_PRE_EXTERNE_BRP', 'SUM_EPARGNE_BRP', 'SUM_MTENCBIE_EPARGNE_BRP', 'SUM_TAX_FISC_BRP','COUT_NOTAIRE_BRP','COUT_ACQ_BRP']
    variables_na_to_drop = ['rentabilite_loc', 'Epargne_nb_ans_rev_prof']
    fill_by_0(stab_data, variables_a_fill_0)
    drop_columns(stab_data,variables_na_to_drop)

    imputation_for_na(stab_data)

    ###
    print("warning : list_geo_features")
    ###
    list_geo_features=["region_cri","COD_POSTAL_BIEN_CRI","DEPARTEMENT_CRI","NAT_INT_MAX_BRP","CODPAY_NAT_EMPRUNTEUR_CRI","CODPAY_BIEN_CRI"]
    drop_columns(stab_data,list_geo_features)
    
    #Discrétisation :
    
    ###
    print("warning : une liste au tout début de discrétisation + regroupement de modalités")
    ###
    drop_columns(stab_data,["COD_CSP_BRP","STA_CLP_BRP","ROL_INT_MAX_BRP","TOP_CONNU_BRP","COD_CPPOP_CRI","IRPRO_USAGE_V12_MAX"])
    
    #Quali
    list_quali=['TOP_SCI_BRP', 'CODTYPE_PROJET_CRI', 'COD_ETA_BIEN_CRI', 'COD_USAGE_BIEN_CRI', 'TOP_ETR_BRP', 'TOP_PRET_RELAIS_BRP', 'TOP_SURFINANCEMENT_BRP', 'top_exist_conso_revo_BRP', 'IRPAR_USAGE_V12_MAX', 'TOP_ASC_DESC_BRP', 'QUA_INT_1_BRP', 'QUA_INT_MAX_BRP', 'CODTYP_CRT_TRAVAIL_CRI', 'COD_SITU_LOGT_CRI', 'COD_SIT_FAM_EMPRUNTEUR_CRI', 'COD_TYPE_MARCHE_CRI', 'IND_INCIDENT_BDF_CRI', 'IND_PRIMO_ACCEDT_CRI', 'TOP_BIEN_FR_CRI', 'TOP_NAT_FR_CRI', 'CSP_RGP_BRP', 'top_locatif', 'top_pret_int_ext', 'top_autre_pret_int', 'top_autre_pret_ext', 'top_pers_seule']

    replace_encoding_by_real_mod(stab_data)

    #replace_encoding_by_real_mod(stab_data)
    dict_encoding={'TOP_SCI_BRP': {'0': 0, '1': 1},
 'CODTYPE_PROJET_CRI': {'maison individuelle': 2,
  'appartement': 0,
  'terrain non constructible': 1,
  'sci/scpi': 1,
  'garage, box, parking': 1,
  'local professionnel': 1,
  'local mixte': 1,
  'terrain constructible': 1,
  'annexe(s), piscine': 1,
  'péniche': 1},
 'COD_ETA_BIEN_CRI': {'ancien_sup_10_ans': 3,
  "vente en état futur d'achèvement": 1,
  'ancien_inf_10_ans': 2,
  'neuf': 4,
  'contrat construction maison individuelle': 4,
  None: 0,
  "clé en main avec levée d'option": 4},
 'COD_USAGE_BIEN_CRI': {'residence principale': 1,
  'locatif principal': 0,
  'residence secondaire': 2,
  'locatif secondaire': 2,
  'locatif professionnel': 2,
  'residence de retraite': 2},
 'TOP_ETR_BRP': {'0': 0, '1': 1},
 'TOP_PRET_RELAIS_BRP': {'0': 0, '1': 1},
 'TOP_SURFINANCEMENT_BRP': {'0': 0, '1': 1},
 'top_exist_conso_revo_BRP': {'0': 0, '1': 1},
 'IRPAR_USAGE_V12_MAX': {'4.0': 2,
  '3.0': 1,
  '2.0': 0,
  '1.0': 2,
  '8.0': 2,
  '6.0': 2,
  '7.0': 2,
  '5.0': 2,
  '10.0': 2,
  'nan': 2,
  '9.0': 2},
 'TOP_ASC_DESC_BRP': {'0': 0, '1': 1},
 'QUA_INT_1_BRP': {'2': 0, '3': 1, '4': 2, '5': 1},
 'QUA_INT_MAX_BRP': {'monsieur': 3,
  'madame': 1,
  'mademoiselle': 2,
  'entité': 0},
 'CODTYP_CRT_TRAVAIL_CRI': {'cdi et professions libérales': 0,
  'fonctionnaire ou agent public': 2,
  'cdd et intérim': 1,
  'chomage, retraités, inactifs': 1},
 'COD_SITU_LOGT_CRI': {'logement parents': 2,
  'locataire autre': 1,
  'propriétaire accédant': 3,
  'propriétaire': 4,
  'locataire hlm': 2,
  'locataire de fonction': 2,
  'occupant gratuit': 2,
  'autres': 2,
  None: 0},
 'COD_SIT_FAM_EMPRUNTEUR_CRI': {'célibataire': 0,
  'marié': 2,
  'divorcé': 1,
  'union libre': 3,
  'séparé': 1,
  'veuf': 1},
 'COD_TYPE_MARCHE_CRI': {'M1': 0, 'M2': 1},
 'IND_INCIDENT_BDF_CRI': {'N': 0, 'O': 1},
 'IND_PRIMO_ACCEDT_CRI': {'2.0': 1, 'nan': 2, '1.0': 0},
 'TOP_BIEN_FR_CRI': {'1': 1, '0': 0},
 'TOP_NAT_FR_CRI': {'1': 1, '0': 0},
 'CSP_RGP_BRP': {'professions Intermédiaires': 3,
  'ouvriers': 2,
  'cadres et professions intellectuelles supérieures': 0,
  'employés': 1,
  'retraités': 2,
  "artisans, commerçants et chefs d'entreprise": 2,
  'autres personnes sans activité professionnelle': 2,
  'agriculteurs exploitants': 2},
 'top_locatif': {'0': 0, '1': 1},
 'top_pret_int_ext': {'0': 0, '1': 1},
 'top_autre_pret_int': {'0': 0, '1': 1},
 'top_autre_pret_ext': {'0': 0, '1': 1},
 'top_pers_seule': {'1': 1, '0': 0}}

    encoding_col_and_cat(stab_data,dict_encoding,list_quali)

    #---> faut faire un .map pour regrouper les modalités selon le dictionnaire
    
    #et s'il faut, on doit drop les variables avec une seule modalité
    
    #Quanti
    discretisation_variables_from_chi2(stab_data)#discrétisation des variables quanti

    return stab_data




"""
Les fonctions pour contruire les graphiques de stabilité
"""
#création de deux variables pour tracer le graphique de la stabilitié temporelle, une avec l'année, une avec
def intervalles_annee(dfff):
    dfff['intervalles_dates_annee']=dfff['date_debloc_avec_crd'].map(lambda x: '{year}'.format(year=x.year))
def intervalles_semestre(dfff):
    dfff['intervalles_dates_semestre']=dfff['date_debloc_avec_crd'].map(lambda x: '{year}-1'.format(year=x.year) if x.month<=6 else '{year}-2'.format(year=x.year))

def stability_according_to_the_target(dfff, col_date, otherdate):
    for col in dfff.drop([otherdate,col_date, 'defaut_36mois'], axis=1):
        print(col)
        df_des_valeurs=pd.DataFrame(dfff.groupby(col_date)[col].value_counts())
        df_des_valeurs1=pd.DataFrame(dfff[dfff['defaut_36mois']==1].groupby(col_date)[col].value_counts())

        df_des_valeurs1index=df_des_valeurs1.reset_index(inplace=False, level=0)
        df_des_valeurs1index.index.names = ['groupes']
        df_des_valeurs1index=df_des_valeurs1index.reset_index(inplace=False)

        df_des_valeursindex=df_des_valeurs.reset_index(inplace=False, level=0)
        df_des_valeursindex.index.names = ['groupes']
        df_des_valeursindex=df_des_valeursindex.reset_index(inplace=False)
        mergedvar = pd.merge(df_des_valeursindex, df_des_valeurs1index, on =[col_date,'groupes'], how ="outer")
        #print(mergedvar.columns)
        mergedvar[col+'%']=np.divide(mergedvar[col+'_y'], mergedvar[col+'_x'])*100
        mergedvar.set_index(col_date, inplace=True)
        mergedvar.groupby('groupes')[col+'%'].plot(legend=True)
        plt.show()
        
def stability_by_repartition_of_modalities(df,date, otherdate):
    for col in df.drop([otherdate, date, 'defaut_36mois'], axis=1):
        df_col= df.groupby([date, col]).size().unstack()
        list_col_drop=[]
        #print(df_col)
        for i in df_col:
            df_col_new=df_col
            df_col_new[i, '%'] = df_col_new[i]*100/df_col_new.iloc[:,:len(df_col_new.columns)].sum(axis=1)
            list_col_drop.append(i)
        df_col_new=df_col_new.drop(list_col_drop, axis=1)
        #print(df_col_new)
        df_col_new.plot()
    
def stability_by_repartition_of_defaults(df,date, otherdate):
    for col in df.drop([otherdate, date, 'defaut_36mois'], axis=1):
        df_col = df.loc[df['defaut_36mois'] == 1 ].groupby([date, col]).size().unstack()
        list_col_drop=[]
        #print(df_col)
        for i in df_col:
            df_col_new=df_col
            df_col_new[i,'en %'] = df_col_new[i]*100/df_col_new.iloc[:,:len(df_col_new.columns)].sum(axis=1)
            list_col_drop.append(i)
        df_col_new=df_col_new.drop(list_col_drop, axis=1)
        #print(df_col_new)
        df_col_new.plot()
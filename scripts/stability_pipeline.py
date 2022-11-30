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
import json

from scripts.firststep_dataviz import min_max_for_datetime_col,missing_rate_report,stabilite_global_temps
from scripts.processing import convert_modalities_to_quali,na_sup_20_fill_by_0_vs_autres,comparer_na_variables_doublons,fill_by_0,imputation_for_na,get_dummies_var,drop_columns,proxys_processing,replace_if_not_in
from scripts.feature_selection import test_chi2_all_quali_variables, test_pointbiserial_all_quanti_variables,cramers_v_all_cat_var,selection_avec_lasso,non_significativité_chi2
from scripts.discretisation import discretisation_variables_from_chi2,verification_par_moyenne_defaut, get_binned_df
from scripts.group_cat_variables import liste_quali_a_regrouper, replace_encoding_by_real_mod, regrouper_modalites,group_modalities_with_optbinning,encoding_col_and_cat,select_quali_variables, NpEncoder

def processing_pipeline(stab_data,richesse,chomage,inondation):
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
                "COD_POSTAL_BIEN_CRI",
                "IRPAR_USAGE_V12_MAX",
                "IRPRO_USAGE_V12_MAX",
                "NBR_INT_BRP",
                "NBR_TOT_COEMPR_CRI",
                "NBR_ENF_ACHARGE_CRI",
                "NBR_OCCUP_CRI",
                ]  
    convert_modalities_to_quali(stab_data, modalities_var)

    ###
    print("warning : high_na")
    ###
    high_na = ['FINANCEMENT_PR_BRP', 'IRPRO_USAGE_V12_MAX', 'SUM_MNT_PRE_INTERNE_BRP', 'TX_APPORT_AGENCE_BRP', 'TX_FINANCEMENT_AGENCE_BRP', 'TX_APPORT_BRP', 'TX_FINANCEMENT_BRP', 'mnt_pret_ext_brp', 'SUM_LOYER_BRP', 'SUM_AUTRES_CHARG_BRP']
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

    proxys_processing(stab_data,richesse,chomage,inondation)
    list_geo_features=["region_cri","COD_POSTAL_BIEN_CRI","DEPARTEMENT_CRI","NAT_INT_MAX_BRP","CODPAY_NAT_EMPRUNTEUR_CRI","CODPAY_BIEN_CRI"]
    drop_columns(stab_data,list_geo_features)
    
    #Discrétisation :
    
    ###
    print("warning : une liste au tout début de discrétisation + regroupement de modalités")
    ###
    drop_columns(stab_data,["COD_CSP_BRP","STA_CLP_BRP","ROL_INT_MAX_BRP","TOP_CONNU_BRP"])
    
    #Quali
    list_quali=['TOP_SCI_BRP', 'NBR_INT_BRP', 'NBR_TOT_COEMPR_CRI', 'CODTYPE_PROJET_CRI', 'COD_CPPOP_CRI', 'COD_ETA_BIEN_CRI', 'COD_USAGE_BIEN_CRI', 'TOP_ETR_BRP', 'TOP_PRET_RELAIS_BRP', 'TOP_SURFINANCEMENT_BRP', 'top_exist_conso_revo_BRP', 'IRPAR_USAGE_V12_MAX', 'TOP_ASC_DESC_BRP', 'QUA_INT_1_BRP', 'QUA_INT_MAX_BRP', 'CODTYP_CRT_TRAVAIL_CRI', 'COD_SITU_LOGT_CRI', 'COD_SIT_FAM_EMPRUNTEUR_CRI', 'COD_TYPE_MARCHE_CRI', 'IND_INCIDENT_BDF_CRI', 'IND_PRIMO_ACCEDT_CRI', 'NBR_ENF_ACHARGE_CRI', 'NBR_OCCUP_CRI', 'TOP_BIEN_FR_CRI', 'TOP_NAT_FR_CRI', 'CSP_RGP_BRP', 'top_locatif', 'top_pret_int_ext', 'top_autre_pret_int', 'top_autre_pret_ext', 'top_pers_seule']

    replace_encoding_by_real_mod(stab_data)

    #replace_encoding_by_real_mod(stab_data)
    f = open("doc/encoding_categorical_variables.json")
    dict_encoding=json.load(f)
    encoding_col_and_cat(stab_data,dict_encoding,list_quali)

    #---> faut faire un .map pour regrouper les modalités selon le dictionnaire
    
    #et s'il faut, on doit drop les variables avec une seule modalité
    
    #Quanti
    discretisation_variables_from_chi2(stab_data)#discrétisation des variables quanti

    return stab_data




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
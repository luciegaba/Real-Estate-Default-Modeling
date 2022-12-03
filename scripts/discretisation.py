import numpy as np
import pandas as pd





""" FONCTION DE DISCRETISATION """

def discretisation_variables_from_chi2(X: pd.DataFrame) -> pd.DataFrame: 
        
    """ Cette fonction permet de discrétiser les variables selon les buckets trouvés initialement
    retourne le dataframe avec les quantis discrétisées
    Argument: 
        - X

     """  


    #X.loc[ X['IRPAR_USAGE_V12_MAX'] <= 2, 'g_IRPAR_USAGE_V12_MAX' ] = 'grp_1' 

    #X.loc[(X['IRPAR_USAGE_V12_MAX'] > 2) & (X['IRPAR_USAGE_V12_MAX'] <= 4), 'g_IRPAR_USAGE_V12_MAX' ] = 'grp_2'

    #X.loc[  (X['IRPAR_USAGE_V12_MAX'] > 4) & (X['IRPAR_USAGE_V12_MAX'] <= 10), 'g_IRPAR_USAGE_V12_MAX' ] = 'grp_3'

    
#################################################################

    X.loc[ (X['MNT_TOT_ASSURANCE_CRI'] < 7683.976) , 'g_MNT_TOT_ASSURANCE_CRI' ] = 'grp_1'

    X.loc[  ((X['MNT_TOT_ASSURANCE_CRI'] > 7683.976) & 
                          (X['MNT_TOT_ASSURANCE_CRI'] <=17283.191)), 'g_MNT_TOT_ASSURANCE_CRI' ] = 'grp_2'

    X.loc[  ((X['MNT_TOT_ASSURANCE_CRI'] > 17283.191) &
                          (X['MNT_TOT_ASSURANCE_CRI'] <=246495.03)), 'g_MNT_TOT_ASSURANCE_CRI' ] = 'grp_3'


#################################################################

    X.loc[(X['COUT_RACHAT_BRP'] >38969.08) & (X['COUT_RACHAT_BRP'] <= 183761.202), 'g_COUT_RACHAT_BRP' ] = 'grp_1'

    X.loc[  ((X['COUT_RACHAT_BRP'] >183761.202) & (X['COUT_RACHAT_BRP'] <= 1826635.0)) 
                       , 'g_COUT_RACHAT_BRP' ] = 'grp_2'

    X.loc[  (X['COUT_RACHAT_BRP'] <= 38969.08)  , 'g_COUT_RACHAT_BRP' ] = 'grp_3'


###################################################################


    X.loc[ (X['SUM_PATR_IMMO_BRP'] > 160000.0) & (X['SUM_PATR_IMMO_BRP'] <= 250000.0), 'g_SUM_PATR_IMMO_BRP' ] = 'grp_1' 

    X['g_SUM_PATR_IMMO_BRP'] = X['g_SUM_PATR_IMMO_BRP'].replace(np.nan, 'grp_2')

###################################################################

    X.loc[ ((X['quotite'] > 0.0282) & (X['quotite'] <=  0.802)) | 
                       ((X['quotite'] > 1.019) & (X['quotite'] <= 1.043))
                       , 'g_quotite' ] = 'grp_1'

    X.loc[  ((X['quotite'] > 1) & (X['quotite'] <= 1.019)), 'g_quotite' ] = 'grp_2'

    X.loc[  ((X['quotite'] > 0.802) & (X['quotite'] <= 0.966)) | 
                       ((X['quotite'] > 1.043) & (X['quotite'] <= 12.006))
                       , 'g_quotite' ] = 'grp_3'

    X.loc[  ((X['quotite'] > 0.966) & (X['quotite'] <=1.0)), 'g_quotite' ] = 'grp_4'

###################################################################
    X.loc[(X['PCT_TEG_TAEG_CRI'] > 0.26 ) & (X['PCT_TEG_TAEG_CRI'] <=  2.077), 'g_PCT_TEG_TAEG_CRI' ] = 'grp_1'

    X.loc[  ((X['PCT_TEG_TAEG_CRI'] > 2.077) & (X['PCT_TEG_TAEG_CRI'] <= 2.38)) | 
                        ((X['PCT_TEG_TAEG_CRI'] > 2.624) & (X['PCT_TEG_TAEG_CRI'] <= 2.84))
                       , 'g_PCT_TEG_TAEG_CRI' ] = 'grp_2'

    X.loc[  ((X['PCT_TEG_TAEG_CRI'] > 2.84) & (X['PCT_TEG_TAEG_CRI'] <= 3.29)) | 
                        ((X['PCT_TEG_TAEG_CRI'] > 2.38) & (X['PCT_TEG_TAEG_CRI'] <= 2.624))
                       , 'g_PCT_TEG_TAEG_CRI' ] = 'grp_3'


    X.loc[  (X['PCT_TEG_TAEG_CRI'] >= 3.29) , 'g_PCT_TEG_TAEG_CRI' ] = 'grp_4'

###########################t########################################
    X.loc[ ((X['MOY_ANC_PROF_BRP'] > 5)  & ( X['MOY_ANC_PROF_BRP'] <= 11.5)) |
                        (X['MOY_ANC_PROF_BRP'] > 5)  & ( X['MOY_ANC_PROF_BRP'] <= 11.5) | 
                       (X['MOY_ANC_PROF_BRP'] > 16)  & ( X['MOY_ANC_PROF_BRP'] <= 49),
                       'g_MOY_ANC_PROF_BRP' ] = 'grp_1' 

    X.loc[ (X['MOY_ANC_PROF_BRP'] > 3.5 ) & (X['MOY_ANC_PROF_BRP'] <=  5 ) | 
                       (X['MOY_ANC_PROF_BRP'] > 11.5 ) & (X['MOY_ANC_PROF_BRP'] <=  16 )
                       , 'g_MOY_ANC_PROF_BRP' ] = 'grp_2' 

    X.loc[ (X['MOY_ANC_PROF_BRP'] <= 3.5 )   , 'g_MOY_ANC_PROF_BRP' ] = 'grp_3'


###########################t########################################
    X.loc[ (X['nb_pret'] > 1) & (X['nb_pret'] <= 6), 'g_nb_pret' ] = 'grp_1' 

    X.loc[(X['nb_pret'] <=1)  , 'g_nb_pret' ] = 'grp_2'


###########################t########################################
    X.loc[ (X['MNT_COUT_TOT_CREDIT_CRI'] <=36072.915 ) , 'g_MNT_COUT_TOT_CREDIT_CRI' ] = 'grp_1' 

    X.loc[ (X['MNT_COUT_TOT_CREDIT_CRI'] > 36072.915) &  (X['MNT_COUT_TOT_CREDIT_CRI'] <=59387.94 ) , 'g_MNT_COUT_TOT_CREDIT_CRI' ] = 'grp_2' 

    X.loc[ (X['MNT_COUT_TOT_CREDIT_CRI'] > 59387.94) &  (X['MNT_COUT_TOT_CREDIT_CRI'] <= 1968134.71) , 'g_MNT_COUT_TOT_CREDIT_CRI' ] = 'grp_3' 


###########################t########################################
    X.loc[ (X['BEST_APPORT_TX_BRP'] > 29.74) & (X['BEST_APPORT_TX_BRP'] <=97.6 ) , 'g_BEST_APPORT_TX_BRP' ] = 'grp_1' 

    X.loc[ ((X['BEST_APPORT_TX_BRP'] > 6.81) & (X['BEST_APPORT_TX_BRP'] <=13.15 )) |
                       ((X['BEST_APPORT_TX_BRP'] > 16.07) & (X['BEST_APPORT_TX_BRP'] <= 29.74 ))
                        , 'g_BEST_APPORT_TX_BRP' ] = 'grp_2' 

    X.loc[ ((X['BEST_APPORT_TX_BRP'] > 13.15) & (X['BEST_APPORT_TX_BRP'] <=16.07 )) |
                       ((X['BEST_APPORT_TX_BRP'] >1.8) & (X['BEST_APPORT_TX_BRP'] <= 6.81))
                        , 'g_BEST_APPORT_TX_BRP' ] = 'grp_3' 

    X.loc[ (X['BEST_APPORT_TX_BRP'] <= 1.8)  , 'g_BEST_APPORT_TX_BRP' ] = 'grp_4' 

###########################t########################################
    X.loc[ (X['COUT_PROJET_HF_AT_BRP'] > 82262.854) & (X['COUT_PROJET_HF_AT_BRP'] <= 112500.0 ) , 'g_COUT_PROJET_HF_AT_BRP' ] = 'grp_1' 

    X.loc[ ((X['COUT_PROJET_HF_AT_BRP'] > 844.999) & (X['COUT_PROJET_HF_AT_BRP'] <= 82262.854 )) | 
                       ((X['COUT_PROJET_HF_AT_BRP'] > 112500) & (X['COUT_PROJET_HF_AT_BRP'] <=298000 )) 
                        , 'g_COUT_PROJET_HF_AT_BRP' ] = 'grp_2' 


    X.loc[ (X['COUT_PROJET_HF_AT_BRP'] > 298000) & (X['COUT_PROJET_HF_AT_BRP'] <= 4600000 ) , 'g_COUT_PROJET_HF_AT_BRP' ] = 'grp_3' 
    
    
    
############################################

    X.loc[(X['SUM_MTENCBIE_IMMO_BRP'] > 84400.25) &
                       (X['SUM_MTENCBIE_IMMO_BRP'] <= 193958.604), 'g_SUM_MTENCBIE_IMMO_BRP' ] = 'grp_1'

    X.loc[(X['SUM_MTENCBIE_IMMO_BRP'] > 193958.604) &
                       (X['SUM_MTENCBIE_IMMO_BRP'] <= 252249632.0), 'g_SUM_MTENCBIE_IMMO_BRP' ] = 'grp_2'

    X.loc[(X['SUM_MTENCBIE_IMMO_BRP'] < 84400.25) , 'g_SUM_MTENCBIE_IMMO_BRP' ] = 'grp_3'
    
    
    

##############################################
    X.loc[ ((X['MNT_PRET_CRI'] > 79000.0 ) & (X['MNT_PRET_CRI'] <= 131632.269 )), 'g_MNT_PRET_CRI' ] = 'grp_1'

    X.loc[ ((X['MNT_PRET_CRI'] > 271978.55 ) & (X['MNT_PRET_CRI'] <= 4600000.0 )), 'g_MNT_PRET_CRI' ] = 'grp_3'

    X['g_MNT_PRET_CRI'] = X['g_MNT_PRET_CRI'].replace(np.nan , 'grp_2')

##############################################


    X.loc[(X['AGE_INT_MAX_BRP'] <= 44) | (X['AGE_INT_MAX_BRP'] > 90 ), 'g_AGE_INT_MAX_BRP' ] = 'grp_1' 

    X.loc[(X['AGE_INT_MAX_BRP'] > 44) & (X['AGE_INT_MAX_BRP'] <= 90 ), 'g_AGE_INT_MAX_BRP' ] = 'grp_2' 




    return X




def check_mean_defaut_rate_per_category(X_train:pd.DataFrame,y:pd.Series,type="quanti")-> None:
    
    """ Cette fonction permet d'afficher le taux moyen de defaut pour chaque variable ( pour vérifier si les modalités crées sont pertinentes 
    Argument: 
        -X_train
        -y
        -type: quanti ou quali

    """
    
    columns_to_check = []
    columns = X_train.columns.tolist() 
    if type == "quanti":
        for col in columns :  # on prend que les variables qui commencent par g_ car c'est les variables discrétiser 
            if 'g_' in col : 
                columns_to_check.append(col) 
    else: 
        columns_to_check=columns
    for var in columns_to_check : 
        defaut = pd.crosstab(X_train[var],y)[1]
        print(f'------ pour la variable {var} : ' )
        print()
        display(defaut)
        print()
        




""" FONCTION INITIALEMENT UTILISE POUR CREER LES BUCKETS"""


""""
def Chi_2_discretisation(X:pd.DataFrame,y:pd.Series, var_to_bin:str, n_cut=8) :

    Fonction qui permet de discrétiser selon le critere du Chi 2 (elle créée des groupes qui doivent obligatoirement être significatif )
    retourne le dataframe avec buckets pour les variables quantitatives préalablement sélectionnées, ainsi que les valeurs de la variable initiale dans chaque bucket 
    / Elle affiche aussi les taux de défaut moyen par bucket
    Argument:
        -X
        -y
        -var_to_bin


    df= X.copy()
    df['defaut_36mois'] = y 
    raw_data = df[[ var_to_bin , 'defaut_36mois' ]]

    # création des cuts
    raw_data["intervalle_cut"] = pd.qcut(raw_data[var_to_bin], q = n_cut ,duplicates="drop")
    print('nb de cuts:',raw_data["intervalle_cut"].nunique())

    #groupement des cuts et moyenne de taux de defaut
    df_to_group = raw_data.groupby("intervalle_cut")['defaut_36mois'].agg(["mean"]).reset_index()
    print('cut regrouper avec taux de defaut moyen :')
    print()
    print(df_to_group.sort_values(by='mean'))
    print()

    # raw_data['intervalle_cut'] = raw_data['intervalle_cut'].astype('str')  # important

    # ############################## création du dictionnaire

    # # creation de dict
    # dico = dict()
    # for inter, dx in zip( df_a_grouper["intervalle_cut"].astype(str) , df_a_grouper["mean"].round(2) ) :
    #     dico[inter] = dx

    # liste_de_valeurs = list(dico.values())

    # from pprint import pprint
    # dic = dict()

    # # création des clés qui seront le taux moyen de defaut de chaque cut
    # for c in set([ round(i,2) for i in liste_de_valeurs ]):
    #     dic[c] = list()

    # # inverser le dictionnaire
    # # création du couple clé valeurs : clé = tx moyen defaut , valeur : liste des intervalle
    # for k,v in dico.items() :
    #     if round(v, 2) in dic :
    #         dic[round(v,2)].append(k)
    #     else :
    #         pass

    # print('dictionnaire intiale avec les différents taux de défaut :')
    # print()
    # pprint(dic)
    # print()

    # # création des groupes dans la variables basé sur les tx moyen de defaut
    # for k, v in dic.items() :
    #     for lst in v:
    #         lst = str(lst)
    #         raw_data.loc[raw_data['intervalle_cut'].isin([lst]) , f'groupe_{var_a_discretiser}'  ] = f'grp_{round(k*100)}'

    # print('moyenne par bucket crée :')
    # print()
    # print(raw_data.groupby(f'groupe_{var_a_discretiser}').mean()[target])
    # print()

    # ################################## test de la p-value

    # def khi_test(data, var1:str,var2:str)  :

    #     CrossTabResult=pd.crosstab(index=data[var1], columns=data[var2])
    #     ChiSqResult = chi2_contingency(CrossTabResult)

    #     if (ChiSqResult[1] < 0.05):
    #         return (round(ChiSqResult[1],6))
    # # si non significative, elle retourne None

   

    # print('P-Value total : sur la variable du test =' , khi_test(raw_data, target, f'groupe_{var_a_discretiser}' ) )
    # print()
    # print('------------------- a toi de jouer maintenant et de regouper les buckets pertinentes ------------')

    #return raw_data.drop_duplicates(subset=['intervalle_cut' ,f'groupe_{var_a_discretiser}'])



def group_by_value_counts(X_train:pd.DataFrame, var:str) -> None : 
    
    print('Effectif de chaque groupe') 
    print(X_train[var].value_counts(normalize = True )*100)
    print()
    print('Moyenne par bucket créé') 
    print(X_train.groupby(var).mean()['defaut_36mois']*100)

"""



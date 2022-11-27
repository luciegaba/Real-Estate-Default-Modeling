import numpy as np
from scipy.stats import chi2_contingency
import pandas as pd

# pour s'assurer du bon découpage des variable quanti 
def group_by_et_value_counts(X, var) : 
    
    print('value counts de chaque groupe') 
    print(X[var].value_counts(normalize = True )*100)
    
    print()
    print('moyenne par bucket crée') 
    print(X.groupby(var).mean()['defaut_36mois']*100)


# fonction qui permet de discrétiser selon le critere du khi 2 ( crée des groupes qui doivement obligatoirement etre significatif )
def khi_2_discretisation(df, var_a_discretiser:str, target:str, n_cut=8) :

    """ prend un data frame et retourne un data frame avec
    des buckets pour la variable quantitatives et les valeurs de la variable intiale dans chaque buckets
    affiche aussi des p_values du test du KHI-2 associé a chaque buckets
    """

    raw_data = df[[ var_a_discretiser , target ]]

    # création des cuts
    raw_data["intervalle_cut"] = pd.qcut(raw_data[var_a_discretiser], q = n_cut ,duplicates="drop")
    print('nb de cuts:',raw_data["intervalle_cut"].nunique())

    #groupement des cuts et moyenne de taux de defaut
    df_a_grouper = raw_data.groupby("intervalle_cut")[target].agg(["mean"]).reset_index()
    print('cut regrouper avec taux de defaut moyen :')
    print()
    print(df_a_grouper.sort_values(by='mean'))
    print()

    raw_data['intervalle_cut'] = raw_data['intervalle_cut'].astype('str')  # important

    ############################## création du dictionnaire

    # creation de dict
    dico = dict()
    for inter, dx in zip( df_a_grouper["intervalle_cut"].astype(str) , df_a_grouper["mean"].round(2) ) :
        dico[inter] = dx

    liste_de_valeurs = list(dico.values())

    from pprint import pprint
    dic = dict()

    # création des clés qui seront le taux moyen de defaut de chaque cut
    for c in set([ round(i,2) for i in liste_de_valeurs ]):
        dic[c] = list()

    # inverser le dictionnaire
    # création du couple clé valeurs : clé = tx moyen defaut , valeur : liste des intervalle
    for k,v in dico.items() :
        if round(v, 2) in dic :
            dic[round(v,2)].append(k)
        else :
            pass

    print('dictionnaire intiale avec les différents taux de défaut :')
    print()
    pprint(dic)
    print()

    # création des groupes dans la variables basé sur les tx moyen de defaut
    for k, v in dic.items() :
        for lst in v:
            lst = str(lst)
            raw_data.loc[raw_data['intervalle_cut'].isin([lst]) , f'groupe_{var_a_discretiser}'  ] = f'grp_{round(k*100)}'

    print('moyenne par bucket crée :')
    print()
    pprint(raw_data.groupby(f'groupe_{var_a_discretiser}').mean()[target])
    print()

    ################################## test de la p-value

    def khi_test(data, var1:str,var2:str)  :

        CrossTabResult=pd.crosstab(index=data[var1], columns=data[var2])
        ChiSqResult = chi2_contingency(CrossTabResult)

        if (ChiSqResult[1] < 0.05):
            return (round(ChiSqResult[1],6))
    # si non significative, elle retourne None

   

    print('P-Value total : sur la variable du test =' , khi_test(raw_data, target, f'groupe_{var_a_discretiser}' ) )
    print()
    print('------------------- a toi de jouer maintenant et de regouper les buckets pertinentes ------------')

    #return raw_data.drop_duplicates(subset=['intervalle_cut' ,f'groupe_{var_a_discretiser}'])







# fonction qui execute la discrétisation choisit grace a la fonction khi 2 




def discretisation_variables_from_chi2(X_train_quanti) : 
        
    """ cette fonction permet de discrétiser les variables """  

#################################################################

    X_train_quanti.loc[ (X_train_quanti['MNT_TOT_ASSURANCE_CRI'] < 7683.976) , 'g_MNT_TOT_ASSURANCE_CRI' ] = 'grp_1'

    X_train_quanti.loc[  ((X_train_quanti['MNT_TOT_ASSURANCE_CRI'] > 7683.976) & 
                          (X_train_quanti['MNT_TOT_ASSURANCE_CRI'] <=17283.191)), 'g_MNT_TOT_ASSURANCE_CRI' ] = 'grp_2'

    X_train_quanti.loc[  ((X_train_quanti['MNT_TOT_ASSURANCE_CRI'] > 17283.191) &
                          (X_train_quanti['MNT_TOT_ASSURANCE_CRI'] <=246495.03)), 'g_MNT_TOT_ASSURANCE_CRI' ] = 'grp_3'


#################################################################

    X_train_quanti.loc[(X_train_quanti['COUT_RACHAT_BRP'] >38969.08) & (X_train_quanti['COUT_RACHAT_BRP'] <= 183761.202), 'g_COUT_RACHAT_BRP' ] = 'grp_1'

    X_train_quanti.loc[  ((X_train_quanti['COUT_RACHAT_BRP'] >183761.202) & (X_train_quanti['COUT_RACHAT_BRP'] <= 1826635.0)) 
                       , 'g_COUT_RACHAT_BRP' ] = 'grp_2'

    X_train_quanti.loc[  (X_train_quanti['COUT_RACHAT_BRP'] <= 38969.08)  , 'g_COUT_RACHAT_BRP' ] = 'grp_3'


###################################################################


    X_train_quanti.loc[ (X_train_quanti['SUM_PATR_IMMO_BRP'] > 160000.0) & (X_train_quanti['SUM_PATR_IMMO_BRP'] <= 250000.0), 'g_SUM_PATR_IMMO_BRP' ] = 'grp_1' 

    X_train_quanti['g_SUM_PATR_IMMO_BRP'] = X_train_quanti['g_SUM_PATR_IMMO_BRP'].replace(np.nan, 'grp_2')

###################################################################

    X_train_quanti.loc[ ((X_train_quanti['quotite'] > 0.0282) & (X_train_quanti['quotite'] <=  0.802)) | 
                       ((X_train_quanti['quotite'] > 1.019) & (X_train_quanti['quotite'] <= 1.043))
                       , 'g_quotite' ] = 'grp_1'

    X_train_quanti.loc[  ((X_train_quanti['quotite'] > 1) & (X_train_quanti['quotite'] <= 1.019)), 'g_quotite' ] = 'grp_2'

    X_train_quanti.loc[  ((X_train_quanti['quotite'] > 0.802) & (X_train_quanti['quotite'] <= 0.966)) | 
                       ((X_train_quanti['quotite'] > 1.043) & (X_train_quanti['quotite'] <= 12.006))
                       , 'g_quotite' ] = 'grp_3'

    X_train_quanti.loc[  ((X_train_quanti['quotite'] > 0.966) & (X_train_quanti['quotite'] <=1.0)), 'g_quotite' ] = 'grp_4'

###################################################################
    X_train_quanti.loc[(X_train_quanti['PCT_TEG_TAEG_CRI'] > 0.26 ) & (X_train_quanti['PCT_TEG_TAEG_CRI'] <=  2.077), 'g_PCT_TEG_TAEG_CRI' ] = 'grp_1'

    X_train_quanti.loc[  ((X_train_quanti['PCT_TEG_TAEG_CRI'] > 2.077) & (X_train_quanti['PCT_TEG_TAEG_CRI'] <= 2.38)) | 
                        ((X_train_quanti['PCT_TEG_TAEG_CRI'] > 2.624) & (X_train_quanti['PCT_TEG_TAEG_CRI'] <= 2.84))
                       , 'g_PCT_TEG_TAEG_CRI' ] = 'grp_2'

    X_train_quanti.loc[  ((X_train_quanti['PCT_TEG_TAEG_CRI'] > 2.84) & (X_train_quanti['PCT_TEG_TAEG_CRI'] <= 3.29)) | 
                        ((X_train_quanti['PCT_TEG_TAEG_CRI'] > 2.38) & (X_train_quanti['PCT_TEG_TAEG_CRI'] <= 2.624))
                       , 'g_PCT_TEG_TAEG_CRI' ] = 'grp_3'


    X_train_quanti.loc[  (X_train_quanti['PCT_TEG_TAEG_CRI'] >= 3.29) , 'g_PCT_TEG_TAEG_CRI' ] = 'grp_4'

###########################t########################################
    X_train_quanti.loc[ ((X_train_quanti['MOY_ANC_PROF_BRP'] > 5)  & ( X_train_quanti['MOY_ANC_PROF_BRP'] <= 11.5)) |
                        (X_train_quanti['MOY_ANC_PROF_BRP'] > 5)  & ( X_train_quanti['MOY_ANC_PROF_BRP'] <= 11.5) | 
                       (X_train_quanti['MOY_ANC_PROF_BRP'] > 16)  & ( X_train_quanti['MOY_ANC_PROF_BRP'] <= 49),
                       'g_MOY_ANC_PROF_BRP' ] = 'grp_1' 

    X_train_quanti.loc[ (X_train_quanti['MOY_ANC_PROF_BRP'] > 3.5 ) & (X_train_quanti['MOY_ANC_PROF_BRP'] <=  5 ) | 
                       (X_train_quanti['MOY_ANC_PROF_BRP'] > 11.5 ) & (X_train_quanti['MOY_ANC_PROF_BRP'] <=  16 )
                       , 'g_MOY_ANC_PROF_BRP' ] = 'grp_2' 

    X_train_quanti.loc[ (X_train_quanti['MOY_ANC_PROF_BRP'] <= 3.5 )   , 'g_MOY_ANC_PROF_BRP' ] = 'grp_3'


###########################t########################################
    X_train_quanti.loc[ (X_train_quanti['nb_pret'] > 1) & (X_train_quanti['nb_pret'] <= 6), 'g_nb_pret' ] = 'grp_1' 

    X_train_quanti.loc[(X_train_quanti['nb_pret'] <=1)  , 'g_nb_pret' ] = 'grp_2'


###########################t########################################
    X_train_quanti.loc[ (X_train_quanti['MNT_COUT_TOT_CREDIT_CRI'] <=36072.915 ) , 'g_MNT_COUT_TOT_CREDIT_CRI' ] = 'grp_1' 

    X_train_quanti.loc[ (X_train_quanti['MNT_COUT_TOT_CREDIT_CRI'] > 36072.915) &  (X_train_quanti['MNT_COUT_TOT_CREDIT_CRI'] <=59387.94 ) , 'g_MNT_COUT_TOT_CREDIT_CRI' ] = 'grp_2' 

    X_train_quanti.loc[ (X_train_quanti['MNT_COUT_TOT_CREDIT_CRI'] > 59387.94) &  (X_train_quanti['MNT_COUT_TOT_CREDIT_CRI'] <= 1968134.71) , 'g_MNT_COUT_TOT_CREDIT_CRI' ] = 'grp_3' 


###########################t########################################
    X_train_quanti.loc[ (X_train_quanti['BEST_APPORT_TX_BRP'] > 29.74) & (X_train_quanti['BEST_APPORT_TX_BRP'] <=97.6 ) , 'g_BEST_APPORT_TX_BRP' ] = 'grp_1' 

    X_train_quanti.loc[ ((X_train_quanti['BEST_APPORT_TX_BRP'] > 6.81) & (X_train_quanti['BEST_APPORT_TX_BRP'] <=13.15 )) |
                       ((X_train_quanti['BEST_APPORT_TX_BRP'] > 16.07) & (X_train_quanti['BEST_APPORT_TX_BRP'] <= 29.74 ))
                        , 'g_BEST_APPORT_TX_BRP' ] = 'grp_2' 

    X_train_quanti.loc[ ((X_train_quanti['BEST_APPORT_TX_BRP'] > 13.15) & (X_train_quanti['BEST_APPORT_TX_BRP'] <=16.07 )) |
                       ((X_train_quanti['BEST_APPORT_TX_BRP'] >1.8) & (X_train_quanti['BEST_APPORT_TX_BRP'] <= 6.81))
                        , 'g_BEST_APPORT_TX_BRP' ] = 'grp_3' 

    X_train_quanti.loc[ (X_train_quanti['BEST_APPORT_TX_BRP'] <= 1.8)  , 'g_BEST_APPORT_TX_BRP' ] = 'grp_4' 

###########################t########################################
    X_train_quanti.loc[ (X_train_quanti['COUT_PROJET_HF_AT_BRP'] > 82262.854) & (X_train_quanti['COUT_PROJET_HF_AT_BRP'] <= 112500.0 ) , 'g_COUT_PROJET_HF_AT_BRP' ] = 'grp_1' 

    X_train_quanti.loc[ ((X_train_quanti['COUT_PROJET_HF_AT_BRP'] > 844.999) & (X_train_quanti['COUT_PROJET_HF_AT_BRP'] <= 82262.854 )) | 
                       ((X_train_quanti['COUT_PROJET_HF_AT_BRP'] > 112500) & (X_train_quanti['COUT_PROJET_HF_AT_BRP'] <=298000 )) 
                        , 'g_COUT_PROJET_HF_AT_BRP' ] = 'grp_2' 


    X_train_quanti.loc[ (X_train_quanti['COUT_PROJET_HF_AT_BRP'] > 298000) & (X_train_quanti['COUT_PROJET_HF_AT_BRP'] <= 4600000 ) , 'g_COUT_PROJET_HF_AT_BRP' ] = 'grp_3' 
    
    
    
############################################

    X_train_quanti.loc[(X_train_quanti['SUM_MTENCBIE_IMMO_BRP'] > 84400.25) &
                       (X_train_quanti['SUM_MTENCBIE_IMMO_BRP'] <= 193958.604), 'g_SUM_MTENCBIE_IMMO_BRP' ] = 'grp_1'

    X_train_quanti.loc[(X_train_quanti['SUM_MTENCBIE_IMMO_BRP'] > 193958.604) &
                       (X_train_quanti['SUM_MTENCBIE_IMMO_BRP'] <= 252249632.0), 'g_SUM_MTENCBIE_IMMO_BRP' ] = 'grp_2'

    X_train_quanti.loc[(X_train_quanti['SUM_MTENCBIE_IMMO_BRP'] < 84400.25) , 'g_SUM_MTENCBIE_IMMO_BRP' ] = 'grp_3'
    
    
    

##############################################
    X_train_quanti.loc[ ((X_train_quanti['MNT_PRET_CRI'] > 79000.0 ) & (X_train_quanti['MNT_PRET_CRI'] <= 131632.269 )), 'g_MNT_PRET_CRI' ] = 'grp_1'

    X_train_quanti.loc[ ((X_train_quanti['MNT_PRET_CRI'] > 271978.55 ) & (X_train_quanti['MNT_PRET_CRI'] <= 4600000.0 )), 'g_MNT_PRET_CRI' ] = 'grp_3'

    X_train_quanti['g_MNT_PRET_CRI'] = X_train_quanti['g_MNT_PRET_CRI'].replace(np.nan , 'grp_2')

##############################################


    X_train_quanti.loc[(X_train_quanti['AGE_INT_MAX_BRP'] <= 44) | (X_train_quanti['AGE_INT_MAX_BRP'] > 90 ), 'g_AGE_INT_MAX_BRP' ] = 'grp_1' 

    X_train_quanti.loc[(X_train_quanti['AGE_INT_MAX_BRP'] > 44) & (X_train_quanti['AGE_INT_MAX_BRP'] <= 90 ), 'g_AGE_INT_MAX_BRP' ] = 'grp_2' 




    return X_train_quanti



# verification par group by 

def verification_par_moyenne_defaut(x,y,type="quanti"):
    
    """ cette fonction permet d'afficher le taux moyen de defaut pour chaque var ( verifier si les modalités crées sont pertinentes"""
    
    colonnes_a_verifier = []
    colonnes = x.columns.tolist() 
    x=pd.concat([x.reset_index(),y.reset_index()],axis=1)
    if type == "quanti":
        for col in colonnes :  # on prend que les variables qui commencent par g_ car c'est les variables discrétiser 
            if 'g_' in col : 
                colonnes_a_verifier.append(col) 
    else: 
        colonnes_a_verifier=colonnes
    for var in colonnes_a_verifier : 
        defaut = x.groupby(var).mean()['defaut_36mois']
        print(f'------ pour la variable {var} : ' )
        print()
        print(defaut) 
        print()
        

# recuperer seulemnt les variables discrétiser et supprimer les variables quanti de bases 
def get_binned_df(X) : 
    
    col_a_garder = []
    
    for col in X.columns.tolist() : 
        if 'g_' in col : 
            col_a_garder.append(col)
            
    
    X_quanti_final = X[col_a_garder]
    
    return X_quanti_final


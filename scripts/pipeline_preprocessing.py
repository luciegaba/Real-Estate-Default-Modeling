
import pandas as pd
import json

from scripts.processing import convert_modalities_to_quali,na_sup_20_fill_by_0_vs_autres,fill_by_0,imputation_for_na,drop_columns
from scripts.discretisation import  discretisation_variables_from_chi2
from scripts.group_cat_variables import replace_encoding_by_real_labels,encoding_categorical_variables,group_modalities_with_optbinning


def pipeline_processing(X: pd.DataFrame,y:pd.Series,quanti_selected:list, quali_selected:list, finally_selected_variables:list,stability_sample=True, training_option=False ) -> pd.DataFrame:

    """ retourne le dataframe à utiliser pour régression logistique, ML, et stabilité temporelle 
    Arguments:
        - X : peut être x_train, x_test, n'importe quel échantillon qui à l'origine dispose des mêmes variables finales (facilement adaptable pour un nouveau jeu de données)
        - y : pour réentrainement donc optionnel 
        - quanti_selected: Variables qualitatives sélectionnées pour la partie ML
        - quali_selected: Variables quantitatives sélectionnées pour la partie ML 
        - finally_selected_variables: Variables finalement choisies post rf (avec g_)
        - training_option: si il s'agit d'un échantillon sur lequel on veut apprendre (encoding doit se baser dessus)
        Rq: certaines variables sont suceptibles de ne pas être stable (il faudra les supprimer des variables sélectionnées)

    """
    selected_variables_for_process=quanti_selected+ quali_selected
    returning_variables=finally_selected_variables.copy()
    selected_variables_for_process.append('date_debloc_avec_crd')
    returning_variables.append(y.name)

    """ ON CONSERVE UNIQUEMENT LES VARIABLES FINALES """

    X=pd.concat([X[selected_variables_for_process],y],axis=1)

    """ SI STABILITE TEMPORELLE: TRAITEMENT DE VARIABLE DATE"""
    X["date_debloc_avec_crd"]=pd.to_datetime(X["date_debloc_avec_crd"],format="%Y%m")


    """  IMPUTATION DES VALEURS MANQUANTES """

    print("Warning : Imputing missing values ...")
    variables_a_fill_0,variables_a_fill_standard= na_sup_20_fill_by_0_vs_autres(quanti_selected)
    fill_by_0(X[quanti_selected+quali_selected], variables_a_fill_0)
    imputation_for_na(X[quanti_selected+quali_selected])
    print("DONE")


    """  CONVERSION DES MODALITES """

    print("Warning : Converting modalities to 'object' type ...")
    convert_modalities_to_quali(X, quali_selected)

    print("DONE")

    print("Warning : Encoding categorical variables ...")
    replace_encoding_by_real_labels(X)

    if training_option==True:
        dict_encoding=group_modalities_with_optbinning(X,y,quali_selected,0.05,0.001,encoding=True,display_=False)

    else:
        f = open("doc/encoding_categorical_variables.json")
        dict_encoding=json.load(f)

    encoding_categorical_variables(X,dict_encoding,quali_selected)
    print("DONE")

    

    """  DISCRETISATION VARIABLES QUANTITATIVES """

    print("Warning : Binning continuous variables ...")
    discretisation_variables_from_chi2(X)
    drop_columns(X,quanti_selected)
    print("DONE")

    if stability_sample==True:
        returning_variables.append("date_debloc_avec_crd")  

    """ AJUSTEMENT FINALE: SUPRESSION DES VARIABLES RETENUES PAR RF (liste exhaustive)"""

    X=X[returning_variables]

    return X

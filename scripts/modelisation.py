
import sklearn
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import confusion_matrix,roc_auc_score,roc_curve
import matplotlib.pyplot as plt
import pandas as pd



""" PREPARER LES DONNEES A LA MODELISATION """


def prepare_data_for_ml(data:pd.DataFrame,col_selected:list):
    """ Retourne x et y séparés et prêts pour ml
    Arguments: 
    - data : x et y 
    -col_selected : variables sélectionnées post-rf
    """
    y = data['defaut_36mois'].astype(int)
    x = data[col_selected].astype(str)
    x = pd.get_dummies(x, drop_first=True)
    return x,y


def verif_nb_colonnes(x_train:pd.DataFrame,x_other:pd.DataFrame)-> pd.DataFrame : 
    """ Retourne dataframe disposant des mêmes colonnes que les features utilisés pour l'entraînement du modèle (x_train)
    Arguments: 
    - x_train: features finaux
    - x_other: dataframe à check et éventuellement corriger si colonnes différentes
    """

    # Get missing columns in the training test
    missing_cols = set( x_train.columns ) - set( x_other.columns )
    # Add a missing column in test set with default value equal to 0
    for c in missing_cols:
        x_other[c] = 0
    # Ensure the order of column in the test set is in the same order than in train set
    x_other = x_other[x_train.columns]
    return x_other




def fitting_model(model, x_train:pd.DataFrame, y_train:pd.Series) : 
    """ Retourne modèle fitté
    - x_train
    - y_train
    """
    model.fit(x_train , y_train)
    return model







def gridsearchcv_for_model(model,x_train:pd.DataFrame,y_train:pd.Series,params:dict,train_mod_with_best_params=False,scoring='recall',split_for_kfold=4):
    """"
    Retourne les paramètres optimaux ou le modèle directement entraîné avec si souhaité 
    - model
    - x_train
    - x_test
    - params :grid
    - train_mod_with_best_params: True ==> on traine le modèle direct avec les paramètres optimaux de la gridsearchcv
    - scoring: métrique à retenir (par défaut recall)
    - split_for_kflod:  Nombre de split pour KFold

    """"
    kfold = StratifiedKFold( split_for_kfold = split_for_kfold) 
    grid = GridSearchCV(model, params, cv=kfold ,scoring = scoring) 
    grid.fit(x_train , y_train)
    if train_mod_with_best_params==True:
        return grid,grid.best_params_
    else:
        return grid.best_params_






def evaluation(model,x_train:pd.DataFrame,x_test:pd.DataFrame,y_train:pd.DataFrame,y_test:pd.DataFrame)-> None:
    """"
    Retourne un ensemble de métriques et graphiques pour évaluer la performance du modèle
    Arguments: 
    - model
    - x_train
    - x_test
    - y_train
    - y_test

    """"

    y_train_pred=model.predict_proba(x_train)
    y_train_pred = pd.DataFrame(y_train_pred).iloc[: , 1]
    y_test_pred=model.predict_proba(x_test)
    y_test_pred = pd.DataFrame(y_test_pred).iloc[: , 1]
    score_auc_train = roc_auc_score(y_train,y_train_pred )
    score_auc_test =  roc_auc_score(y_test,y_test_pred)
    y_pred_class=model.predict(x_test)
    y_pred_class_train=model.predict(x_train)
    display(confusion_matrix(y_train,y_pred_class_train))
    display(confusion_matrix(y_test,y_pred_class))
    print('AUC TRAIN :' ,score_auc_train.round(2)) 
    print('AUC TEST :' ,score_auc_test.round(2)) 
    fpr_train, tpr_train, _ = roc_curve(y_train, y_train_pred , pos_label = 1 )
    fpr_test, tpr_test, _ = roc_curve(y_test, y_test_pred , pos_label = 1 )
    plt.plot(fpr_train, tpr_train, label = f'auc_train {score_auc_train.round(2)}') 
    plt.plot(fpr_test, tpr_test, label = f'auc_test {score_auc_test.round(2)}') 
    plt.plot([0,1], [0,1])
    plt.title('auc')
    plt.xlabel('Taux faux positifs')
    plt.ylabel('Taux vrais positifs')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.legend()
    plt.figure(figsize= (9,9))
    plt.figure(figsize= (9,9))




def get_coeff_for_model(logistic_model, x_train:pd.DataFrame)-> pd.DataFrame:
        """
        
        """
        coeff = pd.DataFrame( abs(logistic_model.coef_.T), index = x_train.columns, columns = ['coef']).reset_index()
        coeff = coeff.sort_values(by = 'index') 
        return coeff




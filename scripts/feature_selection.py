
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.stats import pointbiserialr,kruskal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns



"""    TESTS UNIVARIES VARIABLES QUANTITATIVES    """


def test_kruskall_wallis(X_train: pd.DataFrame,y: pd.Series) -> pd.DataFrame :

    """ retourne un DataFrame avec les statistiques du test Kruskall Wallis et les pvalues associées

    Argument: 
    - X_train: X_train comportant les variables quantitatives 
    - y: y_train

    """    
    var_quanti=X_train.select_dtypes(exclude="object")
    df_point_bis = pd.DataFrame(columns=['variable', 'corr', 'pvalue'])    
    for i in var_quanti :
        pbc = kruskal(y, var_quanti[i])
        df_point_bis.loc[len(df_point_bis)] = i, round(pbc.statistic, 3) , round(pbc.pvalue, 3)
    return df_point_bis



def test_biserial(X_train: pd.DataFrame,y: pd.Series) -> pd.DataFrame :

    """ retourne un DataFrame avec les coefficients du test biserial et les pvalues
    Argument: 
    - X_train: X_train comportant les variables quantitatives 
    - y: y_train

    """    
    var_quanti=X_train.select_dtypes(exclude="object")
    df_point_bis = pd.DataFrame(columns=['variable', 'corr', 'pvalue'])    
    for i in var_quanti :
        pbc = pointbiserialr(y, var_quanti[i])
        df_point_bis.loc[len(df_point_bis)] = i, round(pbc.correlation, 3) , round(pbc.pvalue, 3)
    return df_point_bis


"""    TESTS UNIVARIES VARIABLES QUALITATIVES    """


def test_chi2_independance(X_train: pd.DataFrame,y: pd.Series) -> pd.DataFrame:

    """ retourne un DataFrame avec les coefficients du Chi2 et les p values associées
    Argument: 
    - X_train: X_train comportant les variables qualitatives préselectionnées avec le test du Chi2 
    - y: y_train

    """
    
    var, corr, pval = [], [], []
    df_chi2 = pd.DataFrame(columns = ['variable', 'Chi2', 'pvalue'])
    for variable in X_train.select_dtypes(include="object").columns.tolist():
        crosstab = pd.crosstab(X_train[variable], y)
        chi2, pval, dof, expected = ss.chi2_contingency(crosstab)
        df_chi2.loc[len(df_chi2)] = variable, chi2, round(pval, 3)
    return df_chi2

    

def cramers_v_btw_X(X_train:pd.DataFrame) -> pd.DataFrame:

    """ retourne un DataFrame avec les coefficients du V de Cramer entre variables qualitatives
    Argument: 
    - X_train: X_train comportant les variables qualitatives préselectionnées avec le test du Chi2 r 

    """
    
    categorical_columns = list(X_train.select_dtypes('object').columns)
    categorical_correlation_df = pd.DataFrame(index = categorical_columns, columns = categorical_columns)
    for i in categorical_columns:
        for j in categorical_columns:
            contingency_table = pd.crosstab(X_train[i], X_train[j])
            if contingency_table.size != 0:               
                categorical_correlation_df.loc[i, j] = cramers_v(contingency_table)
    return categorical_correlation_df



def cramers_v_with_target(X_train:pd.DataFrame,y:pd.Series) -> pd.DataFrame:

    """ retourne un DataFrame avec les coefficients du V de Cramer
    Argument: 
    - X_train: X_train comportant les variables qualitatives préselectionnées avec le test du Chi2 
    - y: y_train

    """

    categorical_columns = list(X_train.select_dtypes('object').columns)
    categorical_correlations = pd.DataFrame(columns=["Variable","coefficient"])
    for i in categorical_columns:
        contingency_table = pd.crosstab(X_train[i], y)
        if contingency_table.size != 0:  
            value=   cramers_v(contingency_table)   
            categorical_correlations.loc[len(categorical_correlations)] = i, value
    return categorical_correlations



def cramers_v(contingency_table: pd.DataFrame):
    
    chi2 = ss.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2_coff = chi2/n
    r,c = contingency_table.shape
    unbiased_phi2_coff = max(0, (phi2_coff - (((c-1)*(r-1))/(n-1))))    
    r_corr = r - (((r-1)**2)/(n-1))
    c_corr = c - (((c-1)**2)/(n-1))
    return np.sqrt(unbiased_phi2_coff / min((c_corr-1), (r_corr-1)))



"""    REGRESSION LASSO    """


def selection_avec_lasso(X_train:pd.DataFrame,y:pd.Series,n=20,var_to_fit="all") -> list : # attention au fit_transform lors de l'utilisation de X_test et base_HT
    
    """ retourne une liste des variables les plus importantes
    Argument : 
        - X : X_train initial 
        - y: y_train
        - n: Nombre de variables à sélectionner
        - var_to_fit: "num", "quali", "all"

    """ 

    X_quanti = X_train.select_dtypes(exclude='object')
    X_quali=X_train.select_dtypes(include='object')

    if var_to_fit=='num':
        sc = StandardScaler()
        X_quanti_normalise = sc.fit_transform(X_quanti)
        X_to_fit = pd.DataFrame(X_quanti_normalise, columns = X_quanti.columns)
    
    elif var_to_fit=='quali':
        X_quali_encoded=pd.get_dummies(X_quali,prefix=X_quali.columns.tolist())
        X_to_fit=X_quali_encoded
    
    elif var_to_fit=='all':
        
        sc = StandardScaler()
        X_quanti_normalise = sc.fit_transform(X_quanti)
        X_quanti_normalise=pd.DataFrame(X_quanti_normalise, columns = X_quanti.columns)
        X_quali_encoded=pd.get_dummies(X_quali,prefix=X_quali.columns.tolist())
        X_to_fit=pd.concat([X_quanti_normalise.reset_index(drop=True),X_quali_encoded.reset_index(drop=True)],axis=1)
    
   

    lr =  LogisticRegression(penalty = 'l1', class_weight= 'balanced', solver = 'saga' ,random_state = 42, C =0.01)
    lr.fit(X_to_fit, y)
    
    # récuperation des 20 var les + importantes 
    coef_apres_lasso = pd.DataFrame(lr.coef_.T ,  index = X_to_fit.columns, columns = ['coef']) # a regarder 
    coef_apres_lasso['coef'] = coef_apres_lasso['coef'].abs()

    # selection des n variables avec le coeff le plus important
    display(coef_apres_lasso.sort_values(ascending=False, by='coef')[:n])
    
    var_selection = coef_apres_lasso.sort_values(ascending=False, by='coef')[:n].index.tolist()
    
    return  var_selection




""" SELECTION DES VARIABLES FINALES: RANDOM FOREST """



def get_feature_selection_rf(X_train:pd.DataFrame,y:pd.Series):

    """ retourne un graphique de feature importance du rf
    Argument: 
    - X: X_train comportant les variables qualitatives préselectionnées avec le test du Chi2 
    - y: y_train

    """
    X_train_discretise = pd.get_dummies( X_train, drop_first = False) 
    x_train, x_test, y_train, y_test = train_test_split(X_train_discretise ,y, stratify = y) 

    param_rf = { 
                'n_estimators' : [400] , 
                'max_depth' : [ 3, 5 ] , 
                'min_samples_leaf' : [ 100] ,
                'class_weight' : ['balanced'] }

    kfold = StratifiedKFold( n_splits = 3 ) 

    rf = RandomForestClassifier()
    model = GridSearchCV(rf, param_rf, cv=kfold ,scoring = 'recall') 
    model.fit(x_train , y_train)

    print(model.best_params_)
    plt.figure(figsize = (11,11))
    importances_rf = pd.DataFrame( model.best_estimator_.feature_importances_.T, index = x_train.columns, columns = ['importance'] ).reset_index()
    fig=sns.barplot(importances_rf, x='importance' , y= 'index')
    plt.show()
    return fig


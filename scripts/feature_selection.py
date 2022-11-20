
import pandas as pd
from scipy.stats import pointbiserialr
import numpy as np
import pandas as pd
import scipy.stats as ss
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def test_pointbiserial_all_quanti_variables(X,y):
    var_quanti=X.select_dtypes(exclude="object")
    var, corr, pval = [], [], []
    df_point_bis = pd.DataFrame(columns=['variable', 'corr', 'pvalue'])    
    for i in var_quanti :
        pbc = pointbiserialr(y, var_quanti[i])
        df_point_bis.loc[len(df_point_bis)] = i, round(pbc.correlation, 3) , round(pbc.pvalue, 3)
    return df_point_bis



def test_chi2_all_quali_variables(X,y):
    var, corr, pval = [], [], []
    df_chi2 = pd.DataFrame(columns = ['variable', 'Chi2', 'pvalue'])
    for variable in X.select_dtypes(include="object").columns.tolist():
        crosstab = pd.crosstab(X[variable], y)
        chi2, pval, dof, expected = ss.chi2_contingency(crosstab)
        df_chi2.loc[len(df_chi2)] = variable, chi2, round(pval, 3)
    return df_chi2


def non_significativité_chi2(X,y):
    
    results_chi2=test_chi2_all_quali_variables(X,y)
    colonnes_non_significatives = results_chi2["variable"][results_chi2["pvalue"]<0.05].tolist()
    print("VARIABLES QUALITATIVES NON SIGNIFICATIVES AU SEUIL DE 5%")
    print(colonnes_non_significatives)
    return colonnes_non_significatives
    

def cramers_v(contingency_table):
    if contingency_table.shape[0]==2:
        correct=False
    else:
        correct=True
    chi2 = ss.chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    phi2_coff = chi2/n
    r,c = contingency_table.shape
    unbiased_phi2_coff = max(0, (phi2_coff - (((c-1)*(r-1))/(n-1))))    
    r_corr = r - (((r-1)**2)/(n-1))
    c_corr = c - (((c-1)**2)/(n-1))
    return np.sqrt(unbiased_phi2_coff / min((c_corr-1), (r_corr-1)))

def cramers_v_all_cat_var(X):
    categorical_columns = list(X.select_dtypes('object').columns)
    categorical_correlation_df = pd.DataFrame(index = categorical_columns, columns = categorical_columns)
    for i in categorical_columns:
        for j in categorical_columns:
            contingency_table = pd.crosstab(X[i], X[j])
            if contingency_table.size != 0:                 #to avoid this statement deal with the missing values first
                categorical_correlation_df.loc[i, j] = cramers_v(contingency_table)
    return categorical_correlation_df

# verification de la corrélation entre les variables 
# recuperation des variables quantitatives pertinentes avec un lasso



def selection_avec_lasso(X,y,n=20,var_to_fit="all") : # attention au fit_transform lors de l'utilisation de X_test et base_HT
    
    """ retourne une liste des variables les plus importantes
    Argument : 
        - X : X_train initial 
    """ 
    # selection X_train et y 
    X_quanti = X.select_dtypes(exclude='object')
    X_quali=X.select_dtypes(include='object')

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
        print(X_to_fit)
    
   

    # model lasso
    lr =  LogisticRegression(penalty = 'l1', class_weight= 'balanced', solver = 'saga' ,random_state = 42, C =0.01)
    lr.fit(X_to_fit, y)
    
    # récuperation des 20 var les + importantes 
    coef_apres_lasso = pd.DataFrame(lr.coef_.T ,  index = X_to_fit.columns, columns = ['coef']) # a regarder 
    coef_apres_lasso['coef'] = coef_apres_lasso['coef'].abs()

    # selection des n variables avec le coeff le plus important
    display(coef_apres_lasso.sort_values(ascending=False, by='coef')[:n])
    
    var_selection = coef_apres_lasso.sort_values(ascending=False, by='coef')[:n].index.tolist()
    
    return  var_selection

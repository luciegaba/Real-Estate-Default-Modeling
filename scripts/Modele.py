import pandas as pd 
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings 
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.metrics import roc_curve 
import matplotlib.pyplot as plt 

# y = pd.read_csv(r"y_train.csv")
# X = pd.read_csv(r"x_train.csv") 

def modele_finale(X, y ) : 


        # pre process avant l'utilisation 

        X.drop(columns = ["Unnamed: 0"] , inplace = True) 

        for col in X : 
                X[col] = X[col].astype('category')

        #################################################################################### selection variables random forest #################################################################################

        # X_train_discretise = pd.get_dummies( X, drop_first = False) 

        # x_train, x_test, y_train, y_test = train_test_split(X_train_discretise , y.iloc[: ,1], stratify = y.iloc[: ,1] ) 

        # param_rf = { 
        #              'n_estimators' : [400] , 
        #              'max_depth' : [ 3, 5 ] , 
        #              'min_samples_leaf' : [ 100] ,
        #              'class_weight' : ['balanced']
        # }

        # kfold = StratifiedKFold( n_splits = 3 ) 

        # rf = RandomForestClassifier()
        # model = GridSearchCV(rf, param_rf, cv=kfold ,scoring = 'recall') 
        # model.fit(x_train , y_train)

        # model.best_params_
        # plt.figure(figsize = (11,11))
        # importances_rf = pd.DataFrame( model.best_estimator_.feature_importances_.T, index = x_train.columns, columns = ['importance'] ).reset_index()
        # sns.barplot(importances_rf, x='importance' , y= 'index')

        # variables non pertinentes qui ressortent apres un features importances du Random forest
        X.drop(columns = ['g_AGE_INT_MAX_BRP', 'g_MNT_PRET_CRI', 'g_SUM_MTENCBIE_IMMO_BRP', 'g_COUT_PROJET_HF_AT_BRP', "TOP_ETR_BRP", "TOP_PRET_RELAIS_BRP", 
                        'TOP_SURFINANCEMENT_BRP' , "top_exist_conso_revo_BRP", 'COD_CSP_BRP', 'TOP_ASC_DESC_BRP','g_SUM_PATR_IMMO_BRP'], inplace = True)

        ################################################## logit ##############################################################################
                
        X_train_discretise = pd.get_dummies( X, drop_first = True) 

        x_train, x_test, y_train, y_test = train_test_split(X_train_discretise , y.iloc[: ,1], stratify = y.iloc[: ,1] ) 

        # param = {'penalty' : ['elasticnet' ] , 'C' : [0.01, 0.1 , 10] , 'solver' :  ['saga'], 'l1_ratio' : [ 0.01 , 0.1], 
        # 'class_weight' : [ {1 : class_} for class_ in [1,2,5]  ] } 

        # kfold = StratifiedKFold( n_splits = 3 ) 

        # logit = LogisticRegression(class_weight= 'balanced', random_state = 42, fit_intercept=True)
        # model = GridSearchCV(logit, param, cv=kfold ,scoring = 'recall') 
        # model.fit(x_train , y_train)

        # model.best_params_
        # entrainer le modele sur les meilleurs params trouvés grace au grid search cv 

        logit_apres_grid_search = LogisticRegression( random_state = 42 ,  fit_intercept=True, class_weight = '{1: 2}', 
                                                C= 1, l1_ratio =  0, penalty = 'elasticnet', solver = 'saga')

        logit_apres_grid_search.fit(x_train , y_train)
        y_pred_train = logit_apres_grid_search.predict_proba(x_train) 
        y_pred_train = pd.DataFrame(y_pred_train).iloc[: , 1]

        # auc train et test
        score_auc_train = sklearn.metrics.roc_auc_score(y_train, y_pred_train)
        print('AUC TRAIN :' ,score_auc_train.round(2)) 

        y_pred_test = logit_apres_grid_search.predict_proba(x_test) 
        y_pred_test = pd.DataFrame(y_pred_test).iloc[: , 1]
        score_auc_test = sklearn.metrics.roc_auc_score(y_test, y_pred_test)
        print('AUC TEST :' ,score_auc_test.round(2)) 

        fpr_train, tpr_train, _ = roc_curve(y_train, y_pred_train , pos_label = 1 )
        fpr_test, tpr_test, _ = roc_curve(y_test, y_pred_test , pos_label = 1 )

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
        coeff = pd.DataFrame( abs(logit_apres_grid_search.coef_.T), index = x_train.columns, columns = ['coef']).reset_index()
        coeff = coeff.sort_values(by = 'index') 

modele_finale(X,y)
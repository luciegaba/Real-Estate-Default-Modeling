import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px


""" EXPLORATORY DATAVIZ """


def min_max_for_datetime_col(raw_data:pd.DataFrame,col:str)-> None:

    print(f"Date minimale pour {col}:", raw_data["date_debloc_avec_crd"].min())
    print(f"Date maximale pour {col}:", raw_data["date_debloc_avec_crd"].max())


""" MISSING RATE REPORT """
def missing_rate_report(X_train:pd.DataFrame) -> pd.DataFrame:

    missing_rate=pd.DataFrame({'count': X_train.isna().sum(),
                             'rate': (X_train.isna().sum()*100/X_train.shape[0])}).sort_values(by = 'rate', ascending = False)
    display(missing_rate)
    return missing_rate


""" MAP AVEC DEFAUT"""
def map_for_default_risk(df_geo:pd.DataFrame):
 
    fig = px.density_mapbox(df_geo, lat='latitude', lon='longitude',z='defaut_36mois', radius=8,
                        center=dict(lat=df_geo['latitude'].mean(), lon=df_geo['longitude'].mean()), zoom=4.5,
                        mapbox_style="carto-positron", opacity=0.8,width=800, height=800, title="Map of default occurences (1=> many defaults; 0=> few defaults)"
                        , range_color =[21,21])
    fig.show()




""" HISTOGRAMME DEFAUT DANS LE TEMPS"""

def stabilite_global_temps(raw_data:pd.DataFrame) -> None:

    sns.histplot(raw_data["date_debloc_avec_crd"].astype(str).str[:7])
    plt.title("Stabilité de l'ensemble de l'échantillon sur la période d'étude (par mois)")
    plt.xticks(rotation=80, size = 8)
    plt.figure(figsize=(14, 14))
    plt.show()

    sns.histplot(raw_data['date_debloc_avec_crd'].astype(str).str[:4])
    plt.title("Stabilité de l'ensemble de l'échantillon sur la période d'étude (par année)")
    plt.xticks(rotation=80, size = 8)
    plt.figure(figsize=(14, 14))





""" TIME STABILITY """

""" Les fonctions suivantes ont toutes les mêmes arguments: 
    - raw_data: le dataframe sur lequel on vérifie la stabilité temporelle
    - col_date représente la variable crée avec les fonctions "intervall" telles que 'year_intervall' ou 'semester_intervall'. 
    Selon l'intervalle de temps considéré pour la stabilité temporelle, on mettra col_date en colonne utilisée 

"""


def dataviz_stability(data:pd.DataFrame, time_interval='year', stability_for_modelisation=True)-> None:
    
        """ Fonction qui renvoie les graphiques de stabilité temporelle selon l'intervalle de temps souhaité 
        Arguments:
            - raw_data: dataframe initial
            - time_interval: 'year' ou 'semester'
            - stability_for_modelisation: garde uniquement les graphiques essentiels pour finaliser la sélection des variables
            
        """

        raw_data=data.copy()
        if time_interval=='year':
            year_intervall(raw_data)

        elif time_interval=='semester':
            semester_intervall(raw_data)
        raw_data=raw_data.drop(columns='date_debloc_avec_crd')
        
        print("STABILITE DES MODALITES PAR RAPPORT A LA TARGET")
        stability_according_to_the_target(raw_data,time_interval)
        stability_by_repartition_of_defaults(raw_data,time_interval)

        if stability_for_modelisation != True:
            stability_by_repartition_of_modalities(raw_data,time_interval)



def stability_according_to_the_target(raw_data:pd.DataFrame, date:str) -> None:

    """ retourne un graphique pour chaque variable candidate à la modélisation de la stabilité du taux de défaut de ses modalités
    """ 
    print("STABILITE DES MODALITES PAR RAPPORT AU TAUX DE DEFAUT")

    for col in raw_data.drop([date, 'defaut_36mois'], axis=1):
        print(col)
        df_des_valeurs=pd.DataFrame(raw_data.groupby(date)[col].value_counts())
        df_des_valeurs1=pd.DataFrame(raw_data[raw_data['defaut_36mois']==1].groupby(date)[col].value_counts())

        df_des_valeurs1index=df_des_valeurs1.reset_index(inplace=False, level=0)
        df_des_valeurs1index.index.names = ['groupes']
        df_des_valeurs1index=df_des_valeurs1index.reset_index(inplace=False)

        df_des_valeursindex=df_des_valeurs.reset_index(inplace=False, level=0)
        df_des_valeursindex.index.names = ['groupes']
        df_des_valeursindex=df_des_valeursindex.reset_index(inplace=False)

        mergedvar = pd.merge(df_des_valeursindex, df_des_valeurs1index, on =[date,'groupes'], how ="outer")
        mergedvar[col+'%']=np.divide(mergedvar[col+'_y'], mergedvar[col+'_x'])*100
        mergedvar.set_index(date, inplace=True)
        mergedvar.groupby('groupes')[col+'%'].plot(legend=True)
        plt.show()
        


def stability_by_repartition_of_modalities(raw_data:pd.DataFrame,date:str)-> None:

    """ retourne un graphique pour chaque variable candidate à la modélisation de la stabilité de la répartition du défaut selon ses modalités
    """ 
    print("STABILITE DE LA REPARTITION DES MODALITES")

    for col in raw_data.drop([ date, 'defaut_36mois'], axis=1):

        df_col= raw_data.groupby([date, col]).size().unstack()
        list_col_drop=[]

        for i in df_col:
            
            df_col_new=df_col
            df_col_new[i, '%'] = df_col_new[i]*100/df_col_new.iloc[:,:len(df_col_new.columns)].sum(axis=1)
            list_col_drop.append(i)

        df_col_new=df_col_new.drop(list_col_drop, axis=1)
        df_col_new.plot()
    




def stability_by_repartition_of_defaults(raw_data:pd.DataFrame ,date: str):
    """ retourne un graphique pour chaque variable candidate à la modélisation de la stabilité de la répartition des modalités
    """ 
    print("STABILITE DE LA REPARTITION DES MODALITES SELON LE TAUX DE DEFAUT")


    for col in raw_data.drop([date, 'defaut_36mois'], axis=1):

        df_col = raw_data.loc[raw_data['defaut_36mois'] == 1 ].groupby([date, col]).size().unstack()
        list_col_drop=[]

        for i in df_col:

            df_col_new=df_col
            df_col_new[i,'en %'] = df_col_new[i]*100/df_col_new.iloc[:,:len(df_col_new.columns)].sum(axis=1)
            list_col_drop.append(i)

        df_col_new=df_col_new.drop(list_col_drop, axis=1)
        df_col_new.plot()





def year_intervall(raw_data:pd.DataFrame)->None:
    raw_data['year']=raw_data['date_debloc_avec_crd'].map(lambda x: '{year}'.format(year=x.year))



def semester_intervall(raw_data:pd.DataFrame)->None:
    raw_data['semester']=raw_data['date_debloc_avec_crd'].map(lambda x: '{year}-1'.format(year=x.year) if x.month<=6 else '{year}-2'.format(year=x.year))

import streamlit as st
import pandas as pd


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from streamlit_folium import folium_static
from folium.plugins import HeatMap
import folium
from PIL import Image




def app_lcl(df_map,df_cluster,coef_log,feature_imp,):
         # creation of different tabs for the different corresponding sections
    st.header("Dashboard LCL Scoring")

    tab1, tab2, tab3, tab4,tab5= st.tabs(
    ["Mapping et analyse exploratoire ", "Sélection de variables et stabilité temporelle ", "Machine Learning Viz", "Grille de score", " Classe de risque"])
    with tab1:
            st.markdown("Mapping du taux de défaut")
            m=create_map(df_map)
            folium_static(m)
    with tab2:
        st.markdown("Exemple de regroupements de modalité via Optibinning pour IRPAR_USAGE_V12_MAX")
        rf=Image.open("pics/opti.png")
        st.image(rf,caption="Optibinning regroupe les modalités pertinentes ensemble")
        st.markdown("Sélection de variables post-regroupement: Random Forest")
        rf=Image.open("pics/rf.png")
        st.image(rf,caption="Feature importance du Random Forest")
        st.markdown("Stabilité temporelle: exemple pour IRPAR_USAGE_V12_MAX")
        rf=Image.open("pics/graph_stab_1.png")
        st.image(rf,caption="Evolution semestrielle du taux de défaut par modalité pour IRPAR_USAGE_V12_MAX ")
        rf=Image.open("pics/graph_stab_2.png")
        st.image(rf,caption="Evolution semestrielle de la répartition du taux de défaut entre modalités pour IRPAR_USAGE_V12_MAX ")

    with tab3:
        st.markdown("Machine Learning Viz")
        logistic=Image.open("pics/auc_logistic.png")
        st.image(logistic,caption="Performance de la régression logistique")
        st.markdown("Coefficient de la régression logistique")
        st.dataframe(coef_log)
        xgboost=Image.open("pics/auc_xgboost.png")
        st.image(xgboost,caption="Performance du modèle xgboost")
        shapley=Image.open("pics/shap_values_plot.png")
        st.image(shapley,caption="Feature importance: valeurs de Shapley")
        st.markdown("Moyenne des contributions aux valeurs de Shapley sur l'échantillon train")
        st.dataframe(feature_imp)

    with tab4:
        st.markdown("Grille de score")
        grille=Image.open("pics/grille.jpeg")
        st.image(grille)

    with tab5:
        st.markdown("Clusters de risque")
        st.table(df_cluster["cluster"].value_counts())
        annuel=Image.open("pics/clustering_annuel.png")
        st.image(annuel,caption="Clustering sur période annuelle")
        semestre=Image.open("pics/clustering_semestre.png")
        st.image(semestre,caption="Clustering sur période annuelle")






def create_map(df_geo: pd.DataFrame) -> folium.Map():
    df_geo=df_geo.dropna()
    mylist=list(map(list,zip(df_geo["latitude"],df_geo["longitude"],df_geo["defaut_36mois"])))
    location = df_geo['latitude'].mean(), df_geo['longitude'].mean()
    m = folium.Map(location=location, zoom_start=4,)
    HeatMap(mylist).add_to(m)
    return m


def year_intervall(raw_data:pd.DataFrame)->None:
    raw_data['year']=raw_data['date_debloc_avec_crd'].map(lambda x: '{year}'.format(year=x.year))




    


  



def get_data(file):
    
    return pd.read_csv(file).drop(columns="Unnamed: 0")





if __name__ == '__main__':
    
    st.set_page_config(
        page_title="Projet scoring Immo LCL",
        page_icon="", # this icon appears on the tab of the web page
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            'About':"""
            Contributions:
            - Lucie Gabagnous
            - Ghiles Idris
            - Armand L'Huillier
            - Yanis Rehoune
            
            """
        }
    ) 
    
    #@st.cache(allow_output_mutation=True) # for improved performance in (re-)loading data

 
    df_map= get_data("app_data/geocoordonneesBases_recents.csv")
    df_cluster=get_data("app_data/dataframe_classification.csv")
    feature_imp=get_data("app_data/coeff.csv")
    coef_log=get_data("app_data/feature_imp.csv")
    app_lcl(df_map,df_cluster,feature_imp,coef_log)

    





            
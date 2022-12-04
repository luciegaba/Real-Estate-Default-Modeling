import streamlit as st
import pandas as pd


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import streamlit as st
from streamlit_folium import folium_static
import folium
from functions_app import get_data, lcl_app
import branca



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

    df_global = get_data("app_data/df_etude_clean_for_viz.csv")
    df_map= get_data("app_data/geocoordonneesBases_recents.csv")
    df_rf=get_data("app_data/for_random_forest.csv")
    df_ml=get_data("app_data/df_etude_clean_for_viz.csv")
    df_stability=get_data("app_data/for_stability.csv")


    

def app_lcl(df_global,df_map,df_rf,df_ml,df_stability):
         # creation of different tabs for the different corresponding sections
    tab1, tab2, tab3, tab4 = st.tabs(
    ["Mapping et analyse exploratoire ", "Sélection de variables ", "Machine Learning Viz", "Grille de score",])

    with tab1:
        st.markdown("Mapping du taux de défaut")
        m=create_map(df=df_geo)
        folium_static(m)


def create_map(df: pd.DataFrame) -> folium.Map():
    m = folium.Map(location=location, zoom_start=4)

    location = df_geo['latitude'].mean(), df_geo['longitude'].mean()
    m = folium.Map(location=location, zoom_start=4)

    for lat, lon, defaut in zip(
        df['latitude'],
        df['longitude'],
        df['defaut_36mois']):
    
    html = popup_html(shape=defaut)
    iframe = branca.element.IFrame(html=html)
    popup = folium.Popup(folium.Html(html, script=True), parse_html=True)
        #Creating the marker
    folium.Marker(
            #Coordinates of the country
        location = [lat, lon],
            #Popup that shows up if click the marker
        popup = popup,
        ).add_to(m)
    return m



    def 


def get_data(file):
    
    return pd.read_csv(file)


def app_lcl(df: pd.DataFrame):
    
   
    
    
    with tab2:
        
        col1, col2 = st.columns(2)
        
        with col1: # display histograms
            
            fig1 = hist_chart(....)
            st.plotly_chart(fig1)
            
            fig2 = hist_chart(.....)
            st.plotly_chart(fig2)
        
        with col2:
            
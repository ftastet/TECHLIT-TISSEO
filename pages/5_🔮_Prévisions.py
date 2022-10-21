#Import des packages 

import streamlit as st
st.set_page_config(page_title="TISSEO PREVS", page_icon="🔮", layout="wide")

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

#Téléchargements des fichier

itv = pd.read_csv('./Dataset/V1/itv_cleaned_abaques.csv', index_col = 0)
ae_all = pd.read_csv('./Dataset/V1/ae_all_abaques_cleaned.csv', index_col = 0)
ae_tech = pd.read_csv('./Dataset/V1/ae_tech_abaques_cleaned.csv', index_col = 0)
ae_cr = pd.read_csv('./Dataset/V1/ae_cr_abaques_cleaned.csv', index_col = 0)

st.sidebar.subheader("✍🏻 Volume")

st.sidebar.text("")
nbr_itv = st.sidebar.number_input("NBR D'ITV JOURNEE A PREVOIR (RACC & SAV) :", step = 1)

st.sidebar.text("")
st.sidebar.subheader("🛠️ Filtres")

st.sidebar.text("")
jsem = st.sidebar.radio("🔧 JOUR DE LA SEMAINE", key = 14, options = ('Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi'), horizontal = True)

st.sidebar.text("")
type_bts = st.sidebar.radio("🔧 PERIODE", key = 16, options = ("Hors BTS" , "Période BTS"), horizontal = True)    

st.subheader("👉 DIMENSIONNEMENT (Tableau)")
st.text('')           

#Selection dataframe sur jour selectionné   
def dataframe_temp_dim(nbr_itv, flux, jsem, nb_jours, type_bts):

    #selection du flux
    if flux == 'ALL':
        df_temp = ae_all
    if flux == 'TECH':
        df_temp = ae_tech         
    if flux == 'CR':
        df_temp = ae_cr

    #selection du jour de la semaine
    df_temp = df_temp.loc[df_temp['jsem']==jsem]

    #récupérer si BTS ou non    
    if type_bts == "Période BTS":
        df_temp = df_temp[df_temp['BTS']==True]
    else:
        df_temp = df_temp[df_temp['BTS']==False]             

    #Selection des dates à prendre en compte pour les abaques en fonction du nb de jour souhaité
    max_date_temp = df_temp['Date'].unique()[-1]
    if str(nb_jours) != "ALL":
        liste = list(df_temp['Date'].unique())
        max_date_index = liste.index(max_date_temp)
        liste_jour_sel = []
        for i in np.arange(1 , int(nb_jours) + 1, 1):  
            liste_jour_sel.append(liste[max_date_index - i])
        df_temp = df_temp[df_temp['Date'].isin(liste_jour_sel)]                

    df_temp = df_temp[['Date', 'jsem', 'creneau', 'nb_ae', 'dmt', 'BTS']]
    df_temp = df_temp.merge(right = itv, on = ['Date'], how = 'left')
    df_temp['ae_par_itv'] = round((df_temp['nb_ae'] / df_temp['nb_itv']),4)
    df_temp.dropna(axis = 0, inplace = True, how = 'any', subset = ['nb_itv'])
    df_temp.reset_index(inplace = True)
    df_temp = df_temp.loc[df_temp['creneau'] != '20:00']
    df_temp.set_index('creneau', inplace = True)
    df_temp.fillna(0, axis = 1, inplace = True)  

    df_dmt = df_temp.groupby('creneau').agg(
        med_dmt = ('dmt','median'))       

    df_ae = df_temp.groupby('creneau').agg(
        med_ae = ('ae_par_itv','median')) 

    df_temp = pd.concat([df_ae, df_dmt], axis = 1)              

    df_temp['AE'] = round((df_temp['med_ae'] * nbr_itv),1) 
    df_temp['ETP'] = round((df_temp['AE'] * (df_temp['med_dmt']+15)) / 30 / 51,1)  
    df_temp['Sum AE'] = df_temp['AE'].cumsum()        

    #df_temp['Diff. ETP'] = df_temp['ETP'] - df_temp['ETP'].shift(1)  
    #df_temp['Diff. ETP'].fillna(0, inplace = True)

    tot_ae = df_temp['AE'].sum()
    tot_etp = df_temp['ETP'].sum()/2/8    

    return df_temp[['AE', 'ETP', 'Sum AE']], tot_ae, tot_etp

c1, c2, c3, c4 = st.columns((1, 1, 1, 3))     

df_tech, tot_ae_tech, tot_etp_tech = dataframe_temp_dim(
    nbr_itv = nbr_itv, flux = "TECH", jsem = jsem, nb_jours = 4, type_bts = type_bts)

df_cr, tot_ae_cr, tot_etp_cr = dataframe_temp_dim(
    nbr_itv = nbr_itv, flux = "CR", jsem = jsem, nb_jours = 4, type_bts = type_bts)         

df_all = df_tech.add(df_cr, fill_value=0)

with c1:
    st.markdown("**TECH + CR**")
    st.text('')
    st.dataframe(df_all.style.format("{:.0f}"), 300, 750)        

with c2:
    st.markdown("**TECH**")
    st.text('')        
    st.dataframe(df_tech.style.format("{:.0f}"), 300, 750)        

with c3:
    st.markdown("**CR**")
    st.text('')               
    st.dataframe(df_cr.style.format("{:.0f}"), 300, 750)        


df_concat = df_all
df_concat.rename({'AE':'All_AE', 'Sum AE':'All_Sum_AE', 'ETP':'All_ETP'}, axis = 1, inplace = True)
df_concat = pd.concat([df_concat, df_tech[['AE', 'Sum AE', 'ETP']]], axis = 1)
df_concat.rename({'AE':'TECH_AE', 'Sum AE':'TECH_Sum_AE', 'ETP':'TECH_ETP'}, axis = 1, inplace = True)
df_concat = pd.concat([df_concat, df_cr[['AE', 'Sum AE', 'ETP']]], axis = 1)
df_concat.rename({'AE':'CR_AE', 'Sum AE':'CR_Sum_AE', 'ETP':'CR_ETP'}, axis = 1, inplace = True) 
df_concat = round(df_concat,1) 

st.text('')

@st.cache
def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
    return df.to_csv(decimal=",").encode('utf-8')

csv = convert_df(df_concat)

st.download_button(
     label="📩 Télécharger les données 📩",
     data=csv,
     file_name='dimensionnement.csv',
     mime='text/csv',
 )


st.subheader("👉 DIMENSIONNEMENT (Graph)")
st.text('') 

c1, c2, c3 = st.columns((1, 1, 3))  

with c1: 
    dim_output = st.radio("🔧 DIMENSIONNEMENT", key = 4, options = ('ETP', 'AE'), horizontal = True)

with c2:
    dim_flux = st.radio("🔧 FLUX", key = 5, options = ('ALL', 'TECH', 'CR'), horizontal = True)  
    
if dim_flux == 'ALL':   
    df_to_chart = df_all
    df_to_chart.rename({'All_AE':'AE'}, axis = 1, inplace = True)
    df_to_chart.rename({'All_ETP':'ETP'}, axis = 1, inplace = True)    
if dim_flux == 'TECH':   
    df_to_chart = df_tech   
if dim_flux == 'CR':   
    df_to_chart = df_cr   
    
if dim_output == 'ETP':
    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))
    fig.add_trace(go.Bar(x=df_to_chart.index, y=df_to_chart.ETP, 
                            name='Médiane 4J',
                            text=round(df_to_chart.ETP),
                            textposition='auto', 
                            marker=dict(color='royalblue'), 
                                ))
    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))  
    fig.update_layout(
        title= "Nombre de personnes à prévoir par créneau horaire",
        xaxis_title="Créneau horaire",
        yaxis_title="ETP",
        legend_title="ETP",
        font=dict(
            size=14,
            color="black"
            ))   
    st.plotly_chart(fig, use_container_width=True) 
    
if dim_output == 'AE':
    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))
    fig.add_trace(go.Bar(x=df_to_chart.index, y=df_to_chart.AE,
                            name='Médiane 4J',
                            text=round(df_to_chart.AE),
                            textposition='auto',
                            marker=dict(color='wheat')
                                ))
    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))  
    fig.update_layout(
        title= "Nombre de personnes à prévoir par créneau horaire",
        xaxis_title="Créneau horaire",
        yaxis_title="AE",
        legend_title="AE",
        font=dict(
            size=14,
            color="black"
            ))   
    st.plotly_chart(fig, use_container_width=True)

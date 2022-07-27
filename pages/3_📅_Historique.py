#Import des packages 

import streamlit as st
st.set_page_config(page_title="TISSEO HISTO", page_icon="📅", layout="wide")

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

st.sidebar.text("")
st.sidebar.subheader("🛠️ Filtres")

#Sélection d'une date    
min_date = ae_all['Date'].unique()[0]
max_date = ae_all['Date'].unique()[-1] 

st.sidebar.text("")
date_choix = st.sidebar.date_input("🔧 DATE :",
              date(int(max_date[0:4]), int(max_date[5:7]), int(max_date[8:10])), 
              min_value = date(int(min_date[0:4]), int(min_date[5:7]), int(min_date[8:10])), 
              max_value = date(int(max_date[0:4]), int(max_date[5:7]), int(max_date[8:10]))
             )  
date = str(date_choix)

#récupérer le jour semaine de la date sélectionnée
jsem = ae_all.loc[ae_all['Date']==date]['jsem'].mode()[0]

st.sidebar.text("")
flux = st.sidebar.radio("🔧 TYPE DE FLUX", key = 1, options = ('ALL', 'TECH', 'CR'), horizontal = True)    

st.sidebar.text("")
nb_jours = st.sidebar.radio("🔧 NOMBRE DE " + jsem + " PRECEDENTS", key = 2, options = (4, 8, "All"), horizontal = True) 

#Selection dataframe sur jour selectionné   
def dataframe_temp_flux(date, flux, jsem, nb_jours):

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
    bts = df_temp[df_temp['Date']==date]['BTS'].mode()[0]
    if bts == True:
        df_temp = df_temp[df_temp['BTS']==True]
    else:
        df_temp = df_temp[df_temp['BTS']==False]             

    #Selection des dates à prendre en compte pour les abaques en fonction du nb de jour souhaité
    max_date_temp = df_temp['Date'].unique()[-1]
    if str(nb_jours) != "All":
        liste = list(df_temp['Date'].unique())
        max_date_index = liste.index(max_date_temp)
        liste_jour_sel = []
        for i in np.arange(1 , int(nb_jours) + 1, 1):  
            liste_jour_sel.append(liste[max_date_index - i])
        df_temp = df_temp[df_temp['Date'].isin(liste_jour_sel)]                

    df_temp = df_temp[['Date', 'jsem', 'creneau', 'nb_ae', 'dmr', 'TR90', 'cs_ae_rep', 'cs_tr90', 'BTS']]
    df_temp = df_temp.merge(right = itv, on = ['Date'], how = 'left')
    df_temp['ae_par_itv'] = round((df_temp['nb_ae'] / df_temp['nb_itv']),4)
    df_temp.dropna(axis = 0, inplace = True, how = 'any', subset = ['nb_itv'])
    df_temp.reset_index(inplace = True)
    df_temp = df_temp.loc[df_temp['creneau'] != '20:00']
    df_temp.set_index('creneau', inplace = True)
    df_temp.fillna(0, axis = 1, inplace = True)                

    return df_temp

df_initial = dataframe_temp_flux(date = date, flux = flux, jsem = jsem, nb_jours = nb_jours)  

#Selection dataframe sur jour selectionné   
def dataframe_temp_date(date, flux):

    #selection du flux
    if flux == 'ALL':
        df_temp_date = ae_all
    if flux == 'TECH':
        df_temp_date = ae_tech         
    if flux == 'CR':
        df_temp_date = ae_cr

    df_temp_date = df_temp_date.loc[df_temp_date['Date']==date] 

    df_temp_date = df_temp_date[['Date', 'jsem', 'creneau', 'nb_ae', 'dmr', 'TR90', 'cs_ae_rep', 'cs_tr90', 'BTS']]
    df_temp_date = df_temp_date.merge(right = itv, on = ['Date'], how = 'left')
    df_temp_date['ae_par_itv'] = round((df_temp_date['nb_ae'] / df_temp_date['nb_itv']),4)
    df_temp_date.dropna(axis = 0, inplace = True, how = 'any', subset = ['nb_itv'])
    df_temp_date.reset_index(inplace = True)
    df_temp_date = df_temp_date.loc[df_temp_date['creneau'] != '20:00']
    df_temp_date.set_index('creneau', inplace = True)
    df_temp_date.fillna(0, axis = 1, inplace = True)                

    return df_temp_date

df_initial_date = dataframe_temp_date(date = date, flux = flux)    

#----------------------------#
#   DF FLUX REEL
#----------------------------#    

df_flux = df_initial
df_flux_date = df_initial_date

#Fonction percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_    

#création du Dataframe
df_flux = df_flux.groupby('creneau').agg(
    q1 = ('ae_par_itv',percentile(20)), 
    med = ('ae_par_itv','median'), 
    q3 = ('ae_par_itv',percentile(80)), 
)

#Suppression des NAs si il y en a
df_flux.fillna(0, axis = 1, inplace = True)

#nb itv du jour sélectionné
nb_itv = int(itv.loc[itv['Date']==date]['nb_itv'])

#Création des stats pour les calculs et visus
df_flux['min'] = round(df_flux['q1'] * nb_itv).astype(int)
df_flux['med_itv'] = round(df_flux['med'] * nb_itv).astype(int)
df_flux['max'] = round(df_flux['q3'] * nb_itv).astype(int)
df_flux['cumul_med'] = round(df_flux['med_itv'].cumsum()).astype(int)

#----------------------------#
#          DF TR90
#----------------------------#    

df_tr90 = df_initial
df_tr90_date = df_initial_date

#Fonction percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_ 

#création du Dataframe
df_tr90 = df_tr90.groupby('creneau').agg(
    q1 = ('TR90',percentile(20)), 
    med = ('TR90','median'), 
    q3 = ('TR90',percentile(80)), 
)     
df_tr90  = df_tr90 .assign(obj_min=70)      

#----------------------------#
#          DF DMR
#----------------------------#    

df_dmr = df_initial
df_dmr_date = df_initial_date

#Fonction percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_ 

#création du Dataframe
df_dmr = df_dmr.groupby('creneau').agg(
    q1 = ('dmr',percentile(25)), 
    med = ('dmr','median'), 
    q3 = ('dmr',percentile(75)), 
) 
df_dmr  = df_dmr .assign(obj_min=90)          

#----------------------------#
#         GRAPH AE REEL
#----------------------------#       

st.subheader("👉 CHIFFRES CLE")
st.text('')   

c1, c2, c3, c4 = st.columns((1, 1, 1, 3))     

with c1:
    
    st.markdown("**🔰 ITV**")
    st.markdown(str(df_flux_date['nb_itv'].max()) + " interventions")
                
with c2:

    st.markdown("**📞 AE**")              
    st.markdown(str(df_flux_date['nb_ae'].sum()) + " AE reçus")

    df_flux['prev_ae'] = round(df_flux['med'] * df_flux_date['nb_itv'].max()).astype(int)
    df_flux['prev_ae_cs'] = round(df_flux['prev_ae'].cumsum()).astype(int)
    st.markdown(str(df_flux['prev_ae_cs'].max()) + " AE prévus")

    tx_prev = (df_flux['prev_ae_cs'].max() / df_flux_date['nb_ae'].sum()) * 100
    st.markdown("Prevs vs réel = " + str(round(tx_prev,1)) + "%")

with c3:

    st.markdown("**📉 TR90**")   
                
    tr90 = int(df_flux_date['cs_tr90'].iloc[[-1]])
    st.markdown("TR90 de la journée = " + str(tr90) + "%")

    nb_bad_creneau_tr = df_flux_date.loc[df_flux_date['TR90'] < 70].shape[0]
    tx_nb_bad_creneau_tr = round((nb_bad_creneau_tr / 24) * 100,1)
    st.markdown("Dont " + str(nb_bad_creneau_tr) + " créneaux < 70%")
    st.markdown("soit " + str(tx_nb_bad_creneau_tr) + "% de la journée")

    

    
st.text('')    
    
#----------------------------#
#         GRAPH AE REEL
#----------------------------#       

st.subheader("👉 VOLUME AE vs OBSERVATIONS")
st.text('')   

my_expander = st.expander("VOLUME AE vs OBSERVATIONS", expanded=True)
with my_expander:      

    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))

    fig.add_trace(go.Scatter(x=df_flux.index, y=df_flux['max'], 
                             line=dict(color='darkblue', width=0.8, dash = 'dot'),
                             name='80p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)

    fig.add_trace(go.Scatter(x=df_flux.index, y=df_flux['med_itv'], 
                            fill='tonexty', fillcolor='white',  
                            line=dict(color='black', width=1, dash = 'dot'), 
                            mode = 'lines', opacity = 0.7, name = 'med')) 
    fig.update_traces(hovertemplate=None) 

    fig.add_trace(go.Scatter(x=df_flux.index, y=df_flux['min'], 
                             fill='tonexty', fillcolor='white', 
                             line=dict(color='midnightblue', width=0.8, dash = 'dot'),
                             name='20p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)

    fig.add_trace(go.Scatter(x=df_flux_date.index, y=df_flux_date['nb_ae'],
                        line=dict(color='darkblue', width=6), 
                        mode='lines+markers',
                        marker=dict(size=15, color='midnightblue', line = dict(color='black', width = 2)), 
                        name='Réel'
                            ))
    fig.update_traces(hovertemplate=None)  

    fig.update_layout(hovermode="x")

    fig.update_yaxes(tickfont=dict(size=16), 
                     showline=True, 
                     showgrid=False, 
                     zeroline = False, 
                     linewidth=1.5, 
                     linecolor='black', 
                     mirror=True)
    fig.update_xaxes(tickfont=dict(size=16), 
                     showline=True, 
                     showgrid=True, 
                     zeroline = False, 
                     linewidth=1.5, 
                     linecolor='black', 
                     mirror=True)    

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 

    fig.update_layout(
        title= "NB AE vs OBSERVATIONS",
        xaxis_title="Créneau horaire",
        yaxis_title="Volume AE",
        legend_title="Volume appels",
        font=dict(
            size=14,
            color="black"
            ))

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))            

    st.plotly_chart(fig, use_container_width=True)  

#----------------------------#
#         GRAPH TR90
#----------------------------#       

st.subheader("👉 TR90s vs OBSERVATIONS")
st.text('')   

my_expander = st.expander("TR90s vs OBSERVATIONS", expanded=True)
with my_expander:   


    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))

    fig.add_trace(go.Scatter(x=df_tr90.index, y=df_tr90['q3'], 
                             line=dict(color='darkgreen', width=0.8, dash = 'dot'),
                             name='80p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)

    fig.add_trace(go.Scatter(x=df_tr90.index, y=df_tr90['med'], 
                            fill='tonexty', fillcolor='white',  
                            line=dict(color='black', width=1, dash = 'dot'), 
                            mode = 'lines', opacity = 0.7, name = 'med')) 
    fig.update_traces(hovertemplate=None) 

    fig.add_trace(go.Scatter(x=df_tr90.index, y=df_tr90['q1'], 
                             fill='tonexty', fillcolor='white', 
                             line=dict(color='darkgreen', width=0.8, dash = 'dot'),
                             name='20p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)

    fig.add_trace(go.Scatter(x=df_tr90.index, y=df_tr90['obj_min'],
                    line=dict(color='red', width=3, dash = 'dot'),                 
                    name='obj min'
                    ))                 

    fig.add_trace(go.Scatter(x=df_tr90_date.index, y=df_tr90_date['TR90'],
                        line=dict(color='green', width=6), 
                        mode='lines+markers',
                        marker=dict(size=15, color='green', line = dict(color='black', width = 2)), 
                        name='Réel'
                            ))
    fig.update_traces(hovertemplate=None)  

    fig.update_layout(hovermode="x")

    fig.update_yaxes(tickfont=dict(size=16), 
                     showline=True, 
                     showgrid=False, 
                     zeroline = False, 
                     linewidth=1.5, 
                     linecolor='black', 
                     mirror=True)
    fig.update_xaxes(tickfont=dict(size=16), 
                     showline=True, 
                     showgrid=True, 
                     zeroline = False, 
                     linewidth=1.5, 
                     linecolor='black', 
                     mirror=True)    

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 

    fig.update_layout(
        title= "TR90 vs OBSERVATIONS",
        xaxis_title="Créneau horaire",
        yaxis_title="TR90s",
        legend_title="TR90s",
        font=dict(
            size=14,
            color="black"
            ))

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))            

    st.plotly_chart(fig, use_container_width=True)  





#----------------------------#
#         GRAPH DMR
#----------------------------#       

st.subheader("👉 DMR vs OBSERVATIONS")
st.text('') 

my_expander = st.expander("DMR vs OBSERVATIONS", expanded=True)
with my_expander:              


    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))

    fig.add_trace(go.Scatter(x=df_dmr.index, y=df_dmr['q3'], 
                             line=dict(color='darkorange', width=0.8, dash = 'dot'),
                             name='80p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)

    fig.add_trace(go.Scatter(x=df_dmr.index, y=df_dmr['med'], 
                            fill='tonexty', fillcolor='white',  
                            line=dict(color='black', width=1, dash = 'dot'), 
                            mode = 'lines', opacity = 0.7, name = 'med')) 
    fig.update_traces(hovertemplate=None) 

    fig.add_trace(go.Scatter(x=df_dmr.index, y=df_dmr['q1'], 
                             fill='tonexty', fillcolor='white', 
                             line=dict(color='darkorange', width=0.8, dash = 'dot'),
                             name='20p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)

    fig.add_trace(go.Scatter(x=df_dmr.index, y=df_dmr['obj_min'],
                    line=dict(color='red', width=3, dash = 'dot'),                 
                    name='obj min'
                    ))              

    fig.add_trace(go.Scatter(x=df_dmr_date.index, y=df_dmr_date['dmr'],
                        line=dict(color='orange', width=6), 
                        mode='lines+markers',
                        marker=dict(size=15, color='orange', line = dict(color='black', width = 2)), 
                        name='Réel'
                            ))
    fig.update_traces(hovertemplate=None)  

    fig.update_layout(hovermode="x")

    fig.update_yaxes(tickfont=dict(size=16), 
                     showline=True, 
                     showgrid=False, 
                     zeroline = False, 
                     linewidth=1.5, 
                     linecolor='black', 
                     mirror=True)
    fig.update_xaxes(tickfont=dict(size=16), 
                     showline=True, 
                     showgrid=True, 
                     zeroline = False, 
                     linewidth=1.5, 
                     linecolor='black', 
                     mirror=True)    

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 

    fig.update_layout(
        title= "DMR vs OBSERVATIONS",
        xaxis_title="Créneau horaire",
        yaxis_title="DMR",
        legend_title="DMR",
        font=dict(
            size=14,
            color="black"
            ))

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))            

    st.plotly_chart(fig, use_container_width=True)          
        
#Import des packages 

import streamlit as st
st.set_page_config(page_title="TISSEO ANALYSE", page_icon="🎯", layout="wide")

import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

import os

#Téléchargements des fichier

itv = pd.read_csv('./Dataset/V1/itv_cleaned_abaques.csv', index_col = 0)
ae_all = pd.read_csv('./Dataset/V1/ae_all_abaques_cleaned.csv', index_col = 0)
ae_tech = pd.read_csv('./Dataset/V1/ae_tech_abaques_cleaned.csv', index_col = 0)
ae_cr = pd.read_csv('./Dataset/V1/ae_cr_abaques_cleaned.csv', index_col = 0)

st.sidebar.subheader("🛠️ CONFIGURATION & INPUT")

#Sélection d'une date    
min_date = ae_all['Date'].unique()[0]
max_date = ae_all['Date'].unique()[-1] 

st.sidebar.text("")
jour = st.sidebar.date_input("🔧 SELECTION JOUR", date(int(max_date[0:4]), int(max_date[5:7]), int(max_date[8:10])))  
date = str(jour)  

#récupérer le jour semaine de la date sélectionnée
jsem = ae_all.loc[ae_all['Date']==date]['jsem'].mode()[0]

st.sidebar.text("")
type_bts = st.sidebar.radio("🔧 PERIODE", key = 1, options = ("Hors BTS" , "Période BTS"), horizontal = True)    

st.sidebar.text("")
uploaded_genesys = st.sidebar.file_uploader("📁 FICHIER GENESYS")

st.sidebar.text("")
uploaded_sfs = st.sidebar.file_uploader("📁 FICHIER SFS")

st.sidebar.text("")
launch = st.sidebar.button(label = 'LANCER 🚀')
st.sidebar.text("")
st.sidebar.text("")
st.sidebar.text("")


st.subheader("👉 INTRODUCTION")
st.text('')  

st.markdown(""" 
    Cette fonctionnalité permet d'analyser un jour passé qui n'est pas encore historisée voir même une journée en cours.

    Il suffit de remplir les prérequis à gauche et de déposer les fichiers
    - Pas besoin d'avoir les extract précis du jour en question
    - **Quand les prerequis et options à gauche sont remplies, cliquez sur LANCER** dans la partie de gauche                     
    """)
st.text('') 

if launch == True:  
    if uploaded_genesys is not None: 
        if uploaded_sfs is not None: 

            #Ouverture du fichier genesys
            ae = pd.read_excel(uploaded_genesys, sheet_name = "Appels par activité flux")
            ae.drop('Unnamed: 0', axis = 1, inplace = True)
            headers = ae.iloc[0]
            ae = pd.DataFrame(ae.values[1:], columns=headers)

            #Manip de ménage
            ae["Durée moyenne de réponse IVR inclus"].fillna(0, inplace = True)
            ae["DMR"] = ae["Durée moyenne de réponse IVR inclus"].astype(int)
            ae["DMT"].fillna(0, inplace = True)
            ae["DMT"] = ae["DMT"].astype(int)

            ae_temp = ae[['Date', 
                          'Flux', 
                          'Nb appels reçus', 
                          "Nb appels répondus", 
                          "Nb Appels Répondus en moins de 90s", 
                          'DMR', 
                          "DMT"
                         ]]
            ae_temp.dropna(axis=0, how='any', inplace = True)

            ae_temp.rename({"Nb appels répondus":"nb_ae", 
                            "Nb Appels Répondus en moins de 90s":"nb_ae_tr90", 
                            "DMT":"dmt"}, axis = 1, inplace = True)                

            ae_temp['Date'] = pd.to_datetime(ae_temp['Date'], errors='coerce', utc=True) 

            ae_temp['Year'] = ae_temp['Date'].dt.year
            ae_temp['Mois'] = ae_temp['Date'].dt.month
            ae_temp['Semaine'] = ae_temp['Date'].dt.isocalendar().week
            ae_temp['Year_Sem'] = ae_temp['Year'].astype(str) + '_' + ae_temp['Semaine'].astype(str)
            ae_temp['jsem'] = ae_temp['Date'].dt.dayofweek
            ae_temp['creneau'] = ae_temp['Date'].astype(str).apply(lambda x: x[11:-9])    
            ae_temp['Date'] = ae_temp['Date'].dt.date            

            ae_temp['BTS'] = ae_temp['Year_Sem'].apply(lambda x: True if type_bts == "Période BTS" else False)
            ae_temp['jsem'].replace([0,1,2,3,4,5,6], ['Lundi','Mardi','Mercredi','Jeudi','Vendredi','Samedi','Dimanche'], inplace = True)                   

            #Création du dataframe AE avec le type de jour
            ae_temp['Date'] = pd.to_datetime(ae_temp['Date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')   
            ae_temp = ae_temp.loc[ae_temp['Date']==date]

            ae_temp = ae_temp.loc[ae_temp['creneau'] != '20:00']
            ae_temp = ae_temp.loc[ae_temp['creneau'] != '20:30']
            ae_temp = ae_temp.loc[ae_temp['creneau'] != '07:00']
            ae_temp = ae_temp.loc[ae_temp['creneau'] != '07:30']
            ae_temp = ae_temp.loc[ae_temp['creneau'] != '00:00']                

            ae_temp_all = ae_temp[ae_temp['Flux'].isin(['PILOTAGE_TECHNICIEN', 'VALIDATION_CR'])]
            
            ae_temp_all['cs_ae_tr90']  = ae_temp_all['nb_ae_tr90'].cumsum() 
            ae_temp_all['cs_ae']  = ae_temp_all['nb_ae'].cumsum() 
            ae_temp_all['cs_tr90']  = (ae_temp_all['cs_ae_tr90'] / ae_temp_all['cs_ae']) * 100         
            
            ae_temp_tech = ae_temp[ae_temp['Flux'].isin(['PILOTAGE_TECHNICIEN'])]
            
            ae_temp_tech['cs_ae_tr90']  = ae_temp_tech['nb_ae_tr90'].cumsum() 
            ae_temp_tech['cs_ae']  = ae_temp_tech['nb_ae'].cumsum() 
            ae_temp_tech['cs_tr90']  = (ae_temp_tech['cs_ae_tr90'] / ae_temp_tech['cs_ae']) * 100             
                     
            ae_temp_cr = ae_temp[ae_temp['Flux'].isin(['VALIDATION_CR'])]    
            
            ae_temp_cr['cs_ae_tr90']  = ae_temp_cr['nb_ae_tr90'].cumsum() 
            ae_temp_cr['cs_ae']  = ae_temp_cr['nb_ae'].cumsum() 
            ae_temp_cr['cs_tr90']  = (ae_temp_cr['cs_ae_tr90'] / ae_temp_cr['cs_ae']) * 100                
            
            #Sauvegarde du fichier uploader si besoin usage plus tard
            #currentDateTime = datetime.now().strftime("%m-%d-%Y")
            #ae_temp_all.to_csv(f"./Dataset/Upload_files/ae_temp_all_{currentDateTime}.csv", index = False) 
            ae_temp_all.to_csv(f"./Dataset/Upload_files/ae_temp_all.csv", index = False)                    
            ae_temp_tech.to_csv(f"./Dataset/Upload_files/ae_temp_tech.csv", index = False) 
            ae_temp_cr.to_csv(f"./Dataset/Upload_files/ae_temp_cr.csv", index = False)                 

        else:
            st.warning("Il manque le fichier SFS")        

    if uploaded_sfs is not None: 
        if uploaded_genesys is not None:   

            cols_to_keep = ['ID Externe', 'Type de travail', 
                      'Statut', 'Motif de Clôture', "Début de la fenêtre d'arrivée", "Fin de la fenêtre d'arrivée", 
                      'Début planifié', 'Fin planifiée']

            itv_temp = pd.read_excel(uploaded_sfs)
            itv_temp = itv_temp[cols_to_keep]
            itv_temp['Date'] = itv_temp["Début de la fenêtre d'arrivée"].dt.date
            #itv_temp['Date'] = pd.to_datetime(itv_temp['Date'], format='%Y-%m-%d')

            itv_temp['Date'] = pd.to_datetime(itv_temp['Date'], errors='coerce', utc=True).dt.strftime('%Y-%m-%d')                  
            itv_temp = itv_temp.loc[itv_temp['Date']==date]  

            #SELECTION STATUT
            itv_temp = itv_temp[(itv_temp['Statut']=='Terminé')|(itv_temp['Statut']=='Incomplet')]

            #RENOMMAGE DES COLONNES
            itv_temp.rename({'Type de travail':'WTN', 
                        "Début de la fenêtre d'arrivée":"Debut", 
                        "Fin de la fenêtre d'arrivée":"Fin", 
                        'Motif de Clôture': 'code_cloture', 

                        }, axis = 1, inplace = True)

            #AJOUT STATUT OK/KO
            ok_ko = []
            for i in itv_temp['Statut']:
                if i == 'Terminé':
                    ok_ko.append('OK')
                else:
                    ok_ko.append('KO') 
            itv_temp['OK_KO'] = ok_ko

            #AJOUT DU TYPE D'ITV
            type_itv = []
            for i in itv_temp['WTN']:
                if i[:3] == 'SAV':
                    type_itv.append('SAV')
                else:
                    type_itv.append('RACC')
            itv_temp['type_itv'] = type_itv

            itv_temp = itv_temp[(itv_temp['WTN']!='ASSIST_N')&(itv_temp['WTN']!='ASSIST_T')]
            itv_temp = itv_temp[(itv_temp['code_cloture']!='Garder en main')]

            #itv_temp['Mois'] = itv_temp['Debut'].dt.month

            itv_temp.dropna(axis=0, how='any', inplace = True)
            itv_temp = itv_temp[['Date', 'WTN']]
            itv_temp.sort_values(by = ['Date'], inplace = True)                            

            #itv_temp['Date'] = pd.to_datetime(itv_temp['Date'])
            itv_temp = itv_temp.groupby('Date').agg(nb_itv = ('WTN', 'count')).reset_index()

            #Sauvegarde du fichier uploader si besoin usage plus tard
            #currentDateTime = datetime.now().strftime("%m-%d-%Y")
            #itv_temp.to_csv(f"./Dataset/Upload_files/itv_temp_{currentDateTime}.csv", index = False) 
            itv_temp.to_csv(f"./Dataset/Upload_files/itv_temp.csv", index = False)                                       

        else:
            st.warning("Il manque le fichier Genesys")                 

    
st.subheader("👉 RESULTATS")
st.text('')  

itv_temp = pd.read_csv('./Dataset/Upload_files/itv_temp.csv')
ae_temp_all = pd.read_csv('./Dataset/Upload_files/ae_temp_all.csv')
ae_temp_tech = pd.read_csv('./Dataset/Upload_files/ae_temp_tech.csv')
ae_temp_cr = pd.read_csv('./Dataset/Upload_files/ae_temp_cr.csv')

nbr_itv = itv_temp['nb_itv'].max()
date_cache = itv_temp['Date'].unique()[-1]

st.info("Pour information l'analyse actuellement affichée concerne la date du " + str(date_cache))
 
c1, c2, c3 = st.columns((1, 1, 2))  

with c1:
    flux = st.radio("🔧 TYPE DE FLUX", key = 3, options = ('ALL', 'TECH', 'CR'), horizontal = True)  
    st.text('')  

    #----------------------------#
    #   DF HISTORIQUE
    #----------------------------#      

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

        df_temp = df_temp[['Date', 'jsem', 'creneau', 'nb_ae', 'dmt', 'dmr', 'TR90', 'BTS']]               

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

        df_tr = df_temp.groupby('creneau').agg(
            med_tr = ('TR90','median')) 

        df_dmr = df_temp.groupby('creneau').agg(
            med_dmr = ('dmr','median'))                

        df_temp = pd.concat([df_ae, df_dmt, df_tr, df_dmr], axis = 1)              

        df_temp['AE'] = round((df_temp['med_ae'] * nbr_itv),1) 
        df_temp['Sum AE'] = df_temp['AE'].cumsum()      

        tot_ae = df_temp['AE'].sum()   

        return df_temp[['AE', 'Sum AE', 'med_tr', 'med_dmr', 'med_dmt']], tot_ae

    df_tech, tot_ae_tech = dataframe_temp_dim(
        nbr_itv = nbr_itv, flux = "TECH", jsem = jsem, nb_jours = 4, type_bts = type_bts)

    df_cr, tot_ae_cr = dataframe_temp_dim(
        nbr_itv = nbr_itv, flux = "CR", jsem = jsem, nb_jours = 4, type_bts = type_bts)

    df_all, tot_ae_all = dataframe_temp_dim(
        nbr_itv = nbr_itv, flux = "ALL", jsem = jsem, nb_jours = 4, type_bts = type_bts)      


    if flux == 'ALL':
        df = df_all

        df_tr = ae_temp_all.groupby('creneau').agg(
            nb_ae_tr90 = ('nb_ae_tr90','sum'), nb_ae = ('nb_ae','sum')) 
        df_tr['TR90'] = (df_tr['nb_ae_tr90'] / df_tr['nb_ae'] * 100) 

        df_fe = ae_temp_all.groupby('creneau').agg(nb_ae = ('nb_ae','sum'))   
        df_dmr = ae_temp_all.groupby('creneau').agg(dmr = ('DMR','mean'))   
        df_dmt = ae_temp_all.groupby('creneau').agg(dmt = ('dmt','mean'))

    if flux == 'TECH':
        df = df_tech

        df_tr = ae_temp_tech.groupby('creneau').agg(
            nb_ae_tr90 = ('nb_ae_tr90','sum'), nb_ae = ('nb_ae','sum')) 
        df_tr['TR90'] = (df_tr['nb_ae_tr90'] / df_tr['nb_ae'] * 100)       

        df_fe = ae_temp_tech.groupby('creneau').agg(nb_ae = ('nb_ae','sum'))  
        df_dmr = ae_temp_tech.groupby('creneau').agg(dmr = ('DMR','mean'))
        df_dmt = ae_temp_tech.groupby('creneau').agg(dmt = ('dmt','mean'))    


    if flux == 'CR':
        df = df_cr    

        df_tr = ae_temp_cr.groupby('creneau').agg(
            nb_ae_tr90 = ('nb_ae_tr90','sum'), nb_ae = ('nb_ae','sum')) 
        df_tr['TR90'] = (df_tr['nb_ae_tr90'] / df_tr['nb_ae'] * 100)   

        df_fe = ae_temp_cr.groupby('creneau').agg(nb_ae = ('nb_ae','sum'))  
        df_dmr = ae_temp_cr.groupby('creneau').agg(dmr = ('DMR','mean'))      
        df_dmt = ae_temp_cr.groupby('creneau').agg(dmt = ('dmt','mean'))     


    df_fe = pd.merge(df_fe.reset_index(), df.reset_index(), on="creneau")
    df_fe = df_fe[['creneau', 'nb_ae', 'AE']]
    df_fe.rename({'AE':'nb_ae_prev'}, axis = 1, inplace = True)        
    
    df_tr = pd.merge(df_tr.reset_index(), df.reset_index(), on="creneau")
    df_tr = df_tr[['creneau', 'TR90', 'med_tr']]

    df_dmr = pd.merge(df_dmr.reset_index(), df.reset_index(), on="creneau")
    df_dmr = df_dmr[['creneau', 'dmr', 'med_dmr']]

    df_dmt = pd.merge(df_dmt.reset_index(), df.reset_index(), on="creneau")
    df_dmt = df_dmt[['creneau', 'dmt', 'med_dmt']]

with c2:
    analyse = st.radio("🔧 INFO A ANALYSER", key = 4, options = ('AE', 'TR90', 'DMR', 'DMT'), horizontal = True)  
    st.text('')  
    
if analyse == 'AE':  
    
    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))

    fig.add_trace(go.Scatter(x=df_fe.creneau, y=round(df_fe['nb_ae_prev'],1), 
                            #fill='tonexty', fillcolor='white',  
                            line=dict(color='blue', width=3), 
                            mode='lines+markers',
                            marker=dict(size=8, color='blue', line = dict(color='black', width = 1)), 
                            name='Médiane 4J'
                                ))
    fig.update_traces(hovertemplate=None) 
    
    fig.add_trace(go.Scatter(x=df_fe.creneau, y=round(df_fe['nb_ae'],1), 
                            #fill='tonexty', fillcolor='white',  
                            line=dict(color='darkblue', width=6), 
                            mode='lines+markers',
                            marker=dict(size=15, color='midnightblue', line = dict(color='black', width = 2)), 
                            name='Réel'
                                ))
    fig.update_traces(hovertemplate=None)               

    fig.update_layout(hovermode="x")

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))  
    
    fig.update_layout(
        title= "Volume d'appels entrants",
        xaxis_title="Créneau horaire",
        yaxis_title="AE",
        legend_title="AE",
        font=dict(
            size=14,
            color="black"
            ))   

    st.plotly_chart(fig, use_container_width=True)  
    
if analyse == 'TR90':  
    
    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))

    fig.add_trace(go.Scatter(x=df_tr.creneau, y=round(df_tr['med_tr'],1), 
                            #fill='tonexty', fillcolor='white',  
                            line=dict(color='limegreen', width=3), 
                            mode='lines+markers',
                            marker=dict(size=8, color='limegreen', line = dict(color='black', width = 1)), 
                            name='Médiane 4J'
                                ))
    fig.update_traces(hovertemplate=None) 
    
    fig.add_trace(go.Scatter(x=df_tr.creneau, y=round(df_tr['TR90'],1), 
                            #fill='tonexty', fillcolor='white',  
                            line=dict(color='green', width=6), 
                            mode='lines+markers',
                            marker=dict(size=15, color='green', line = dict(color='black', width = 2)), 
                            name='Réel'
                                ))
    fig.update_traces(hovertemplate=None)               

    fig.update_layout(hovermode="x")

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))         

    fig.update_layout(
        title= "Temps de réponse en moins de 90 secondes",
        xaxis_title="Créneau horaire",
        yaxis_title="TR90s",
        legend_title="TR90s",
        font=dict(
            size=14,
            color="black"
            ))   
   
    st.plotly_chart(fig, use_container_width=True)  
    
if analyse == 'DMR':  
    
    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))

    fig.add_trace(go.Scatter(x=df_dmr.creneau, y=round(df_dmr['med_dmr'],1), 
                            #fill='tonexty', fillcolor='white',  
                            line=dict(color='wheat', width=3), 
                            mode='lines+markers',
                            marker=dict(size=8, color='wheat', line = dict(color='black', width = 1)), 
                            name='Médiane 4J'
                                ))
    fig.update_traces(hovertemplate=None) 
    
    fig.add_trace(go.Scatter(x=df_dmr.creneau, y=round(df_dmr['dmr'],1), 
                            #fill='tonexty', fillcolor='white', 
                            line=dict(color='orange', width=6), 
                            mode='lines+markers',
                            marker=dict(size=15, color='orange', line = dict(color='black', width = 2)), 
                            name='Réel'
                                ))
    fig.update_traces(hovertemplate=None)     
    
    fig.update_layout(hovermode="x")

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)
    
    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))         

    fig.update_layout(
        title= "Délais moyen de réponse (secondes)",
        xaxis_title="Créneau horaire",
        yaxis_title="DMR",
        legend_title="DMR",
        font=dict(
            size=14,
            color="black"
            ))       
    
    st.plotly_chart(fig, use_container_width=True)    
    
if analyse == 'DMT':  
    
    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))

    fig.add_trace(go.Scatter(x=df_dmt.creneau, y=round(df_dmt['med_dmt'],1), 
                            #fill='tonexty', fillcolor='white',  
                            line=dict(color='yellowgreen', width=3), 
                            mode='lines+markers',
                            marker=dict(size=8, color='yellowgreen', line = dict(color='black', width = 1)), 
                            name='Médiane 4J'
                                ))
    fig.update_traces(hovertemplate=None) 
    
    fig.add_trace(go.Scatter(x=df_dmt.creneau, y=round(df_dmt['dmt'],1), 
                            #fill='tonexty', fillcolor='white',  
                            line=dict(color='darkolivegreen', width=6), 
                            mode='lines+markers',
                            marker=dict(size=15, color='darkolivegreen', line = dict(color='black', width = 2)), 
                            name='Réel'
                                ))
    fig.update_traces(hovertemplate=None)               

    fig.update_layout(hovermode="x")

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))         

    fig.update_layout(
        title= "Délais moyen de traitement (secondes)",
        xaxis_title="Créneau horaire",
        yaxis_title="DMT",
        legend_title="DMT",
        font=dict(
            size=14,
            color="black"
            ))           
    
    st.plotly_chart(fig, use_container_width=True)      
    
    
    
st.subheader("👉 CHIFFRES CLE (TECH+CR)")
st.text('')

st.info("Pour information l'analyse actuellement affichée concerne la date du " + str(date_cache))

c1, c2, c3, c4 = st.columns((1, 1, 1, 3))     

with c1:
    
    st.markdown("**🔰 ITV**")
    st.markdown(str(nbr_itv) + " interventions")
                
with c2:

    st.markdown("**📞 AE**")              
    st.markdown(str(ae_temp_all['nb_ae'].sum()) + " AE reçus")
    st.markdown(str(round(tot_ae_all)) + " AE prévus")

    tx_prev = (tot_ae_all / ae_temp_all['nb_ae'].sum()) * 100
    st.markdown("Prevs vs réel = " + str(round(tx_prev,1)) + "%")

with c3:

    st.markdown("**📉 TR90**")   

    tr90 = int(ae_temp_all['cs_tr90'].iloc[[-1]])
    st.markdown("TR90 de la journée = " + str(tr90) + "%")

    df_tr90 = ae_temp_all.groupby('creneau').agg(
        nb_ae_tr90 = ('nb_ae_tr90','sum'), nb_ae = ('nb_ae','sum')) 
    
    df_tr90['TR90'] = (df_tr90['nb_ae_tr90'] / df_tr90['nb_ae'] * 100)
    
    nb_bad_creneau_tr = df_tr90.loc[df_tr90['TR90'] < 70].shape[0]
    tx_nb_bad_creneau_tr = round((nb_bad_creneau_tr / 24) * 100,1)
    st.markdown("Dont " + str(nb_bad_creneau_tr) + " créneaux < 70%")
    st.markdown("soit " + str(tx_nb_bad_creneau_tr) + "% de la journée")






    
    
    
    

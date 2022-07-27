#Import des packages 

import streamlit as st
st.set_page_config(page_title="TISSEO FLUX", page_icon="🔎", layout="wide")

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

st.sidebar.subheader("🛠️ Filtres")

st.sidebar.text("")
jsem = st.sidebar.radio("🔧 JOUR DE LA SEMAINE", key = 1, options = ('Lundi', 'Mardi', 'Mercredi', 'Jeudi', 'Vendredi', 'Samedi'), horizontal = True)   

st.sidebar.text("")
flux = st.sidebar.radio("🔧 TYPE DE FLUX", key = 2, options = ('ALL', 'TECH', 'CR'), horizontal = True) 

st.sidebar.text("")
nb_jours = st.sidebar.radio("🔧 NOMBRE DE " + jsem, key = 3, options = (4, 6, 8), horizontal = True)  

st.sidebar.text("")
type_bts = st.sidebar.radio("🔧 PERIODE", key = 4, options = ("Hors BTS" , "Période BTS"), horizontal = True)

#########
#########
#########
#########

# TABLEAU RECAP

#########
#########
#########
#########

st.subheader("👉 SYNTHESE : FLUX ENTRANT ET TEMPS DE REPONSE")
st.text('')       

#Selection dataframe sur jour selectionné   
def dataframe_temp_synthese_flux(flux, jsem, nb_jours, type_bts):

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

    df_temp = df_temp[['Date', 'jsem', 'creneau', 'nb_ae', 'dmr', 'TR90', 'cs_ae_rep', 'cs_tr90', 'BTS']]
    df_temp = df_temp.merge(right = itv, on = ['Date'], how = 'left')
    df_temp['ae_par_itv'] = round((df_temp['nb_ae'] / df_temp['nb_itv']),4)
    df_temp.dropna(axis = 0, inplace = True, how = 'any', subset = ['nb_itv'])
    df_temp.reset_index(inplace = True)
    df_temp = df_temp.loc[df_temp['creneau'] != '20:00']
    df_temp.set_index('creneau', inplace = True)
    df_temp.fillna(0, axis = 1, inplace = True)                

    return df_temp

c1, c2, c3 = st.columns((1, 1, 1))     

with c1:
    df_initial_all = dataframe_temp_synthese_flux(flux = "ALL", jsem = jsem, nb_jours = nb_jours, type_bts = type_bts)      

    #Fonction percentiles
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_           

    df_tr90 = df_initial_all.groupby('creneau').agg(
        med_tr = ('TR90','median'))    

    df_dmr = df_initial_all.groupby('creneau').agg(
        med_dmr = ('dmr','median'))       

    df_ae = df_initial_all.groupby('creneau').agg(
        med_ae = ('ae_par_itv','median')) 

    df_ae['Var_fe'] = ((df_ae['med_ae'] - df_ae['med_ae'].shift(1)) / df_ae['med_ae'].shift(1))*100
    df_ae['Var_fe'].fillna(0, inplace = True)
    df_ae['Var_fe_prct'] = round(df_ae['Var_fe'],0)
    df_ae['Var_fe_prct'] = df_ae['Var_fe_prct'].astype(str)
    df_ae['Var_fe_prct'] = df_ae['Var_fe_prct']+"%"  

    df_final = pd.concat([df_tr90, df_dmr, df_ae], axis = 1)  

    df_final['FE'] = pd.cut(df_final['med_ae'], bins = [0, 0.08, 0.11, 10], labels = ['Standard', 'Important', 'Elevé'])    
    df_final['TR90'] = pd.cut(df_final['med_tr'], bins = [0, 70, 80, 100], labels = ['Attention', 'Vigilance', 'OK'])
    df_final['DMR'] = pd.cut(df_final['med_dmr'], bins = [0, 60, 90, 1000], labels = ['OK', 'Vigilance', 'Attention'])
    df_final['VAR'] = pd.cut(df_final['Var_fe'], bins = [-10000, -30, -15, 0, 15, 30, 10000], 
                             labels = ['baisseFo', 'baisseIm', 'baisseFa', 'hausseFa', 'hausseIm', 'hausseFo'])        

    cols = list(df_final.columns)
    cols2 = cols[3:] 

    fig = go.Figure(data=[go.Table(
        columnwidth = 10,            
        header=dict(values=['Creneau', 'Var. FE', 'FE', 'TR90', 'DMR'],
                    fill_color='darkblue',
                    align='center', 
                    font=dict(color='white', size=12)),
        cells=dict(values=[df_final.index, df_final.Var_fe_prct, df_final.FE, df_final.TR90, df_final.DMR],                       
                   #fill_color='whitesmoke',
                   fill = dict(color= ['rgb(245, 245, 245)',
                                       'white',
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df_final.FE],                                        
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else                                            
                                            'rgb(245, 245, 245)' for tr in df_final.TR90],                                            
                                           ['bisque' if dmr == "Vigilance" else 
                                            'lightcoral' if dmr == "Attention" else
                                            'palegreen' if dmr == "OK" else                                              
                                            'rgb(245, 245, 245)' for dmr in df_final.DMR]                                    
                                           ]),
                   font=dict(color=['black', 
                                           ['darkblue' if fe == "baisseFo" else 
                                            'mediumblue' if fe == "baisseIm" else
                                            'blue' if fe == "baisseFa" else
                                            'goldenrod' if fe == "hausseFa" else
                                            'orangered' if fe == "hausseIm" else  
                                            'red' if fe == "hausseFo" else  
                                            'rgb(245, 245, 245)' for fe in df_final.VAR],
                                    'black', 'black', 'black'], size=13),
                   align='center', 
                   height=25))
    ], 
                   layout=go.Layout(height=700, width=480))

    fig.update_layout(title= "TECH + CR", title_x=0.5, font=dict(size=14, color="black")) 

    fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))

    st.plotly_chart(fig, use_container_width=False)  

with c2:
    df_initial_tech = dataframe_temp_synthese_flux(flux = "TECH", jsem = jsem, nb_jours = nb_jours, type_bts = type_bts)      

    #Fonction percentiles
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_           

    df_tr90 = df_initial_tech.groupby('creneau').agg(
        med_tr = ('TR90','median'))    

    df_dmr = df_initial_tech.groupby('creneau').agg(
        med_dmr = ('dmr','median'))       

    df_ae = df_initial_tech.groupby('creneau').agg(
        med_ae = ('ae_par_itv','median')) 

    df_ae['Var_fe'] = ((df_ae['med_ae'] - df_ae['med_ae'].shift(1)) / df_ae['med_ae'].shift(1))*100
    df_ae['Var_fe'].fillna(0, inplace = True)
    df_ae['Var_fe_prct'] = round(df_ae['Var_fe'],0)
    df_ae['Var_fe_prct'] = df_ae['Var_fe_prct'].astype(str)
    df_ae['Var_fe_prct'] = df_ae['Var_fe_prct']+"%"  

    df_final = pd.concat([df_tr90, df_dmr, df_ae], axis = 1)  

    df_final['FE'] = pd.cut(df_final['med_ae'], bins = [0, 0.04, 0.055, 10], labels = ['Standard', 'Important', 'Elevé'])    
    df_final['TR90'] = pd.cut(df_final['med_tr'], bins = [0, 70, 80, 100], labels = ['Attention', 'Vigilance', 'OK'])
    df_final['DMR'] = pd.cut(df_final['med_dmr'], bins = [0, 60, 90, 1000], labels = ['OK', 'Vigilance', 'Attention'])
    df_final['VAR'] = pd.cut(df_final['Var_fe'], bins = [-10000, -30, -15, 0, 15, 30, 10000], 
                             labels = ['baisseFo', 'baisseIm', 'baisseFa', 'hausseFa', 'hausseIm', 'hausseFo'])        

    cols = list(df_final.columns)
    cols2 = cols[3:] 

    fig = go.Figure(data=[go.Table(
        columnwidth = 10,            
        header=dict(values=['Creneau', 'Var. FE', 'FE', 'TR90', 'DMR'],
                    fill_color='darkblue',
                    align='center', 
                    font=dict(color='white', size=12)),
        cells=dict(values=[df_final.index, df_final.Var_fe_prct, df_final.FE, df_final.TR90, df_final.DMR],                       
                   #fill_color='whitesmoke',
                   fill = dict(color= ['rgb(245, 245, 245)',
                                       'white',
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df_final.FE],                                        
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else                                            
                                            'rgb(245, 245, 245)' for tr in df_final.TR90],                                            
                                           ['bisque' if dmr == "Vigilance" else 
                                            'lightcoral' if dmr == "Attention" else
                                            'palegreen' if dmr == "OK" else                                              
                                            'rgb(245, 245, 245)' for dmr in df_final.DMR]                                    
                                           ]),
                   font=dict(color=['black', 
                                           ['darkblue' if fe == "baisseFo" else 
                                            'mediumblue' if fe == "baisseIm" else
                                            'blue' if fe == "baisseFa" else
                                            'goldenrod' if fe == "hausseFa" else
                                            'orangered' if fe == "hausseIm" else  
                                            'red' if fe == "hausseFo" else  
                                            'rgb(245, 245, 245)' for fe in df_final.VAR],
                                    'black', 'black', 'black'], size=13),
                   align='center', 
                   height=25))
    ], 
                   layout=go.Layout(height=700, width=480))

    fig.update_layout(title= "TECH SEUL", title_x=0.5, font=dict(size=14, color="black"))

    fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20)) 

    st.plotly_chart(fig, use_container_width=False)        

with c3:
    df_initial_cr = dataframe_temp_synthese_flux(flux = "CR", jsem = jsem, nb_jours = nb_jours, type_bts = type_bts)      

    #Fonction percentiles
    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_           

    df_tr90 = df_initial_cr.groupby('creneau').agg(
        med_tr = ('TR90','median'))    

    df_dmr = df_initial_cr.groupby('creneau').agg(
        med_dmr = ('dmr','median'))       

    df_ae = df_initial_cr.groupby('creneau').agg(
        med_ae = ('ae_par_itv','median')) 

    df_ae['Var_fe'] = ((df_ae['med_ae'] - df_ae['med_ae'].shift(1)) / df_ae['med_ae'].shift(1))*100
    df_ae['Var_fe'].fillna(0, inplace = True)
    df_ae['Var_fe_prct'] = round(df_ae['Var_fe'],0)
    df_ae['Var_fe_prct'] = df_ae['Var_fe_prct'].astype(str)
    df_ae['Var_fe_prct'] = df_ae['Var_fe_prct']+"%"  

    df_final = pd.concat([df_tr90, df_dmr, df_ae], axis = 1)  

    df_final['FE'] = pd.cut(df_final['med_ae'], bins = [0, 0.04, 0.055, 10], labels = ['Standard', 'Important', 'Elevé'])    
    df_final['TR90'] = pd.cut(df_final['med_tr'], bins = [0, 70, 80, 100], labels = ['Attention', 'Vigilance', 'OK'])
    df_final['DMR'] = pd.cut(df_final['med_dmr'], bins = [0, 60, 90, 1000], labels = ['OK', 'Vigilance', 'Attention'])
    df_final['VAR'] = pd.cut(df_final['Var_fe'], bins = [-10000, -30, -15, 0, 15, 30, 10000], 
                             labels = ['baisseFo', 'baisseIm', 'baisseFa', 'hausseFa', 'hausseIm', 'hausseFo'])        

    cols = list(df_final.columns)
    cols2 = cols[3:] 

    fig = go.Figure(data=[go.Table(
        columnwidth = 10,            
        header=dict(values=['Creneau', 'Var. FE', 'FE', 'TR90', 'DMR'],
                    fill_color='darkblue',
                    align='center', 
                    font=dict(color='white', size=12)),
        cells=dict(values=[df_final.index, df_final.Var_fe_prct, df_final.FE, df_final.TR90, df_final.DMR],                       
                   #fill_color='whitesmoke',
                   fill = dict(color= ['rgb(245, 245, 245)',
                                       'white',
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df_final.FE],                                        
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else                                            
                                            'rgb(245, 245, 245)' for tr in df_final.TR90],                                            
                                           ['bisque' if dmr == "Vigilance" else 
                                            'lightcoral' if dmr == "Attention" else
                                            'palegreen' if dmr == "OK" else                                              
                                            'rgb(245, 245, 245)' for dmr in df_final.DMR]                                    
                                           ]),
                   font=dict(color=['black', 
                                           ['darkblue' if fe == "baisseFo" else 
                                            'mediumblue' if fe == "baisseIm" else
                                            'blue' if fe == "baisseFa" else
                                            'goldenrod' if fe == "hausseFa" else
                                            'orangered' if fe == "hausseIm" else  
                                            'red' if fe == "hausseFo" else  
                                            'rgb(245, 245, 245)' for fe in df_final.VAR],
                                    'black', 'black', 'black'], size=13),
                   align='center', 
                   height=25))
    ], 
                   layout=go.Layout(height=700, width=480))

    fig.update_layout(title= "CR SEUL", title_x=0.5, font=dict(size=14, color="black")) 

    fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))        

    st.plotly_chart(fig, use_container_width=False)  
    
    
#########
#########
#########
#########

# VARIATIONS DU FLUX ENTRANT (FE)

#########
#########
#########
#########

st.subheader("👉 VARIATIONS DU FLUX ENTRANT (FE)")
st.text('') 
st.text('')           


#Selection dataframe sur jour selectionné   
def dataframe_temp_synthese(flux, jsem, nb_jours, type_bts):

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

df_initial = dataframe_temp_synthese(flux = flux, jsem = jsem, nb_jours = nb_jours, type_bts = type_bts)

#Fonction percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_           

df_fe = df_initial.groupby('creneau').agg(
    p20 = ('ae_par_itv',percentile(20)),        
    med_ae = ('ae_par_itv','median'), 
    p80 = ('ae_par_itv',percentile(80)))      

for i in df_initial['Date'].unique():
    df_fe_temp = df_initial[df_initial['Date']==i]
    df_fe_temp.reset_index(inplace = True)
    df_fe_temp = df_fe_temp[['creneau', 'ae_par_itv']]
    df_fe_temp.set_index('creneau', inplace = True)
    df_fe = pd.concat([df_fe, df_fe_temp.reindex(df_fe.index)], axis = 1, join = 'inner')
    df_fe.rename({'ae_par_itv':str(i)[0:10]}, axis = 1, inplace = True)

if flux == 'ALL':
    df_fe = df_fe.assign(important=0.08)   
    df_fe = df_fe.assign(élevé=0.11)
else:
    df_fe = df_fe.assign(important=0.04)   
    df_fe = df_fe.assign(élevé=0.055)        

my_expander = st.expander("Graphique", expanded=True)
with my_expander:        

    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))


    fig.add_trace(go.Scatter(x=df_fe.index, y=df_fe['p80'], 
                             line=dict(color='darkblue', width=0.8, dash = 'dot'),
                             name='80p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)

    fig.add_trace(go.Scatter(x=df_fe.index, y=df_fe['med_ae'], 
                            fill='tonexty', fillcolor='white',  
                            line=dict(color='darkblue', width=6), 
                            mode='lines+markers',
                            marker=dict(size=15, color='midnightblue', line = dict(color='black', width = 2)), 
                            name='Réel'
                                ))
    fig.update_traces(hovertemplate=None) 

    fig.add_trace(go.Scatter(x=df_fe.index, y=df_fe['p20'], 
                             fill='tonexty', fillcolor='white',                          
                             line=dict(color='darkblue', width=0.8, dash = 'dot'),
                             name='20p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)    

    fig.add_trace(go.Scatter(x=df_fe.index, y=df_fe['important'],
                        line=dict(color='orangered', width=1.5, dash = 'dot'),                 
                        name='important'
                        ))  

    fig.add_trace(go.Scatter(x=df_fe.index, y=df_fe['élevé'],
                    line=dict(color='red', width=3, dash = 'dot'),                 
                    name='élevé'
                    ))               

    fig.update_layout(hovermode="x")

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 
    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)

    if flux == 'ALL':
        fig.update_yaxes(range=[0, 0.16])
    else:      
        fig.update_yaxes(range=[0, 0.09])

    fig.update_layout(
        title= "Variations du flux entrant (FE)",
        xaxis_title="Créneau horaire",
        yaxis_title="Appels par ITV",
        legend_title="Appels par ITV",
        font=dict(
            size=14,
            color="black"
            ))   

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))         

    st.plotly_chart(fig, use_container_width=True) 

my_expander = st.expander("Tableau de données", expanded=False)
with my_expander:

    #Partie des tableaux

    #suppression des colonnes qui ne servent pas 
    #Creation d'un dataframe qui servira a créer les bins
    #Cols2 servira pour les noms de colonnes des tableaux
    df_fe_tab = df_fe.drop(['p80', 'p20'], axis = 1)        
    cols = list(df_fe_tab.columns)
    cols2 = cols[1:-2]

    #suppression des colonnes qui ne servent pas 
    #Creation du dataframe qui contiendra les valeurs
    df_fe_tab_temp = df_fe.drop(['p80', 'p20'], axis = 1)   
    cols_temp = list(df_fe_tab_temp.columns)
    cols2_temp = cols_temp[1:-2]       

    #Creation des bins en fonction de la valeur de la variable
    #Utilisation du 1er dataframe
    if flux == 'ALL':
        vmin = 0.08
        vmax = 0.11
        for i in cols:
            df_fe_tab[i] = pd.cut(df_fe_tab[i], bins = [0, vmin, vmax, 10], labels = ['Standard', 'Important', 'Elevé'])
    else:
        vmin = 0.08 / 2
        vmax = 0.11 /2       
        for i in cols:
            df_fe_tab[i] = pd.cut(df_fe_tab[i], bins = [0, vmin, vmax, 10], labels = ['Standard', 'Important', 'Elevé'])            

    #Renommage des colonnes pour les 2 Dataframe
    #Afin de simplifier la création des graph car les noms de colonnes (les dates) changent en permanence
    newcols = []
    val_int = 1
    for i in cols2:
        val = "J"+str(val_int)
        val_int += 1
        newcols.append(val)

    #Renommage des colonnes pour les 2 Dataframe
    #Afin de simplifier la création des graph car les noms de colonnes (les dates) changent en permanence            
    newcols_temp = []
    val_int = 1
    for i in cols2_temp:
        val = "J"+str(val_int)
        val_int += 1
        newcols_temp.append(val)          

    #Renommage
    df = df_fe_tab[cols2].rename(columns=dict(zip(cols2, newcols)))  
    df1 = df_fe_tab_temp[cols2_temp].rename(columns=dict(zip(cols2_temp, newcols_temp)))
    df1 = round(df1, 3)

    #Creation d'un tableau si nb_jours sélectionné = 4
    if nb_jours == 4:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols2[0], cols2[1], cols2[2], cols2[3], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'royalblue',                                        
                                                 'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[df_fe_tab.index, df1.J1, df1.J2, df1.J3, df1.J4, 
                               df_fe_tab.index, round(df_fe_tab_temp.med_ae,3)],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J1], 
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J2],  
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J3],                                               
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J4],                                               
                                           'rgb(245, 245, 245)',
                                          ['bisque' if fe == "Important" else 
                                           'lightcoral' if fe == "Elevé" else
                                           'palegreen' if fe == "Standard" else
                                           'rgb(245, 245, 245)' for fe in df_fe_tab.med_ae],                                             
                                           ]),                                           
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=700))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black"))  

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))            

        st.plotly_chart(fig, use_container_width=False)     

    #Creation d'un tableau si nb_jours sélectionné = 6
    if nb_jours == 6:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,  
            header=dict(values=['Creneau', cols2[0], cols2[1], cols2[2], cols2[3], cols2[4], cols2[5], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue',
                                                 'royalblue',                                        
                                                 'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[df_fe_tab.index, df1.J1, df1.J2, df1.J3, df1.J4, df1.J5, 
                               df1.J6, df_fe_tab.index, round(df_fe_tab_temp.med_ae,3)],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J1], 
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J2],  
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J3],                                               
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J4],                                               
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J5],                                               
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J6], 
                                           'rgb(245, 245, 245)',
                                          ['bisque' if fe == "Important" else 
                                           'lightcoral' if fe == "Elevé" else
                                           'palegreen' if fe == "Standard" else
                                           'rgb(245, 245, 245)' for fe in df_fe_tab.med_ae],                                             
                                           ]),                                           
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=800))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black"))

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)  

    #Creation d'un tableau si nb_jours sélectionné = 8
    if nb_jours == 8:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols2[0], cols2[1], cols2[2], cols2[3], 
                                cols2[4], cols2[5], cols2[6], cols2[7], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue', 
                                                 'darkblue',  
                                                 'darkblue', 
                                                 'darkblue',   
                                                 'royalblue',                                         
                                                 'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[df_fe_tab.index, df1.J1, df1.J2, df1.J3, df1.J4, df1.J5, 
                               df1.J6, df1.J7, df1.J8, df_fe_tab.index,
                               round(df_fe_tab_temp.med_ae,3)],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J1], 
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J2],  
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J3],                                               
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J4],                                               
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J5],                                               
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J6],  
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J7], 
                                           ['bisque' if fe == "Important" else 
                                            'lightcoral' if fe == "Elevé" else
                                            'palegreen' if fe == "Standard" else
                                            'rgb(245, 245, 245)' for fe in df.J8], 
                                           'rgb(245, 245, 245)',
                                          ['bisque' if fe == "Important" else 
                                           'lightcoral' if fe == "Elevé" else
                                           'palegreen' if fe == "Standard" else
                                           'rgb(245, 245, 245)' for fe in df_fe_tab.med_ae],                                             
                                           ]),                                           
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=900))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black")) 

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)              

st.text('')
st.text('')

#########
#########
#########
#########

# VARIATIONS DU TEMPS DE REPONSE (TR90s)

#########
#########
#########
#########
 

st.subheader("👉 VARIATIONS DU TEMPS DE REPONSE (TR90s)")
st.text('')
st.text('')           


#Selection dataframe sur jour selectionné   
def dataframe_temp(flux, jsem, type_bts, nb_jours):

    #selection du flux
    if flux == 'ALL':
        ae_temp = ae_all
    if flux == 'TECH':
        ae_temp = ae_tech         
    if flux == 'CR':
        ae_temp = ae_cr

    #selection du jour de la semaine
    ae_temp = ae_temp.loc[ae_temp['jsem']==jsem]

    #selection de la période
    if type_bts == "Hors BTS" :
        ae_temp = ae_temp.loc[ae_temp['BTS']==False]
    if type_bts == "Période BTS" :
        ae_temp = ae_temp.loc[ae_temp['BTS']==True]

    #Selection des dates à prendre en compte pour les abaques en fonction du nb de jour souhaité
    max_date_temp = ae_temp['Date'].unique()[-1]
    if str(nb_jours) != "All":
        liste = list(ae_temp['Date'].unique())
        max_date_index = liste.index(max_date_temp)
        liste_jour_sel = []
        for i in np.arange(1 , int(nb_jours) + 1, 1):  
            liste_jour_sel.append(liste[max_date_index - i])
        ae_temp = ae_temp[ae_temp['Date'].isin(liste_jour_sel)]    

    return ae_temp

#PARTIE TR90 
df_tr90 = dataframe_temp(flux = flux, jsem = jsem, type_bts = type_bts, nb_jours = nb_jours)

df_tr90_tr = df_tr90[['Date', 'creneau', 'TR90', 'cs_tr90']]
df_tr90_tr = df_tr90_tr.loc[df_tr90_tr['creneau'] != '20:00']    

#Fonction percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_           

df_tr90_tr_final = df_tr90_tr.groupby('creneau').agg(
    p20 = ('TR90',percentile(20)),
    med_tr = ('TR90','median'), 
    p80 = ('TR90',percentile(80)),
    med_cs = ("cs_tr90", 'median'))    

for i in df_tr90_tr['Date'].unique():
    df_tr90_tr_temp = df_tr90_tr[df_tr90_tr['Date']==i]
    df_tr90_tr_temp = df_tr90_tr_temp[['creneau', 'TR90']]
    df_tr90_tr_temp.set_index('creneau', inplace = True)
    df_tr90_tr_final = pd.concat([df_tr90_tr_final, df_tr90_tr_temp.reindex(df_tr90_tr_final.index)], axis = 1, join = 'inner')
    df_tr90_tr_final.rename({'TR90':str(i)[0:10]}, axis = 1, inplace = True)

df_tr90_tr_final.replace(to_replace=0, method='ffill', inplace = True)                  
df_tr90_tr_final.fillna(0, axis = 1, inplace = True)
df_tr90_tr_final = df_tr90_tr_final.astype(int) 
df_tr90_tr_final = df_tr90_tr_final.assign(obj=80)
df_tr90_tr_final = df_tr90_tr_final.assign(obj_min=70)   

my_expander = st.expander("Graphique", expanded=True)
with my_expander:      

    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))


    fig.add_trace(go.Scatter(x=df_tr90_tr_final.index, y=df_tr90_tr_final['p80'], 
                             line=dict(color='darkgreen', width=0.8, dash = 'dot'),
                             name='80p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None) 

    fig.add_trace(go.Scatter(x=df_tr90_tr_final.index, y=df_tr90_tr_final['med_tr'], 
                            fill='tonexty', fillcolor='white',  
                            line=dict(color='green', width=6), 
                            mode='lines+markers',
                            marker=dict(size=15, color='green', line = dict(color='black', width = 2)),                        
                            name='med'
                            ))
    fig.update_traces(hovertemplate=None) 

    fig.add_trace(go.Scatter(x=df_tr90_tr_final.index, y=df_tr90_tr_final['p20'], 
                             fill='tonexty', fillcolor='white', 
                             line=dict(color='darkgreen', width=0.8, dash = 'dot'),
                             name='20p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)     


    fig.add_trace(go.Scatter(x=df_tr90_tr_final.index, y=df_tr90_tr_final['obj'],
                        line=dict(color='orangered', width=1.5, dash = 'dot'),                 
                        name='obj'
                        ))  

    fig.add_trace(go.Scatter(x=df_tr90_tr_final.index, y=df_tr90_tr_final['obj_min'],
                    line=dict(color='red', width=3, dash = 'dot'),                 
                    name='obj min'
                    ))   

    #fig.add_trace(go.Scatter(x=df_tr90_tr_final.index, y=df_tr90_tr_final['med_cs'],
    #                line=dict(color='darkblue', width=6), 
    #                mode='lines+markers',
    #                marker=dict(size=15, color='darkblue', line = dict(color='black', width = 2)),                        
    #                name='TR90 cumulé', opacity = 0.2
    #                ))              

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 

    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_yaxes(range=[40, 110])

    fig.update_layout(
        title= "Variations du TR90s en fonction des jours, période et flux",
        xaxis_title="Créneau horaire",
        yaxis_title="TR90s",
        legend_title="TR90s",
        font=dict(
            size=14,
            color="black"
            )) 

    fig.update_layout(hovermode="x")

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))            

    st.plotly_chart(fig, use_container_width=True)              


my_expander = st.expander("Tableau de données", expanded=False)
with my_expander:

    def dataframe_tableau_bins(df_input):
        #Partie des tableaux        
        #suppression des colonnes qui ne servent pas 
        #Creation d'un dataframe qui servira a créer les bins
        df_bins = df_input.drop(['p80', 'p20', 'med_cs', 'obj', 'obj_min'], axis = 1)  

        #Creation des bins en fonction de la valeur de la variable
        vmin = 70
        vmax = 80
        cols = list(df_bins.columns)

        for i in cols:
            df_bins[i] = pd.cut(df_bins[i], bins = [0, vmin, vmax, 110], labels = ['Attention', 'Vigilance', 'OK'])

        #Renommage des colonnes pour les 2 Dataframe
        #Afin de simplifier la création des graph car les noms de colonnes (les dates) changent en permanence
        cols = cols[1:]
        newcols = []
        val_int = 1
        for i in cols:
            val = "J"+str(val_int)
            val_int += 1
            newcols.append(val)    

        #Renommage et creation du dataframe df pour les valeurs de bins
        dfb = df_bins.rename(columns=dict(zip(cols, newcols)))      

        #Partie des tableaux        
        #suppression des colonnes qui ne servent pas 
        #Creation d'un dataframe qui servira a utiliser les variables              
        df_vals = df_input.drop(['p80', 'p20', 'med_cs', 'obj', 'obj_min'], axis = 1)

        #Renommage et creation du dataframe df1 pour les valeurs
        dfv = df_vals.rename(columns=dict(zip(cols, newcols)))     

        return dfb, dfv, cols


    dfb, dfv, cols = dataframe_tableau_bins(df_tr90_tr_final)

    #Creation d'un tableau si nb_jours sélectionné = 4
    if nb_jours == 4:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols[0], cols[1], cols[2], cols[3], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'royalblue',                                        
                                     'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[dfv.index, dfv.J1, dfv.J2, dfv.J3, dfv.J4, 
                               dfv.index, round(dfv.med_tr,3)],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J1], 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J2],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J3],                                               
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J4],                                               
                                           'rgb(245, 245, 245)',
                                          ['bisque' if tr == "Vigilance" else 
                                           'lightcoral' if tr == "Attention" else
                                           'palegreen' if tr == "OK" else
                                           'rgb(245, 245, 245)' for tr in dfb.med_tr],                                             
                                           ]),                                         
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=700))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black"))  

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)   

    #Creation d'un tableau si nb_jours sélectionné = 6
    if nb_jours == 6:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue',                                         
                                     'royalblue',                                        
                                     'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[dfv.index, dfv.J1, dfv.J2, dfv.J3, dfv.J4, dfv.J5, dfv.J6,
                               dfv.index, round(dfv.med_tr,3)],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J1], 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J2],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J3],                                               
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J4],   
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J5],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J6],                                                 
                                           'rgb(245, 245, 245)',
                                          ['bisque' if tr == "Vigilance" else 
                                           'lightcoral' if tr == "Attention" else
                                           'palegreen' if tr == "OK" else
                                           'rgb(245, 245, 245)' for tr in dfb.med_tr],                                             
                                           ]),                                         
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=800))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black")) 

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)        

    #Creation d'un tableau si nb_jours sélectionné = 8
    if nb_jours == 8:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols[0], cols[1], cols[2], cols[3], cols[4], cols[5],
                                cols[6], cols[7], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue',
                                     'darkblue', 
                                     'darkblue',                                         
                                     'royalblue',                                        
                                     'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[dfv.index, dfv.J1, dfv.J2, dfv.J3, dfv.J4, dfv.J5, dfv.J6,
                               dfv.J7, dfv.J7, dfv.index, round(dfv.med_tr,3)],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J1], 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J2],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J3],                                               
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J4],   
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J5],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J6],                                                 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J7],
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J8],                                                
                                           'rgb(245, 245, 245)',
                                          ['bisque' if tr == "Vigilance" else 
                                           'lightcoral' if tr == "Attention" else
                                           'palegreen' if tr == "OK" else
                                           'rgb(245, 245, 245)' for tr in dfb.med_tr],                                             
                                           ]),                                         
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=900))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black"))  

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)                                 

st.text('')
st.text('')

#########
#########
#########
#########

# VARIATION DES DELAIS D'ATTENTE (DMR)

#########
#########
#########
#########

st.subheader("👉 VARIATION DES DELAIS D'ATTENTE (DMR)")
st.text('')
st.text('')           



#Selection dataframe sur jour selectionné   
def dataframe_temp(flux, jsem, type_bts, nb_jours):

    #selection du flux
    if flux == 'ALL':
        ae_temp = ae_all
    if flux == 'TECH':
        ae_temp = ae_tech         
    if flux == 'CR':
        ae_temp = ae_cr

    #selection du jour de la semaine
    ae_temp = ae_temp.loc[ae_temp['jsem']==jsem]

    #selection de la période
    if type_bts == "Hors BTS" :
        ae_temp = ae_temp.loc[ae_temp['BTS']==False]
    if type_bts == "Période BTS" :
        ae_temp = ae_temp.loc[ae_temp['BTS']==True]

    #Selection des dates à prendre en compte pour les abaques en fonction du nb de jour souhaité
    max_date_temp = ae_temp['Date'].unique()[-1]
    if str(nb_jours) != "All":
        liste = list(ae_temp['Date'].unique())
        max_date_index = liste.index(max_date_temp)
        liste_jour_sel = []
        for i in np.arange(1 , int(nb_jours) + 1, 1):  
            liste_jour_sel.append(liste[max_date_index - i])
        ae_temp = ae_temp[ae_temp['Date'].isin(liste_jour_sel)]    

    return ae_temp

#PARTIE TR90 
df_dmr = dataframe_temp(flux = flux, jsem = jsem, type_bts = type_bts, nb_jours = nb_jours)                  

df_dmr = df_dmr [['Date', 'creneau', 'nb_ae', 'dmr']]
df_dmr = df_dmr.loc[df_dmr['creneau'] != '20:00']    

#Fonction percentiles
def percentile(n):
    def percentile_(x):
        return np.percentile(x, n)
    percentile_.__name__ = 'percentile_%s' % n
    return percentile_       

df_dmr_final = df_dmr.groupby('creneau').agg(
    p20 = ('dmr',percentile(20)),
    med_dmr = ('dmr','median'), 
    p80 = ('dmr',percentile(80)),
    med_ae = ("nb_ae", 'median'))

for i in df_dmr['Date'].unique():
    df_dmr_temp = df_dmr[df_dmr['Date']==i]
    df_dmr_temp = df_dmr_temp[['creneau', 'dmr']]
    df_dmr_temp.set_index('creneau', inplace = True)
    df_dmr_final = pd.concat([df_dmr_final, df_dmr_temp.reindex(df_dmr_final.index)], axis = 1, join = 'inner')
    df_dmr_final.rename({'dmr':str(i)[0:10]}, axis = 1, inplace = True)

df_dmr_final.replace(to_replace=0, method='ffill', inplace = True)                  
df_dmr_final.fillna(0, axis = 1, inplace = True)
df_dmr_final = df_dmr_final.astype(int) 
df_dmr_final = df_dmr_final.assign(attention=90)  
df_dmr_final = df_dmr_final.assign(vigilance=60)    

my_expander = st.expander("Graphique", expanded=True)
with my_expander:         

    ##Fig
    fig = go.Figure(layout=go.Layout(height=600, width=1000, plot_bgcolor='whitesmoke'))  

    fig.add_trace(go.Scatter(x=df_dmr_final.index, y=df_dmr_final['p80'], 
                             line=dict(color='darkorange', width=0.8, dash = 'dot'),
                             name='80p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None) 

    fig.add_trace(go.Scatter(x=df_dmr_final.index, y=df_dmr_final['med_dmr'], 
                            fill='tonexty', fillcolor='white',  
                            line=dict(color='orange', width=6), 
                            mode='lines+markers',
                            marker=dict(size=10, color='orange', line = dict(color='orange', width = 6)),                     
                            name='med'
                            ))      

    fig.add_trace(go.Scatter(x=df_dmr_final.index, y=df_dmr_final['p20'], 
                             fill='tonexty', fillcolor='white', 
                             line=dict(color='darkorange', width=0.8, dash = 'dot'),
                             name='20p', opacity = 0.01)) 
    fig.update_traces(hovertemplate=None)                

    fig.add_trace(go.Scatter(x=df_dmr_final.index, y=df_dmr_final['vigilance'],
                        line=dict(color='orangered', width=1.5, dash = 'dot'),                 
                        name='Vigilance'
                        ))  

    fig.add_trace(go.Scatter(x=df_dmr_final.index, y=df_dmr_final['attention'],
                    line=dict(color='red', width=3, dash = 'dot'),                 
                    name='Attention'
                    ))  

    fig.update_xaxes(tickfont=dict(size=16))   
    fig.update_yaxes(tickfont=dict(size=16)) 

    fig.update_yaxes(showgrid=False, zeroline=False, showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_xaxes(showline=True, linewidth=1.5, linecolor='black', mirror=True)
    fig.update_yaxes(range=[0, 200])

    fig.update_layout(
        title= "Variations de la DMR",
        xaxis_title="Créneau horaire",
        yaxis_title="DMR",
        legend_title="DMR",
        font=dict(
            size=14,
            color="black"
            ))

    fig.update_layout(hovermode="x")

    fig.update_layout(margin  = dict(l=10, r=10, t=40, b=10))    

    st.plotly_chart(fig, use_container_width=True)             

my_expander = st.expander("Tableau de données", expanded=False)
with my_expander:           

    def dataframe_tableau_bins_dmr(df_input):
        #Partie des tableaux        
        #suppression des colonnes qui ne servent pas 
        #Creation d'un dataframe qui servira a créer les bins
        df_bins = df_input.drop(['p80', 'p20', 'med_ae', 'vigilance', 'attention'], axis = 1)  

        #Creation des bins en fonction de la valeur de la variable
        vmin = 60
        vmax = 90
        cols = list(df_bins.columns)

        for i in cols:
            df_bins[i] = pd.cut(df_bins[i], bins = [0, vmin, vmax, 1000], labels = ['OK', 'Vigilance', 'Attention'])

        #Renommage des colonnes pour les 2 Dataframe
        #Afin de simplifier la création des graph car les noms de colonnes (les dates) changent en permanence
        cols = cols[1:]
        newcols = []
        val_int = 1
        for i in cols:
            val = "J"+str(val_int)
            val_int += 1
            newcols.append(val)    

        #Renommage et creation du dataframe df pour les valeurs de bins
        dfb = df_bins.rename(columns=dict(zip(cols, newcols)))      

        #Partie des tableaux        
        #suppression des colonnes qui ne servent pas 
        #Creation d'un dataframe qui servira a utiliser les variables              
        df_vals = df_input.drop(['p80', 'p20', 'med_ae', 'vigilance', 'attention'], axis = 1)

        #Renommage et creation du dataframe df1 pour les valeurs
        dfv = df_vals.rename(columns=dict(zip(cols, newcols)))     

        return dfb, dfv, cols

    dfb, dfv, cols = dataframe_tableau_bins_dmr(df_dmr_final)

    #Creation d'un tableau si nb_jours sélectionné = 4
    if nb_jours == 4:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols[0], cols[1], cols[2], cols[3], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'royalblue',                                        
                                     'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[dfv.index, dfv.J1, dfv.J2, dfv.J3, dfv.J4, 
                               dfv.index, dfv.med_dmr],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J1], 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J2],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J3],                                               
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J4],                                               
                                           'rgb(245, 245, 245)',
                                          ['bisque' if tr == "Vigilance" else 
                                           'lightcoral' if tr == "Attention" else
                                           'palegreen' if tr == "OK" else
                                           'rgb(245, 245, 245)' for tr in dfb.med_dmr],                                             
                                           ]),                                         
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=700))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black"))   

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)         

    #Creation d'un tableau si nb_jours sélectionné = 6
    if nb_jours == 6:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols[0], cols[1], cols[2], cols[3], cols[4], cols[5], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue',                                         
                                     'royalblue',                                        
                                     'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[dfv.index, dfv.J1, dfv.J2, dfv.J3, dfv.J4, dfv.J5, dfv.J6,
                               dfv.index, dfv.med_dmr],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J1], 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J2],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J3],                                               
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J4],   
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J5],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J6],                                                 
                                           'rgb(245, 245, 245)',
                                          ['bisque' if tr == "Vigilance" else 
                                           'lightcoral' if tr == "Attention" else
                                           'palegreen' if tr == "OK" else
                                           'rgb(245, 245, 245)' for tr in dfb.med_dmr],                                             
                                           ]),                                         
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=800))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black"))

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)        

    #Creation d'un tableau si nb_jours sélectionné = 8
    if nb_jours == 8:

        fig = go.Figure(data=[go.Table(
            columnwidth = 10,            
            header=dict(values=['Creneau', cols[0], cols[1], cols[2], cols[3], cols[4], cols[5],
                                cols[6], cols[7], 'Creneau', 'MED'],
                        fill_color=('royalblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue', 
                                     'darkblue',
                                     'darkblue', 
                                     'darkblue',                                         
                                     'royalblue',                                        
                                     'goldenrod'),
                        align='center', 
                        font=dict(color='white', size=12)),
            cells=dict(values=[dfv.index, dfv.J1, dfv.J2, dfv.J3, dfv.J4, dfv.J5, dfv.J6,
                               dfv.J7, dfv.J7, dfv.index, dfv.med_dmr],                       
                       fill = dict(color= ['rgb(245, 245, 245)',
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J1], 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J2],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J3],                                               
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J4],   
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J5],  
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J6],                                                 
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J7],
                                           ['bisque' if tr == "Vigilance" else 
                                            'lightcoral' if tr == "Attention" else
                                            'palegreen' if tr == "OK" else
                                            'rgb(245, 245, 245)' for tr in dfb.J8],                                                
                                           'rgb(245, 245, 245)',
                                          ['bisque' if tr == "Vigilance" else 
                                           'lightcoral' if tr == "Attention" else
                                           'palegreen' if tr == "OK" else
                                           'rgb(245, 245, 245)' for tr in dfb.med_dmr],                                             
                                           ]),                                         
                       font=dict(color='black', size=12),
                       align='center', 
                       height=25))
                             ], 
                       layout=go.Layout(height=700, width=900))

        title = "Données des "+str(nb_jours)+" derniers "+str(jsem)+" "+"("+str(type_bts)+")"

        fig.update_layout(title= title, title_x=0.5, font=dict(size=14, color="black"))   

        fig.update_layout(margin  = dict(l=60, r=60, t=40, b=20))               

        st.plotly_chart(fig, use_container_width=False)                                 


st.text('')
st.text('')


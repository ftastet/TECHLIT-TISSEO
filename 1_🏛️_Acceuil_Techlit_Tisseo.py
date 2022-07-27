import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

st.set_page_config(page_title="TISSEO ACCEUIL", page_icon="🏛️", layout="wide")

st.header("OBJECTIFS DE TECHLIT TISSEO")
    
st.markdown(""" 
        Cette interface à pour role de donner une vue sur les flux d'appels entrant techniciens (Tech & CR) par créneaux horaires (30 minutes)
        
        Afin de...
        - **Comprendre les variations de flux** en fonction type de jour et des périodes
        - **Identifier les variations de la qualité de réponse** en fonction type de jour et des périodes
        - **Augmenter la prise de recul** avec l'analyse des flux passés
        
        
        Avec pour objectifs d'aider à...
        - **Anticiper des variations importantes**
        - **Améliorer la planification en amont**
        - **Optimiser le planficiation à chaud** 
        - **Conserver une continuité dans la qualité du temps de réponse**
        - **Mettre en place des actions et mesure d'améliorations**                     
        """)
st.text('')
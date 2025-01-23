import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import os

apptitle = "SQuAD Visual Inspector"
st.set_page_config(page_title=apptitle, layout='wide',initial_sidebar_state='collapsed')
padding_top = 1
st.title("SDSS Spectral Window Calculator")


SDSS_MIN = 3600
SDSS_MAX = 10400

#####################################################################################
# READ THE VANDEN BERK COMPOSITE AND DEFINE WAVELENGTH WINDOW
#####################################################################################
wavelength = np.arange(1300,3001,2)
import numpy as np
import matplotlib.pyplot as plt

with open("Vanden_Berk.txt", encoding="utf-8-sig") as f:
    lines = f.readlines()

# Save cleaned data to a numpy array
data = np.loadtxt(lines)

# Extract wavelength, flux, and error
VB_Wavelength = np.array(data[:, 0])
VB_Flux = np.array(data[:, 1])

lines = {
    "Lyα": 1215.67,
    "C IV": 1549.48,
    "Mg II": 2798.75,
    "O III]": 1663.46,
    "Hβ": 4862.68,
    "Hα": 6564.61,
    "Si IV": 1397.00,
    "C III]": 1908.73,
    "Fe II": 2600.00,
    " ": 4959.00,
    "O III ": 5007,
    "H $\gamma$": 4340,
    "H $\delta$": 4102,
    "H $\epsilon$": 3970,
    "[Ne III]": 3869,
    "[O II]": 3727,
}

selected_scheme = st.sidebar.selectbox('Select the scheme',('I have the desired redshift window','I have the desired Wavelength window'))
#####################################################################################
if selected_scheme == 'I have the desired redshift window':
    with st.container():
        col1, col2 = st.columns([2, 2])
        with col1:
            z_min = st.number_input(
                'Minimum Redshift',
                min_value=0.00000,
                max_value=7.00000,
                value=0.00000,
                step=0.00001,  # Match the precision of the default value
                format="%.5f"  # Display five decimal places
            )
        with col2:
            z_max = st.number_input(
                'Maximum Redshift',
                min_value=0.00000,
                max_value=7.00000,
                value=0.00000,
                step=0.00001,  # Match the precision
                format="%.5f"  # Display five decimal places
            )

    a1 = SDSS_MIN/(1+z_min)
    a2 = SDSS_MIN/(1+z_max)
    wl_min = max(a1,a2)

    a1 = SDSS_MAX/(1+z_min)
    a2 = SDSS_MAX/(1+z_max)
    wl_max = min(a1,a2)

    with st.container():
        fig1,axt = plt.subplots(figsize=(8,3))
        axt.plot(VB_Wavelength,VB_Flux,lw=1)
        axt.axvspan(wl_min,wl_max,color='red',alpha=0.2)
        for label, wavelength in lines.items():
            axt.axvline(wavelength, color='blue', linestyle='--', alpha=0.7,lw=1)
            axt.text(wavelength, max(VB_Flux) * 0.8, label, rotation=90, fontsize=8, color='blue', ha='right', va='bottom')

        st.pyplot(fig1)

        fig2,axt = plt.subplots(figsize=(8,3))
        axt.set_title(f'$\lambda_i$: {round(wl_min,2)}\t\t\t\t\t\t$\lambda_f$: {round(wl_max,2)}')
        axt.plot(VB_Wavelength[int((wl_min-800)):int((wl_max-800))],VB_Flux[int((wl_min-800)):int((wl_max-800))],lw=1)
        axt.axvspan(wl_min,wl_max,color='red',alpha=0.2)
        for label, wavelength in lines.items():
            if wavelength > wl_min and wavelength < wl_max:
                axt.axvline(wavelength, color='blue', linestyle='--', alpha=0.7,lw=1)
                axt.text(wavelength, max(VB_Flux[int((wl_min-800)):int((wl_max-800))]) * 0.8, label, rotation=90, fontsize=8, color='blue', ha='right', va='bottom')

        st.pyplot(fig2)

if selected_scheme == 'I have the desired Wavelength window':

    with st.container():
        col1, col2 = st.columns([2, 2])
        with col1:
            wl_min = st.number_input(
                'Minimum Wavelength',
                min_value=800.00,
                max_value=10400.00,
                value=3600.00,
                step=0.01,  # Match the precision of the default value
                format="%.5f"  # Display five decimal places
            )
        with col2:
            wl_max = st.number_input(
                'Maximum Wavelength',
                min_value=800.00,
                max_value=10400.00,
                value=10400.00,
                step=0.01,  # Match the precision of the default value
                format="%.5f"  # Display five decimal places
            )

    z_max = (SDSS_MAX/wl_max)-1
    z_min = (SDSS_MIN/wl_min) - 1
    with st.container():
        fig1,axt = plt.subplots(figsize=(8,3))
        axt.plot(VB_Wavelength,VB_Flux,lw=1)
        axt.axvspan(wl_min,wl_max,color='red',alpha=0.2)
        for label, wavelength in lines.items():
            axt.axvline(wavelength, color='blue', linestyle='--', alpha=0.7,lw=1)
            axt.text(wavelength, max(VB_Flux) * 0.8, label, rotation=90, fontsize=8, color='blue', ha='right', va='bottom')

        st.pyplot(fig1)

        fig2,axt = plt.subplots(figsize=(8,3))
        axt.set_title(f'$\lambda_i$: {round(z_min,5)}\t\t\t\t\t\t$\lambda_f$: {round(z_max,5)}')
        axt.plot(VB_Wavelength[int((wl_min-800)):int((wl_max-800))],VB_Flux[int((wl_min-800)):int((wl_max-800))],lw=1)
        axt.axvspan(wl_min,wl_max,color='red',alpha=0.2)
        for label, wavelength in lines.items():
            if wavelength > wl_min and wavelength < wl_max:
                axt.axvline(wavelength, color='blue', linestyle='--', alpha=0.7,lw=1)
                axt.text(wavelength, max(VB_Flux[int((wl_min-800)):int((wl_max-800))]) * 0.8, label, rotation=90, fontsize=8, color='blue', ha='right', va='bottom')

        st.pyplot(fig2)
    

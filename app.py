# ==============================================================================
# LIBRERÍAS REQUERIDAS
# ==============================================================================
import streamlit as tf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración de la página de Streamlit (Layout Ancho para mejor visualización de tableros)
st.set_page_config(
    page_title="App Analizadora de Datasets",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

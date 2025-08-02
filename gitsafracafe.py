# Projeto API para previsão da safra do café

# Este projeto tem o objetivo de desensolver uma geotecnologia para previsão da produtividade do café com o uso de imagens do sensor MSI/Sentinel-2A e algoritmos de machine learning.
# Este trabalho É resultado de pesquisa acadêmica de Mestrado Profissional em Agricultura de Precisão do Colégio Politécnico da Universidade Federal de Santa Maria (UFSM).

# Imports essenciais
import os
import json
import time
import random
import string
import numpy as np
import pandas as pd
import zipfile

import geemap
import geocoder
import ee
import geopandas as gpd
import speech_recognition as sr

from shapely.geometry import Point, mapping, shape as shapely_shape, shape
from ipyfilechooser import FileChooser
from ipywidgets import (
    FileUpload, Button, Dropdown, Output, VBox, HBox, HTML, FloatText,
    ToggleButton, Layout, Checkbox
)
from IPython.display import display, clear_output
from geemap import geojson_to_ee

# Verificação da biblioteca Fiona
try:
    import fiona
    from fiona import drivers
    HAS_FIONA = True
except ImportError:
    HAS_FIONA = False
    print("Fiona não está instalada. Instale manualmente via requirements.txt.")

import json
import ee
import streamlit as st

SERVICE_ACCOUNT = "earth-engine-service@agriculturadeprecisao.iam.gserviceaccount.com"
CREDENTIALS = json.loads(st.secrets["GEE_CREDENTIALS"])

credentials = ee.ServiceAccountCredentials(
    SERVICE_ACCOUNT,
    key_data=json.dumps(CREDENTIALS)
)
ee.Initialize(credentials)




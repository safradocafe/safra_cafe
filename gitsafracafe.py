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

credentials_dict = dict(st.secrets["GEE_CREDENTIALS"])
credentials_json = json.dumps(credentials_dict)
credentials = ee.ServiceAccountCredentials(
    email=credentials_dict["client_email"],
    key_data=credentials_json
)
ee.Initialize(credentials)

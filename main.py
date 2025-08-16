from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/add-info')
def add_info():
    # Carrega bibliotecas de geoprocessamento
    import geemap
    import geopandas as gpd
    return render_template('add_info.html')

@app.route('/machine-learning')
def machine_learning():
    # Carrega bibliotecas de ML
    import pandas as pd
    import numpy as np
    import xgboost as xgb
    return render_template('ml.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))

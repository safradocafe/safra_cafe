import os
import glob
import numpy as np
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap
from shapely.geometry import Point
from shapely.ops import unary_union
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
from scipy.interpolate import griddata, Rbf
import warnings

st.set_page_config(layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 0rem !important; padding-bottom: 0.5rem; }
header, footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

def _find_latest_save_dir(base="/tmp/streamlit_dados"):
    if not os.path.isdir(base):
        return None
    candidates = [d for d in glob.glob(os.path.join(base, "salvamento-*")) if os.path.isdir(d)]
    if not candidates:
        return None
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]

def _load_area_and_points():   
    save_dir = st.session_state.get("tmp_save_dir")
    area_path = st.session_state.get("tmp_area_path")
    pontos_path = st.session_state.get("tmp_pontos_path")
    
    if not save_dir or not os.path.exists(save_dir):
        latest = _find_latest_save_dir()
        if latest:
            save_dir = latest
            area_path = os.path.join(save_dir, "area_amostral.gpkg")
            pontos_path = os.path.join(save_dir, "pontos_produtividade.gpkg")

    if not save_dir or not os.path.exists(save_dir):
        return None, None, None

    if not area_path or not os.path.exists(area_path):
        return None, None, save_dir

    gdf_area = gpd.read_file(area_path)
    if gdf_area.crs is None or gdf_area.crs.to_epsg() != 4326:
        gdf_area = gdf_area.set_crs(4326) if gdf_area.crs is None else gdf_area.to_crs(4326)

    gdf_pontos = None
    if pontos_path and os.path.exists(pontos_path):
        gdf_pontos = gpd.read_file(pontos_path)
        if gdf_pontos.crs is None or gdf_pontos.crs.to_epsg() != 4326:
            gdf_pontos = gdf_pontos.set_crs(4326) if gdf_pontos.crs is None else gdf_pontos.to_crs(4326)
        
        required = ["Code", "maduro_kg", "latitude", "longitude", "geometry"]
        faltando = [c for c in required if c not in gdf_pontos.columns]
        if faltando:
            st.error("Pontos encontrados, mas faltam colunas obrigatórias: " + ", ".join(faltando))
            gdf_pontos = None

    return gdf_area, gdf_pontos, save_dir

def get_local_utm_epsg(lon, lat):
    """Escolhe EPSG UTM local baseado no centroide (hemisfério sul/norte)."""
    zone = int((lon + 180) / 6) + 1
    return (32700 + zone) if lat < 0 else (32600 + zone)

def idw_interpolation(xyz, xi, yi, power=2):
    tree = cKDTree(xyz[:, :2])
    distances, idx = tree.query(np.c_[xi.flatten(), yi.flatten()], k=5)
    with np.errstate(divide="ignore"):
        weights = 1.0 / (distances ** power)
    weights = np.where(np.isinf(weights), 0, weights)
    z = np.sum(weights * xyz[idx, 2], axis=1) / np.sum(weights, axis=1)
    return z.reshape(xi.shape)

def spline_interpolation(xyz, xi, yi):
    rbf = Rbf(xyz[:, 0], xyz[:, 1], xyz[:, 2], function='thin_plate')
    return rbf(xi, yi)

def nearest_interpolation(xyz, xi, yi):
    return griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (xi, yi), method='nearest')

def linear_interpolation(xyz, xi, yi):
    return griddata((xyz[:, 0], xyz[:, 1]), xyz[:, 2], (xi, yi), method='linear')

METHODS = {
    'idw':    {'function': idw_interpolation,    'description': 'IDW'},
    'spline': {'function': spline_interpolation, 'description': 'Spline'},
    'nearest':{'function': nearest_interpolation,'description': 'Vizinho mais próximo'},
    'linear': {'function': linear_interpolation, 'description': 'Interpolação linear'},
}

st.markdown(
    "<h3 style='margin:0 0 .5rem 0; font-weight:700;'>🗺️ Mapa de variabilidade da produtividade</h3>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.subheader("Interpolação")
    method_key = st.selectbox(
        "Método",
        options=list(METHODS.keys()),
        format_func=lambda k: METHODS[k]['description'],
        index=1 if 'spline' in METHODS else 0
    )
    grid_res_m = st.slider("Resolução do grid (metros)", 2, 20, 5, 1)
    cmap_name = st.selectbox("Paleta de cores", ["YlGn", "viridis", "plasma", "magma"], index=0)
    show_points = st.checkbox("Mostrar pontos", value=True)
    show_area   = st.checkbox("Mostrar polígono da área", value=True)
    add_heat    = st.checkbox("Adicionar HeatMap (rápido)", value=False)
    heat_radius = st.slider("Raio HeatMap (px)", 8, 40, 18, 1, disabled=not add_heat)
    heat_blur   = st.slider("Desfoque HeatMap", 8, 35, 20, 1, disabled=not add_heat)
    heat_opacity= st.slider("Opacidade HeatMap", 0.2, 1.0, 0.6, 0.05, disabled=not add_heat)

gdf_area, gdf_pontos, base_dir = _load_area_and_points()

if gdf_area is None:
    st.warning("⚠️ Não encontrei a área amostral em /tmp. Volte para 'Adicionar informações' e salve os dados.")
    st.stop()

if gdf_pontos is None or gdf_pontos.empty:
    st.warning("⚠️ Pontos de produtividade não encontrados. Sem dados para interpolar.")
    st.stop()

cent = gdf_area.geometry.unary_union.centroid
epsg_utm = get_local_utm_epsg(cent.x, cent.y)

area_utm = gdf_area.to_crs(epsg=epsg_utm)
pontos_utm = gdf_pontos.to_crs(epsg=epsg_utm)

xmin, ymin, xmax, ymax = area_utm.total_bounds
pad = grid_res_m * 2
xmin, ymin, xmax, ymax = xmin - pad, ymin - pad, xmax + pad, ymax + pad

xi = np.arange(xmin, xmax, grid_res_m)
yi = np.arange(ymin, ymax, grid_res_m)
xi_grid, yi_grid = np.meshgrid(xi, yi)

vals = pontos_utm["maduro_kg"].astype(float).values
xyz = np.c_[pontos_utm.geometry.x, pontos_utm.geometry.y, vals]

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    Z = METHODS[method_key]['function'](xyz, xi_grid, yi_grid)

poly = unary_union(area_utm.geometry)  
mask = np.zeros_like(Z, dtype=bool)
px = xi_grid + (grid_res_m / 2.0)
py = yi_grid + (grid_res_m / 2.0)
for i in range(py.shape[0]):
    pts = [Point(x, y) for x, y in zip(px[i, :], py[i, :])]
    mask[i, :] = np.array([poly.contains(pt) for pt in pts])

Z_masked = np.where(mask, Z, np.nan)

vmin = np.nanpercentile(Z_masked, 2)
vmax = np.nanpercentile(Z_masked, 98)
if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
    vmin = np.nanmin(Z_masked)
    vmax = np.nanmax(Z_masked)
norm = Normalize(vmin=vmin, vmax=vmax, clip=True)
cmap = cm.get_cmap(cmap_name)

rgba = cmap(norm(Z_masked))
rgba[..., 3] = np.where(np.isnan(Z_masked), 0.0, 0.75)  

tmp_png_path = "/tmp/interp_overlay.png"
plt.imsave(tmp_png_path, rgba, format="png", origin="lower")

bb_ll = gpd.GeoSeries([Point(xmin, ymin)], crs=f"EPSG:{epsg_utm}").to_crs(4326)[0]
bb_ur = gpd.GeoSeries([Point(xmax, ymax)], crs=f"EPSG:{epsg_utm}").to_crs(4326)[0]
bounds = [[bb_ll.y, bb_ll.x], [bb_ur.y, bb_ur.x]]

start_loc = [cent.y, cent.x]
m = folium.Map(location=start_loc, zoom_start=17, tiles="OpenStreetMap")
folium.TileLayer(
    tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    attr='Esri', name='Satélite', overlay=False, control=True
).add_to(m)

if show_area:
    folium.GeoJson(
        gdf_area[['geometry']].__geo_interface__,
        name="Área Amostral",
        style_function=lambda x: {"color": "blue", "fillColor": "blue", "fillOpacity": 0.1, "weight": 2}
    ).add_to(m)

overlay = folium.raster_layers.ImageOverlay(
    image=tmp_png_path,
    bounds=bounds,
    opacity=1.0,
    name=f"Interpolação ({METHODS[method_key]['description']})",
    interactive=False,
    cross_origin=False
)
overlay.add_to(m)

if add_heat:
    vals_h = gdf_pontos["maduro_kg"].astype(float).values
    mn, mx = np.nanmin(vals_h), np.nanmax(vals_h)
    weights = np.ones_like(vals_h) if mx == mn else (vals_h - mn) / (mx - mn)
    heat_data = [
        [row["latitude"], row["longitude"], float(w)]
        for (_, row), w in zip(gdf_pontos.iterrows(), weights)
        if np.isfinite(row["latitude"]) and np.isfinite(row["longitude"])
    ]
    if heat_data:
        HeatMap(
            heat_data, radius=heat_radius, blur=heat_blur,
            min_opacity=heat_opacity, max_zoom=19,
            name="HeatMap (peso = maduro_kg)"
        ).add_to(m)

if show_points:
    for _, row in gdf_pontos.iterrows():
        lat, lon = row["latitude"], row["longitude"]
        if not (np.isfinite(lat) and np.isfinite(lon)):
            continue
        popup_html = (
            f"<b>Code:</b> {row.get('Code','-')}"
            f"<br><b>Produtividade (kg):</b> {row.get('maduro_kg','-')}"
            f"<br><b>Lat/Lon:</b> {lat:.6f}, {lon:.6f}"
        )
        folium.CircleMarker(
            location=[lat, lon],
            radius=4,
            color="black",
            fill=True, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=300)
        ).add_to(m)

folium.LayerControl(collapsed=False).add_to(m)

with st.sidebar:
    st.subheader("Legenda")
    fig, ax = plt.subplots(figsize=(3.8, 0.35))
    fig.subplots_adjust(bottom=0.5)
    cb1 = plt.colorbar(
        cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=ax, orientation='horizontal'
    )
    cb1.set_label('Produtividade (kg)', fontsize=9)
    cb1.ax.tick_params(labelsize=8)
    st.pyplot(fig, use_container_width=True)

st_folium(m, width=1000, height=640, key="mapa_produtividade_interp")

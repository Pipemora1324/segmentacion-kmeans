"""
Segmentación de clientes con K-Means – Despliegue interactivo (Parte 7)

Autor:       Juan Felipe Mora Revelo
Asignatura:  Inteligencia de Negocios y Minería de Datos
Institución: Universidad Cooperativa de Colombia

Ejecutar con:  streamlit run app.py
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

# ──────────────────────────────────────────────────────────────────────────────
# Configuración general
# ──────────────────────────────────────────────────────────────────────────────
AUTOR = "Juan Felipe Mora Revelo"
MATERIA = "Inteligencia de Negocios y Minería de Datos"
UNIVERSIDAD = "Universidad Cooperativa de Colombia"

DATA_PATH = Path(__file__).parent / "Mall_Customers.csv"
NOTEBOOK_PATH = Path(__file__).parent / "Taller_KMeans_Segmentacion.ipynb"
FEATURES = ["Ingreso", "Gasto"]
RANDOM_STATE = 42
K_OPTIMO = 5

st.set_page_config(
    page_title="Segmentación de clientes | K-Means",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────────────
# Tema claro / oscuro
# ──────────────────────────────────────────────────────────────────────────────
TEMAS = {
    "light": dict(
        bg="#F6F7F9", surface="#FFFFFF", surface2="#F0F2F5", ink="#1F2933", muted="#5B6672",
        line="#D9DEE4", accent="#1F4E79", accent_ink="#FFFFFF", grid="rgba(31,41,51,0.08)",
        seg={
            "Clientes de alto valor": "#2F7D5B",
            "Clientes con potencial": "#B7822F",
            "Clientes promedio": "#4A6FA5",
            "Clientes impulsivos": "#7E5A93",
            "Clientes de bajo valor": "#A9524A",
        },
        extra=["#2F7D5B", "#B7822F", "#4A6FA5", "#7E5A93", "#A9524A", "#3E8A8F", "#6B7280", "#8A6D3B", "#55703A", "#9C4D6E"],
    ),
    "dark": dict(
        bg="#171B21", surface="#1F242C", surface2="#272D36", ink="#E4E8ED", muted="#9AA4AF",
        line="#343B45", accent="#7AA7D6", accent_ink="#171B21", grid="rgba(228,232,237,0.08)",
        seg={
            "Clientes de alto valor": "#5FAF8A",
            "Clientes con potencial": "#D6A55A",
            "Clientes promedio": "#7D9CCB",
            "Clientes impulsivos": "#A688B8",
            "Clientes de bajo valor": "#C97C73",
        },
        extra=["#5FAF8A", "#D6A55A", "#7D9CCB", "#A688B8", "#C97C73", "#6FB3B8", "#A3ABB5", "#C2A878", "#8FB070", "#C98AA6"],
    ),
}

PERFILES = {
    "Clientes de alto valor": dict(
        objetivo="Retener y fidelizar",
        descripcion="Ingreso alto y gasto alto. Son el núcleo de los ingresos del centro comercial.",
        estrategias=[
            "Programa VIP o membresía premium con beneficios exclusivos",
            "Acceso anticipado a lanzamientos y eventos privados",
            "Atención personalizada (personal shopper)",
            "Oferta de marcas premium y experiencias",
        ],
    ),
    "Clientes con potencial": dict(
        objetivo="Activar y convertir",
        descripcion="Ingreso alto pero gasto bajo. Tienen capacidad de compra que hoy se va a otro lado.",
        estrategias=[
            "Encuestas para entender por qué no compran aquí",
            "Campañas de marcas de lujo centradas en calidad, no en descuento",
            "Experiencias: gastronomía, eventos y servicios exclusivos",
            "Beneficio de primera compra en categorías premium",
        ],
    ),
    "Clientes promedio": dict(
        objetivo="Mantener y subir el ticket",
        descripcion="Ingreso medio y gasto medio. El grupo más grande, estable y predecible.",
        estrategias=[
            "Programa de puntos por acumulación de compras",
            "Cross-selling, combos y paquetes",
            "Promociones de temporada (regreso a clases, Navidad)",
            "Incentivos para subir de nivel en el programa de fidelización",
        ],
    ),
    "Clientes impulsivos": dict(
        objetivo="Capitalizar el entusiasmo",
        descripcion="Ingreso bajo pero gasto alto. Jóvenes atentos a tendencias y promociones.",
        estrategias=[
            "Marketing en redes sociales e influencers",
            "Moda, tecnología y lanzamientos de temporada",
            "Ventas flash y ediciones limitadas",
            "Pagos a plazos y app con recompensas",
        ],
    ),
    "Clientes de bajo valor": dict(
        objetivo="Eficiencia y bajo costo",
        descripcion="Ingreso bajo y gasto bajo. Compran lo necesario y son muy sensibles al precio.",
        estrategias=[
            "Cupones y descuentos en productos básicos",
            "Tiendas outlet y de precio bajo",
            "Días de promoción específicos",
            "Canales de bajo costo (SMS, redes)",
        ],
    ),
}

if "modo" not in st.session_state:
    try:
        st.session_state.modo = "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        st.session_state.modo = "light"


def css(t: dict) -> str:
    return f"""
<style>
:root {{
  --bg:{t['bg']}; --surface:{t['surface']}; --surface2:{t['surface2']}; --ink:{t['ink']};
  --muted:{t['muted']}; --line:{t['line']}; --accent:{t['accent']}; --accent-ink:{t['accent_ink']};
  --font: "Segoe UI", Roboto, Arial, sans-serif;
}}

html, body, .stApp, [data-testid="stAppViewContainer"] {{ background: var(--bg) !important; color: var(--ink); }}
.stApp {{ font-family: var(--font); }}
.stApp p, .stApp li, .stApp label, .stApp input, .stApp textarea, .stApp button, [data-testid="stMarkdownContainer"] {{ font-family: var(--font); }}
[data-testid="stIconMaterial"] {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp h1, .stApp h2, .stApp h3, .stApp h4 {{ font-family: var(--font) !important; color: var(--ink) !important; letter-spacing: 0; }}
.stApp p, .stApp li, [data-testid="stMarkdownContainer"] {{ color: var(--ink); }}
[data-testid="stHeader"] {{ background: transparent !important; }}
[data-testid="stSidebar"] > div {{ background: var(--surface) !important; border-right: 1px solid var(--line); }}
.block-container {{ padding-top: 3.6rem; padding-bottom: 3rem; max-width: 1200px; }}
[data-testid="stCaptionContainer"], .stApp small {{ color: var(--muted) !important; }}

/* Widgets */
[data-testid="stWidgetLabel"] p {{ color: var(--ink) !important; font-weight: 600; }}
[data-baseweb="input"], [data-baseweb="base-input"], [data-baseweb="select"] > div,
[data-testid="stNumberInputContainer"] {{ background: var(--surface2) !important; border-color: var(--line) !important; }}
[data-baseweb="input"] input, [data-baseweb="select"] div {{ color: var(--ink) !important; -webkit-text-fill-color: var(--ink); }}
.stApp input, .stApp textarea, [data-baseweb="input"] > div, [data-baseweb="base-input"] > *,
[data-baseweb="select"] > div, [data-baseweb="select"] > div > div {{ background-color: var(--surface2) !important; color: var(--ink) !important; -webkit-text-fill-color: var(--ink) !important; }}
[data-baseweb="select"] > div div {{ background-color: transparent !important; }}
[data-baseweb="input"], [data-baseweb="select"] > div, [data-testid="stNumberInputContainer"] {{ border: 1px solid var(--line) !important; border-radius: 6px !important; }}
.stApp input::placeholder {{ color: var(--muted) !important; -webkit-text-fill-color: var(--muted) !important; opacity: .8; }}
[data-baseweb="select"] svg, [data-testid="stNumberInputContainer"] svg {{ fill: var(--ink) !important; color: var(--ink) !important; }}
[data-testid="stTooltipIcon"] svg, [data-testid="stTooltipIcon"] [data-testid="stIconMaterial"] {{ color: var(--muted) !important; }}
[data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] {{ background: var(--surface2) !important; color: var(--ink) !important; }}
[data-baseweb="popover"] ul, [data-baseweb="menu"] {{ background: var(--surface) !important; }}
[data-baseweb="popover"] li {{ color: var(--ink) !important; }}
[data-testid="stSliderThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {{ color: var(--muted) !important; }}
[data-testid="stForm"] {{ background: var(--surface); border: 1px solid var(--line); border-radius: 8px; padding: 1.2rem 1.2rem .5rem; }}
[data-testid="stBaseButton-primaryFormSubmit"], [data-testid="stBaseButton-primary"] {{
  background: var(--accent) !important; color: var(--accent-ink) !important; border: 1px solid var(--accent) !important;
  font-weight: 600; border-radius: 6px; padding: .6rem 1.1rem; }}
[data-testid="stBaseButton-primaryFormSubmit"] p, [data-testid="stBaseButton-primary"] p {{ color: var(--accent-ink) !important; }}
[data-testid="stBaseButton-secondary"] {{ background: var(--surface) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; border-radius: 6px; }}
[data-testid="stBaseButton-secondary"] p {{ color: var(--ink) !important; }}
button:focus-visible, [role="tab"]:focus-visible, input:focus-visible, summary:focus-visible, [role="radio"]:focus-visible,
[role="slider"]:focus-visible {{ outline: 2px solid var(--accent) !important; outline-offset: 2px; }}
[data-testid="stExpander"] details {{ background: var(--surface); border: 1px solid var(--line) !important; border-radius: 8px; }}
[data-testid="stExpander"] summary {{ color: var(--ink) !important; }}
[data-testid="stExpander"] summary:hover {{ background: var(--surface2); }}
[data-testid="stExpander"] summary p {{ color: var(--ink) !important; font-weight: 600; }}
[data-testid="stExpander"] summary svg {{ color: var(--muted) !important; fill: var(--muted) !important; }}
[data-testid="stRadio"] label p, [data-testid="stCheckbox"] label p {{ color: var(--ink) !important; }}
[data-testid="stAlert"] {{ border-radius: 8px; }}

/* Selector de tema */
[data-testid="stButtonGroup"] button {{ background: var(--surface) !important; border: 1px solid var(--line) !important; border-radius: 6px; }}
[data-testid="stButtonGroup"] button p {{ color: var(--muted) !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"], [data-testid="stButtonGroup"] button[kind$="Active"] {{ background: var(--surface2) !important; border-color: var(--accent) !important; }}
[data-testid="stButtonGroup"] button[aria-checked="true"] p, [data-testid="stButtonGroup"] button[kind$="Active"] p {{ color: var(--ink) !important; font-weight: 600; }}

/* Markdown dentro de la app (tablas, código y fórmulas de las respuestas) */
[data-testid="stMarkdownContainer"] strong {{ color: var(--ink); }}
[data-testid="stMarkdownContainer"] code {{ background: var(--surface2); color: var(--ink); border-radius: 4px; }}
[data-testid="stMarkdownContainer"] table {{ display: block; max-width: 100%; overflow-x: auto; border-collapse: collapse; font-size: .92rem; }}
[data-testid="stMarkdownContainer"] th {{ background: var(--surface2) !important; color: var(--ink) !important; border: 1px solid var(--line) !important; font-weight: 600; text-align: left; }}
[data-testid="stMarkdownContainer"] td {{ background: transparent !important; color: var(--ink) !important; border: 1px solid var(--line) !important; vertical-align: top; }}
.katex {{ color: var(--ink); }}

/* Pestañas: subrayado simple */
.stTabs [role="tablist"] {{ gap: 1.4rem; background: transparent; padding: 0; border-bottom: 1px solid var(--line) !important; overflow-x: auto; scrollbar-width: none; }}
.stTabs [role="tablist"]::-webkit-scrollbar {{ display: none; }}
.stTabs [role="tablist"]::after, .stTabs [role="tablist"]::before {{ display: none !important; }}
.stTabs [role="tab"] {{ height: auto !important; padding: .55rem .1rem !important; margin-bottom: -1px; border-radius: 0; background: transparent !important;
  white-space: nowrap; border: none !important; border-bottom: 2px solid transparent !important; cursor: pointer; }}
.stTabs [role="tab"] p {{ color: var(--muted) !important; font-weight: 500; font-size: .95rem; }}
.stTabs [role="tab"]:hover p {{ color: var(--ink) !important; }}
.stTabs [role="tab"][aria-selected="true"] {{ border-bottom-color: var(--accent) !important; }}
.stTabs [role="tab"][aria-selected="true"] p {{ color: var(--ink) !important; font-weight: 600; }}
.stTabs .react-aria-SelectionIndicator, .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display: none !important; }}
.stTabs [role="tabpanel"] {{ padding-top: 1.4rem; }}

/* Encabezado */
.page-head {{ margin-bottom: 1.4rem; padding-bottom: 1.4rem; border-bottom: 1px solid var(--line); }}
.page-meta {{ color: var(--muted); font-size: .9rem; margin-bottom: .6rem; }}
.page-head h1 {{ font-size: 2.2rem; line-height: 1.15; margin: 0 0 .6rem; padding: 0; font-weight: 700; }}
.page-head p.lead {{ font-size: 1.02rem; line-height: 1.6; color: var(--muted); max-width: 46rem; margin: 0 0 1.2rem; }}
.metrics {{ display: grid; grid-template-columns: repeat(4, 1fr); border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
.metric {{ padding: .85rem 1.1rem; border-left: 1px solid var(--line); }}
.metric:first-child {{ border-left: none; }}
.metric b {{ display: block; font-size: 1.45rem; font-weight: 600; color: var(--ink); line-height: 1.2; font-variant-numeric: tabular-nums; }}
.metric span {{ font-size: .84rem; color: var(--muted); }}

/* Bloques de contenido */
.section-title {{ font-size: 1.4rem; font-weight: 600; color: var(--ink); margin: .2rem 0 .3rem; }}
.section-sub {{ color: var(--muted); margin-bottom: 1.1rem; max-width: 46rem; line-height: 1.55; }}
.part-title {{ font-size: 1.12rem; font-weight: 600; color: var(--ink); margin: 1.4rem 0 .6rem; padding-bottom: .35rem; border-bottom: 1px solid var(--line); }}
.callout {{ border: 1px solid var(--line); border-left: 4px solid var(--accent); background: var(--surface); border-radius: 6px; padding: .9rem 1.1rem; color: var(--ink); margin: .6rem 0 1rem; line-height: 1.55; }}

.tbl-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--surface); }}
table.tbl {{ width: 100%; border-collapse: collapse; font-size: .93rem; font-variant-numeric: tabular-nums; }}
table.tbl th {{ text-align: left; padding: .65rem .9rem; color: var(--muted); font-weight: 600; background: var(--surface2); white-space: nowrap; }}
table.tbl td {{ padding: .6rem .9rem; border-top: 1px solid var(--line); color: var(--ink); white-space: nowrap; }}
table.tbl tr.hl td {{ background: color-mix(in srgb, var(--accent) 12%, transparent); font-weight: 600; }}
.dot {{ display: inline-block; width: .7rem; height: .7rem; border-radius: 50%; margin-right: .45rem; vertical-align: middle; }}

/* Tarjetas de segmento */
.seg-card {{ background: var(--surface); border: 1px solid var(--line); border-left: 4px solid var(--c); border-radius: 8px; padding: 1.1rem 1.2rem; height: 100%; }}
.seg-head {{ margin-bottom: .5rem; }}
.seg-head .nm {{ font-weight: 600; font-size: 1.08rem; color: var(--ink); line-height: 1.25; }}
.seg-num {{ color: var(--muted); font-size: .84rem; }}
.seg-stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: .4rem; margin: .8rem 0; padding: .7rem 0; border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }}
.seg-stats b {{ display: block; font-size: 1.1rem; font-weight: 600; color: var(--ink); font-variant-numeric: tabular-nums; }}
.seg-stats span {{ font-size: .76rem; color: var(--muted); }}
.seg-card p {{ font-size: .92rem; color: var(--muted) !important; line-height: 1.5; margin: 0 0 .5rem; }}
.seg-card p.seg-goal {{ color: var(--ink) !important; font-weight: 700; margin: 0 0 .3rem; }}
.seg-card ul {{ margin: .3rem 0 0; padding-left: 1.1rem; }}
.seg-card li {{ font-size: .88rem; color: var(--ink) !important; margin-bottom: .2rem; }}

/* Resultado de clasificación */
.result {{ background: var(--surface); border: 1px solid var(--line); border-left: 5px solid var(--c); border-radius: 8px; padding: 1.1rem 1.3rem 1.2rem; }}
.result .lbl {{ font-size: .86rem; color: var(--muted); }}
.result .big {{ display: flex; align-items: center; gap: .55rem; font-size: 1.6rem; font-weight: 600; color: var(--ink); line-height: 1.2; margin: .25rem 0 .3rem; }}
.result .big .dot {{ width: .9rem; height: .9rem; margin: 0; flex: none; }}
.result .who {{ font-size: .93rem; color: var(--muted); margin-bottom: .8rem; padding-bottom: .8rem; border-bottom: 1px solid var(--line); }}
.result p {{ color: var(--muted) !important; margin: 0 0 .6rem; line-height: 1.55; }}
.result p b {{ color: var(--ink); }}
.result li {{ color: var(--ink) !important; margin-bottom: .25rem; }}
.conf {{ display: flex; flex-wrap: wrap; gap: .3rem 1rem; justify-content: space-between; font-size: .85rem; color: var(--muted); margin-top: .7rem; }}
.empty {{ border: 1px dashed var(--line); border-radius: 8px; padding: 2rem 1.5rem; text-align: center; color: var(--muted); background: var(--surface); }}
.empty b {{ color: var(--ink); display: block; font-size: 1.05rem; margin-bottom: .4rem; }}

.footer {{ margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--line); color: var(--muted); font-size: .88rem; display: flex; flex-wrap: wrap; gap: .4rem 1.4rem; justify-content: space-between; }}

/* Responsive */
@media (max-width: 900px) {{
  .block-container {{ padding-left: 1rem; padding-right: 1rem; padding-top: 3.4rem; }}
}}
@media (max-width: 640px) {{
  .page-head h1 {{ font-size: 1.75rem; }}
  .metrics {{ grid-template-columns: repeat(2, 1fr); }}
  .metric:nth-child(3) {{ border-left: none; }}
  .metric:nth-child(n+3) {{ border-top: 1px solid var(--line); }}
  .metric b {{ font-size: 1.25rem; }}
  .section-title {{ font-size: 1.25rem; }}
  .stTabs [role="tablist"] {{ gap: 1rem; }}
}}
</style>
"""


# ──────────────────────────────────────────────────────────────────────────────
# Datos y modelos (con caché)
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def cargar_datos() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.rename(columns={
        "Genre": "Genero", "Age": "Edad",
        "Annual Income (k$)": "Ingreso", "Spending Score (1-100)": "Gasto",
    })


@st.cache_resource
def escalador(_df: pd.DataFrame):
    sc = StandardScaler().fit(_df[FEATURES].values)
    return sc, sc.transform(_df[FEATURES].values)


@st.cache_data
def curvas(_X: np.ndarray):
    inercia = {k: KMeans(k, n_init=10, random_state=RANDOM_STATE).fit(_X).inertia_ for k in range(1, 11)}
    silueta = {k: silhouette_score(_X, KMeans(k, n_init=10, random_state=RANDOM_STATE).fit_predict(_X)) for k in range(2, 11)}
    return inercia, silueta


@st.cache_resource
def modelo(k: int, _X: np.ndarray):
    return KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=RANDOM_STATE).fit(_X)


def asignar_perfil(ingreso: float, gasto: float, ing_med: float, gas_med: float, margen: float = 15) -> str:
    alto_i, bajo_i = ingreso > ing_med + margen, ingreso < ing_med - margen
    alto_g, bajo_g = gasto > gas_med + margen, gasto < gas_med - margen
    if alto_i and alto_g:
        return "Clientes de alto valor"
    if alto_i and bajo_g:
        return "Clientes con potencial"
    if bajo_i and bajo_g:
        return "Clientes de bajo valor"
    if bajo_i and alto_g:
        return "Clientes impulsivos"
    return "Clientes promedio"


@st.cache_data
def preguntas_taller(ruta: str, mtime: float) -> dict:
    """Lee del cuaderno las celdas markdown que empiezan con '**❓' y las agrupa por parte."""
    nb = json.loads(Path(ruta).read_text(encoding="utf-8"))
    partes, parte = {}, None
    for celda in nb["cells"]:
        if celda["cell_type"] != "markdown":
            continue
        texto = "".join(celda["source"]).strip()
        m = re.search(r"^# Parte (\d+)", texto, flags=re.M)
        if m:
            parte = int(m.group(1))
        if texto.startswith("**❓") and parte is not None:
            titulo, _, respuesta = texto.partition("\n")
            titulo = titulo.strip().removeprefix("**❓").removesuffix("**").strip()
            partes.setdefault(parte, []).append((titulo, respuesta.strip()))
    return partes


# ──────────────────────────────────────────────────────────────────────────────
# Utilidades visuales
# ──────────────────────────────────────────────────────────────────────────────
PLOT_FONT = '"Segoe UI", Roboto, Arial, sans-serif'


def estilo_fig(fig: go.Figure, t: dict, height: int = 440, legend: bool = True) -> go.Figure:
    fig.update_layout(
        height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=PLOT_FONT, color=t["ink"], size=13),
        margin=dict(l=10, r=10, t=30, b=10), showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, bgcolor="rgba(0,0,0,0)", font=dict(color=t["muted"])),
        hoverlabel=dict(bgcolor=t["surface"], bordercolor=t["line"], font=dict(color=t["ink"], family=PLOT_FONT)),
    )
    ax = dict(gridcolor=t["grid"], zerolinecolor=t["grid"], linecolor=t["line"], tickfont=dict(color=t["muted"]),
              title_font=dict(color=t["muted"]))
    fig.update_xaxes(**ax)
    fig.update_yaxes(**ax)
    return fig


def tabla_html(df: pd.DataFrame, resaltar=None, colores=None) -> str:
    cab = "".join(f"<th>{c}</th>" for c in df.columns)
    filas = []
    for i, row in df.iterrows():
        celdas = []
        for j, v in enumerate(row):
            txt = f"{v:,.2f}" if isinstance(v, (float, np.floating)) else f"{v}"
            if j == 0 and colores is not None:
                txt = f'<span class="dot" style="background:{colores[i]}"></span>{txt}'
            celdas.append(f"<td>{txt}</td>")
        cls = ' class="hl"' if resaltar is not None and i == resaltar else ""
        filas.append(f"<tr{cls}>{''.join(celdas)}</tr>")
    return f'<div class="tbl-wrap"><table class="tbl"><thead><tr>{cab}</tr></thead><tbody>{"".join(filas)}</tbody></table></div>'


CFG = {"displaylogo": False, "responsive": True, "modeBarButtonsToRemove": ["lasso2d", "select2d"]}

# ──────────────────────────────────────────────────────────────────────────────
# Preparación
# ──────────────────────────────────────────────────────────────────────────────
df = cargar_datos()
scaler, X = escalador(df)
inercias, siluetas = curvas(X)
ING_MED, GAS_MED = df["Ingreso"].mean(), df["Gasto"].mean()

# Barra lateral: ajustes
with st.sidebar:
    st.markdown("### Ajustes")
    k_sel = st.slider("Número de clusters (K)", 2, 10, K_OPTIMO,
                      help="K = 5 es el valor óptimo según el método del codo y el coeficiente de silueta.")
    if k_sel != K_OPTIMO:
        st.caption(f"Estás explorando K = {k_sel}. El valor recomendado es K = {K_OPTIMO}.")
    st.markdown("---")
    st.markdown(f"**{AUTOR}**  \n{MATERIA}  \n{UNIVERSIDAD}")

# Selector de tema (arriba a la derecha)
_, col_tema = st.columns([5, 1.25])
with col_tema:
    elegido = st.segmented_control(
        "Tema", ["Claro", "Oscuro"],
        default="Oscuro" if st.session_state.modo == "dark" else "Claro",
        label_visibility="collapsed", key="tema_ctrl",
    )
    if elegido:
        st.session_state.modo = "dark" if elegido == "Oscuro" else "light"

T = TEMAS[st.session_state.modo]
st.markdown(css(T), unsafe_allow_html=True)

km = modelo(k_sel, X)
df["Cluster"] = km.labels_
centros = pd.DataFrame(scaler.inverse_transform(km.cluster_centers_), columns=FEATURES)
centros["Perfil"] = [asignar_perfil(r.Ingreso, r.Gasto, ING_MED, GAS_MED) for r in centros.itertuples()]
centros["Clientes"] = df["Cluster"].value_counts().sort_index().values
centros["Edad"] = df.groupby("Cluster")["Edad"].mean().values
perfiles_unicos = centros["Perfil"].nunique() == len(centros)
COLOR = {c: (T["seg"][centros.loc[c, "Perfil"]] if perfiles_unicos else T["extra"][c % 10]) for c in centros.index}
NOMBRE = {c: centros.loc[c, "Perfil"] if perfiles_unicos else f"{centros.loc[c, 'Perfil']} ({c})" for c in centros.index}
df["Perfil"] = df["Cluster"].map(NOMBRE)
sil_actual = silhouette_score(X, km.labels_)

# ──────────────────────────────────────────────────────────────────────────────
# Encabezado
# ──────────────────────────────────────────────────────────────────────────────
orden = centros.sort_values(["Ingreso", "Gasto"], ascending=False).index

st.markdown(f"""
<header class="page-head">
  <div class="page-meta">{AUTOR} &nbsp;|&nbsp; {MATERIA} &nbsp;|&nbsp; {UNIVERSIDAD}</div>
  <h1>Segmentación de clientes con K&#8209;Means</h1>
  <p class="lead">Agrupamos a los 200 clientes de un centro comercial según su ingreso anual y su puntaje de gasto,
  para decidir a quién retener, a quién activar y dónde está la mayor oportunidad de crecimiento.</p>
  <div class="metrics">
    <div class="metric"><b>{len(df)}</b><span>clientes analizados</span></div>
    <div class="metric"><b>K = {k_sel}</b><span>segmentos</span></div>
    <div class="metric"><b>{sil_actual:.3f}</b><span>coeficiente de silueta</span></div>
    <div class="metric"><b>{km.inertia_:.1f}</b><span>inercia (WCSS)</span></div>
  </div>
</header>
""", unsafe_allow_html=True)

tab_datos, tab_codo, tab_sil, tab_seg, tab_cli, tab_preg = st.tabs(
    ["Datos", "Método del codo", "Silueta", "Segmentos", "Clasificar cliente", "Preguntas del taller"]
)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 1: Datos
# ──────────────────────────────────────────────────────────────────────────────
with tab_datos:
    st.markdown('<div class="section-title">Exploración de los datos</div>'
                '<div class="section-sub">200 registros sin valores nulos ni duplicados. Para segmentar se usan el ingreso anual '
                'y el puntaje de gasto; la edad y el género sirven para describir los grupos.</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([1.35, 1], gap="large")
    with c1:
        fig = go.Figure(go.Scatter(
            x=df["Ingreso"], y=df["Gasto"], mode="markers",
            marker=dict(size=10, color=T["accent"], opacity=.75, line=dict(width=1, color=T["surface"])),
            customdata=df[["CustomerID", "Edad", "Genero"]],
            hovertemplate="Cliente %{customdata[0]}<br>Edad %{customdata[1]} · %{customdata[2]}<br>"
                          "Ingreso %{x} k$<br>Gasto %{y}<extra></extra>"))
        fig.update_xaxes(title="Ingreso anual (k$)")
        fig.update_yaxes(title="Puntaje de gasto (1-100)")
        st.plotly_chart(estilo_fig(fig, T, 430, legend=False), width="stretch", config=CFG)
        st.caption("Antes de agrupar ya se distinguen cinco nubes de puntos: un núcleo central y cuatro esquinas.")
    with c2:
        var = st.radio("Distribución de", ["Edad", "Ingreso", "Gasto"], horizontal=True)
        fig = go.Figure()
        for g, col in [("Female", T["seg"]["Clientes impulsivos"]), ("Male", T["seg"]["Clientes promedio"])]:
            fig.add_trace(go.Histogram(x=df.loc[df["Genero"] == g, var], name="Mujeres" if g == "Female" else "Hombres",
                                       marker_color=col, opacity=.8, nbinsx=15))
        fig.update_layout(barmode="overlay")
        st.plotly_chart(estilo_fig(fig, T, 330), width="stretch", config=CFG)
        corr = df[["Edad", "Ingreso", "Gasto"]].corr()
        st.markdown(f"""<div class="callout"><b>Relaciones.</b> Ingreso y gasto no están correlacionados
        (r = {corr.loc['Ingreso', 'Gasto']:.2f}), pero forman grupos. La edad tiene una relación negativa moderada
        con el gasto (r = {corr.loc['Edad', 'Gasto']:.2f}): los más jóvenes gastan más.</div>""", unsafe_allow_html=True)

    desc = df[["Edad", "Ingreso", "Gasto"]].describe().T[["mean", "std", "min", "50%", "max"]].round(2).reset_index()
    desc.columns = ["Variable", "Media", "Desv. estándar", "Mínimo", "Mediana", "Máximo"]
    st.markdown(tabla_html(desc), unsafe_allow_html=True)

    with st.expander("Ver el dataset completo"):
        st.dataframe(df[["CustomerID", "Genero", "Edad", "Ingreso", "Gasto"]], width="stretch", hide_index=True, height=320)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 2: Método del codo
# ──────────────────────────────────────────────────────────────────────────────
with tab_codo:
    st.markdown('<div class="section-title">Método del codo</div>'
                '<div class="section-sub">La inercia (WCSS) suma las distancias al cuadrado de cada cliente a su centroide. '
                'Siempre baja al agregar clusters; buscamos el punto donde deja de bajar con fuerza.</div>', unsafe_allow_html=True)
    ks, vals = list(inercias.keys()), list(inercias.values())
    c1, c2 = st.columns([1.7, 1], gap="large")
    with c1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=ks, y=vals, mode="lines+markers", name="Inercia",
                                 line=dict(color=T["accent"], width=2.5),
                                 marker=dict(size=9, color=T["accent"], line=dict(width=2, color=T["surface"])),
                                 hovertemplate="K = %{x}<br>Inercia = %{y:.2f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=[K_OPTIMO], y=[inercias[K_OPTIMO]], mode="markers", name=f"Codo (K = {K_OPTIMO})",
                                 marker=dict(size=26, color="rgba(0,0,0,0)", line=dict(width=3, color=T["seg"]["Clientes con potencial"])),
                                 hoverinfo="skip"))
        if k_sel != K_OPTIMO:
            fig.add_trace(go.Scatter(x=[k_sel], y=[inercias[k_sel]], mode="markers", name=f"K elegido ({k_sel})",
                                     marker=dict(size=16, symbol="diamond", color=T["seg"]["Clientes impulsivos"]), hoverinfo="skip"))
        fig.add_annotation(x=K_OPTIMO, y=inercias[K_OPTIMO], text=f"<b>Codo en K = {K_OPTIMO}</b>", showarrow=True,
                           arrowhead=2, ax=70, ay=-60, font=dict(color=T["ink"], size=14), arrowcolor=T["muted"])
        fig.update_xaxes(title="Número de clusters (K)", dtick=1)
        fig.update_yaxes(title="Inercia (WCSS)")
        st.plotly_chart(estilo_fig(fig, T, 450), width="stretch", config=CFG)
    with c2:
        t_codo = pd.DataFrame({"K": ks, "Inercia": vals})
        t_codo["Reducción"] = (-t_codo["Inercia"].pct_change() * 100).map(lambda v: "" if pd.isna(v) else f"{v:.1f} %")
        st.markdown(tabla_html(t_codo, resaltar=K_OPTIMO - 1), unsafe_allow_html=True)
    st.markdown(f"""<div class="callout">Hasta K = {K_OPTIMO} cada cluster nuevo reduce la inercia entre 31 % y 42 %.
    Desde K = 6 la reducción cae al 16 % o menos: agregar más grupos ya no compensa. Por eso el codo está en <b>K = {K_OPTIMO}</b>.</div>""",
                unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 3: Silueta
# ──────────────────────────────────────────────────────────────────────────────
with tab_sil:
    st.markdown('<div class="section-title">Coeficiente de silueta</div>'
                '<div class="section-sub">Compara qué tan cerca está cada cliente de su propio grupo frente al grupo vecino. '
                'Va de −1 a 1: cerca de 1, grupos bien separados; cerca de 0, fronteras difusas; negativo, cliente mal asignado.</div>',
                unsafe_allow_html=True)
    k_best = max(siluetas, key=siluetas.get)
    c1, c2 = st.columns(2, gap="large")
    with c1:
        cols = [T["seg"]["Clientes de alto valor"] if k == k_best else (T["seg"]["Clientes impulsivos"] if k == k_sel else T["line"])
                for k in siluetas]
        fig = go.Figure(go.Bar(x=list(siluetas.keys()), y=list(siluetas.values()), marker_color=cols,
                               text=[f"{v:.3f}" for v in siluetas.values()], textposition="outside",
                               textfont=dict(color=T["ink"]), hovertemplate="K = %{x}<br>Silueta = %{y:.4f}<extra></extra>"))
        fig.update_xaxes(title="Número de clusters (K)", dtick=1)
        fig.update_yaxes(title="Silueta promedio", range=[0, max(siluetas.values()) * 1.18])
        st.plotly_chart(estilo_fig(fig, T, 420, legend=False), width="stretch", config=CFG)
        st.caption(f"Máximo en K = {k_best} (s = {siluetas[k_best]:.3f}). Coincide con el método del codo.")
    with c2:
        sv = silhouette_samples(X, km.labels_)
        fig = go.Figure()
        y0 = 0
        for c in sorted(centros.index):
            v = np.sort(sv[km.labels_ == c])
            fig.add_trace(go.Bar(x=v, y=np.arange(y0, y0 + len(v)), orientation="h", marker_color=COLOR[c],
                                 marker_line_width=0, name=NOMBRE[c], width=1,
                                 hovertemplate=f"{NOMBRE[c]}<br>silueta = %{{x:.3f}}<extra></extra>"))
            y0 += len(v) + 6
        fig.add_vline(x=sv.mean(), line_dash="dash", line_color=T["ink"],
                      annotation_text=f"promedio {sv.mean():.3f}", annotation_font_color=T["ink"])
        fig.update_layout(bargap=0)
        fig.update_xaxes(title=f"Silueta por cliente (K = {k_sel})")
        fig.update_yaxes(showticklabels=False, showgrid=False)
        st.plotly_chart(estilo_fig(fig, T, 420, legend=False), width="stretch", config=CFG)
        st.caption(f"Clientes con silueta negativa: {(sv < 0).sum()} de {len(sv)}.")
    st.markdown(f"""<div class="callout"><b>¿Coinciden los dos métodos?</b> Sí. El codo y la silueta señalan K = {K_OPTIMO}.
    Uno mide solo la compactación de los grupos y el otro además su separación; que ambos lleguen al mismo número
    da confianza en la elección.</div>""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 4: Segmentos
# ──────────────────────────────────────────────────────────────────────────────
with tab_seg:
    st.markdown('<div class="section-title">Los segmentos</div>'
                '<div class="section-sub">Cada color es un grupo de clientes con ingreso y gasto similares. '
                'La X marca el centroide: el cliente típico del grupo.</div>', unsafe_allow_html=True)
    vista = st.radio("Vista", ["2D: ingreso y gasto", "3D: con edad"], horizontal=True, label_visibility="collapsed")
    if vista.startswith("2D"):
        fig = go.Figure()
        for c in sorted(centros.index):
            s = df[df["Cluster"] == c]
            fig.add_trace(go.Scatter(x=s["Ingreso"], y=s["Gasto"], mode="markers", name=NOMBRE[c],
                                     marker=dict(size=11, color=COLOR[c], opacity=.85, line=dict(width=1, color=T["surface"])),
                                     customdata=s[["CustomerID", "Edad", "Genero"]],
                                     hovertemplate="Cliente %{customdata[0]} · %{customdata[1]} años<br>"
                                                   "Ingreso %{x} k$ · gasto %{y}<extra>" + NOMBRE[c] + "</extra>"))
        fig.add_trace(go.Scatter(x=centros["Ingreso"], y=centros["Gasto"], mode="markers", name="Centroides",
                                 marker=dict(symbol="x", size=20, color=T["ink"], line=dict(width=2, color=T["surface"])),
                                 hovertemplate="Centroide<br>Ingreso %{x:.1f} k$ · gasto %{y:.1f}<extra></extra>"))
        fig.add_hline(y=GAS_MED, line_dash="dot", line_color=T["muted"], opacity=.5)
        fig.add_vline(x=ING_MED, line_dash="dot", line_color=T["muted"], opacity=.5)
        fig.update_xaxes(title="Ingreso anual (k$)")
        fig.update_yaxes(title="Puntaje de gasto (1-100)")
        st.plotly_chart(estilo_fig(fig, T, 540), width="stretch", config=CFG)
    else:
        fig = go.Figure()
        for c in sorted(centros.index):
            s = df[df["Cluster"] == c]
            fig.add_trace(go.Scatter3d(x=s["Ingreso"], y=s["Gasto"], z=s["Edad"], mode="markers", name=NOMBRE[c],
                                       marker=dict(size=5, color=COLOR[c], opacity=.9)))
        sc = dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=T["grid"], color=T["muted"], showbackground=False)
        fig.update_layout(scene=dict(xaxis=dict(title="Ingreso", **sc), yaxis=dict(title="Gasto", **sc), zaxis=dict(title="Edad", **sc)))
        st.plotly_chart(estilo_fig(fig, T, 580), width="stretch", config=CFG)
        st.caption("La edad no se usó para entrenar, pero se ve que los grupos de gasto alto son más jóvenes.")

    st.markdown('<div class="section-title" style="margin-top:1.2rem">Perfiles y estrategias de marketing</div>', unsafe_allow_html=True)
    cards = []
    for c in orden:
        p = centros.loc[c, "Perfil"]
        info = PERFILES[p]
        li = "".join(f"<li>{e}</li>" for e in info["estrategias"])
        cards.append(f"""
        <div class="seg-card" style="--c:{COLOR[c]}">
          <div class="seg-head">
            <div class="nm">{NOMBRE[c]}</div><div class="seg-num">Cluster {c} · {centros.loc[c, 'Clientes']} clientes</div></div>
          <p>{info['descripcion']}</p>
          <div class="seg-stats">
            <div><b>{centros.loc[c, 'Ingreso']:.0f} k$</b><span>ingreso</span></div>
            <div><b>{centros.loc[c, 'Gasto']:.0f}</b><span>gasto</span></div>
            <div><b>{centros.loc[c, 'Edad']:.0f}</b><span>edad media</span></div>
          </div>
          <p class="seg-goal">Objetivo: {info['objetivo'].lower()}</p>
          <ul>{li}</ul>
        </div>""")
    n_cols = 3
    for i in range(0, len(cards), n_cols):
        cols = st.columns(n_cols, gap="medium")
        for col, card in zip(cols, cards[i:i + n_cols]):
            col.markdown(card, unsafe_allow_html=True)
        st.write("")

    if k_sel == K_OPTIMO:
        st.markdown("""<div class="callout"><b>Mayor oportunidad de crecimiento: clientes con potencial.</b>
        Tienen el mismo ingreso que los de alto valor (≈88 k$), pero su gasto es de ≈17 frente a ≈82. No hay que crear
        demanda, solo captarla. La segunda oportunidad son los clientes promedio, que por ser el 40 % de la base
        multiplican cualquier mejora en el ticket.</div>""", unsafe_allow_html=True)

    t_c = centros[["Perfil", "Clientes", "Ingreso", "Gasto", "Edad"]].copy()
    t_c.insert(0, "Cluster", t_c.index)
    t_c.columns = ["Cluster", "Perfil", "Clientes", "Ingreso (k$)", "Gasto", "Edad media"]
    st.markdown(tabla_html(t_c, colores=COLOR), unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────────────────────
# Tab 5: Clasificar nuevo cliente
# ──────────────────────────────────────────────────────────────────────────────
with tab_cli:
    st.markdown('<div class="section-title">Clasificar un cliente nuevo</div>'
                '<div class="section-sub">Ingresa los datos del cliente. El modelo lo escala con el mismo StandardScaler '
                'y lo asigna al centroide más cercano.</div>', unsafe_allow_html=True)
    if "historial" not in st.session_state:
        st.session_state.historial = []

    c1, c2 = st.columns([1, 1.25], gap="large")
    with c1:
        with st.form("nuevo_cliente", border=False):
            nombre = st.text_input("Nombre del cliente", placeholder="Opcional, por ejemplo: Laura Gómez")
            a, b = st.columns(2)
            genero = a.selectbox("Género", ["Mujer", "Hombre"])
            edad = b.number_input("Edad", min_value=15, max_value=90, value=30, step=1)
            ingreso = st.slider("Ingreso anual (miles de USD)", 10, 150, 75)
            gasto = st.slider("Puntaje de gasto (1 a 100)", 1, 100, 80,
                              help="Puntaje que asigna el centro comercial según el comportamiento de compra del cliente.")
            enviar = st.form_submit_button("Clasificar cliente", type="primary", width="stretch")

    if enviar:
        z = scaler.transform([[ingreso, gasto]])
        cl = int(km.predict(z)[0])
        dist = np.linalg.norm(km.cluster_centers_ - z, axis=1)
        st.session_state.ultimo = dict(nombre=nombre.strip() or "Cliente nuevo", genero=genero, edad=edad,
                                       ingreso=ingreso, gasto=gasto, cluster=cl, dist=dist, k=k_sel)
        st.session_state.historial.insert(0, {"Cliente": nombre.strip() or "Cliente nuevo", "Género": genero, "Edad": edad,
                                              "Ingreso (k$)": ingreso, "Gasto": gasto, "Cluster": cl, "Perfil": NOMBRE[cl]})

    u = st.session_state.get("ultimo")
    with c2:
        if u and u["k"] == k_sel:
            cl = u["cluster"]
            perfil = centros.loc[cl, "Perfil"]
            info = PERFILES[perfil]
            d = u["dist"]
            orden_d = np.sort(d)
            margen = (orden_d[1] - orden_d[0]) / orden_d[1] * 100 if len(d) > 1 else 100
            li = "".join(f"<li>{e}</li>" for e in info["estrategias"])
            st.markdown(f"""
            <div class="result" style="--c:{COLOR[cl]}">
              <div class="lbl">Resultado de la clasificación</div>
              <div class="big"><span class="dot" style="background:{COLOR[cl]}"></span>{NOMBRE[cl]}</div>
              <div class="who">{u['nombre']} · {u['genero']}, {u['edad']} años · ingreso {u['ingreso']} k$ · gasto {u['gasto']}</div>
              <p><b>Cluster {cl}.</b> {info['descripcion']} Objetivo: <b>{info['objetivo'].lower()}</b>.</p>
              <ul>{li}</ul>
              <div class="conf"><span>Distancia al centroide: {orden_d[0]:.2f}</span>
              <span>{margen:.0f} % más cerca que del segundo grupo</span></div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div class="empty"><b>Aquí aparecerá el resultado</b>
            Completa el formulario y pulsa «Clasificar cliente» para ver su segmento y la estrategia recomendada.</div>""",
                        unsafe_allow_html=True)

    if u and u["k"] == k_sel:
        cl = u["cluster"]
        g1, g2 = st.columns([1.6, 1], gap="large")
        with g1:
            fig = go.Figure()
            for c in sorted(centros.index):
                s = df[df["Cluster"] == c]
                fig.add_trace(go.Scatter(x=s["Ingreso"], y=s["Gasto"], mode="markers", name=NOMBRE[c],
                                         marker=dict(size=9, color=COLOR[c], opacity=.9 if c == cl else .22),
                                         hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[centros.loc[cl, "Ingreso"], u["ingreso"]], y=[centros.loc[cl, "Gasto"], u["gasto"]],
                                     mode="lines", line=dict(color=T["ink"], dash="dot", width=2), showlegend=False, hoverinfo="skip"))
            fig.add_trace(go.Scatter(x=[u["ingreso"]], y=[u["gasto"]], mode="markers+text", name=u["nombre"],
                                     marker=dict(symbol="star", size=30, color=COLOR[cl], line=dict(width=2.5, color=T["ink"])),
                                     text=[u["nombre"]], textposition="top center", textfont=dict(color=T["ink"], size=13),
                                     hovertemplate="Ingreso %{x} k$ · gasto %{y}<extra></extra>"))
            fig.update_xaxes(title="Ingreso anual (k$)")
            fig.update_yaxes(title="Puntaje de gasto")
            st.plotly_chart(estilo_fig(fig, T, 440, legend=False), width="stretch", config=CFG)
        with g2:
            idx = sorted(centros.index, key=lambda c: u["dist"][c])
            fig = go.Figure(go.Bar(
                y=[NOMBRE[c] for c in idx][::-1], x=[u["dist"][c] for c in idx][::-1], orientation="h",
                marker_color=[COLOR[c] if c == cl else T["line"] for c in idx][::-1],
                text=[f"{u['dist'][c]:.2f}" for c in idx][::-1], textposition="outside", textfont=dict(color=T["ink"]),
                hovertemplate="%{y}<br>distancia = %{x:.3f}<extra></extra>"))
            fig.update_xaxes(title="Distancia euclidiana (escala estandarizada)", range=[0, max(u["dist"]) * 1.25])
            fig.update_yaxes(automargin=True)
            fig.update_layout(title=dict(text="Distancia a cada centroide", font=dict(size=15, color=T["ink"])))
            st.plotly_chart(estilo_fig(fig, T, 440, legend=False), width="stretch", config=CFG)

    if st.session_state.historial:
        st.markdown('<div class="section-title" style="font-size:1.2rem;margin-top:1rem">Clientes clasificados en esta sesión</div>',
                    unsafe_allow_html=True)
        hist = pd.DataFrame(st.session_state.historial)
        st.markdown(tabla_html(hist), unsafe_allow_html=True)
        b1, b2, _ = st.columns([1, 1, 3])
        b1.download_button("Descargar CSV", hist.to_csv(index=False).encode("utf-8"), "clientes_clasificados.csv",
                           "text/csv", width="stretch")
        if b2.button("Borrar historial", width="stretch"):
            st.session_state.historial = []
            st.session_state.pop("ultimo", None)
            st.rerun()

# ──────────────────────────────────────────────────────────────────────────────
# Tab 6: Preguntas del taller
# ──────────────────────────────────────────────────────────────────────────────
PARTES = {
    1: "Parte 1 – Exploración",
    2: "Parte 2 – Preprocesamiento",
    3: "Parte 3 – Método del codo",
    4: "Parte 4 – Coeficiente de silueta",
    5: "Parte 5 – Modelo K-Means",
    6: "Parte 6 – Interpretación de negocio",
}

with tab_preg:
    st.markdown('<div class="section-title">Preguntas del taller</div>'
                '<div class="section-sub">Respuestas a las preguntas de cada parte, tomadas del cuaderno '
                '<code>Taller_KMeans_Segmentacion.ipynb</code>.</div>', unsafe_allow_html=True)
    try:
        preguntas = preguntas_taller(str(NOTEBOOK_PATH), NOTEBOOK_PATH.stat().st_mtime)
    except (OSError, ValueError, KeyError):
        preguntas = {}
        st.warning("No se pudo leer el cuaderno Taller_KMeans_Segmentacion.ipynb.")

    for n, titulo_parte in PARTES.items():
        if not preguntas.get(n):
            continue
        st.markdown(f'<div class="part-title">{titulo_parte}</div>', unsafe_allow_html=True)
        for pregunta, respuesta in preguntas[n]:
            with st.expander(pregunta):
                st.markdown(respuesta)

    st.markdown('<div class="part-title">Parte 7 – Despliegue</div>', unsafe_allow_html=True)
    st.markdown(f"""Esta aplicación de Streamlit pone en funcionamiento el modelo del cuaderno. Carga el dataset,
escala el ingreso y el gasto con `StandardScaler` y entrena K-Means con K = {K_OPTIMO}. En las pestañas se pueden
explorar los datos, revisar el método del codo y el coeficiente de silueta, ver los segmentos con sus perfiles y
estrategias, y clasificar un cliente nuevo a partir de su ingreso y su puntaje de gasto. La barra lateral permite
probar otros valores de K y el selector superior cambia entre modo claro y oscuro.""")

# ──────────────────────────────────────────────────────────────────────────────
# Pie de página
# ──────────────────────────────────────────────────────────────────────────────
st.markdown(f"""<div class="footer"><span>{AUTOR} | {MATERIA}</span>
<span>{UNIVERSIDAD} | Taller final de Minería de Datos | 2026</span></div>""", unsafe_allow_html=True)

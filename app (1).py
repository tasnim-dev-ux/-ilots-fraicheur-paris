"""
🌿 Paris — Équité d'accès aux Îlots de Fraîcheur
Interface Streamlit — Data Science · Open Data Paris
"""

import os
os.environ["OMP_NUM_THREADS"] = "3"

import streamlit as st
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
import warnings
import json

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr, chi2_contingency

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Îlots de Fraîcheur · Paris",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ══════════════════════════════════════════════════════════════
# CUSTOM CSS — Aesthetic: Dark urban · Botanical accent
# ══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;900&family=Newsreader:ital,wght@0,300;0,500;1,300&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #070d09;
    color: #daeade;
}
.stApp { background: #070d09; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0b1510 !important;
    border-right: 1px solid #1a3321 !important;
}
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span {
    color: #8abf9a !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #4ade80 !important;
}

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0c2218 0%, #112d1c 40%, #091a0e 100%);
    border: 1px solid #1e4a2a;
    border-radius: 20px;
    padding: 3.5rem 3rem 3rem 3rem;
    margin-bottom: 2.5rem;
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 380px; height: 380px;
    background: radial-gradient(circle, rgba(74,222,128,0.07) 0%, transparent 65%);
    pointer-events: none;
}
.hero::before {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(134,239,172,0.04) 0%, transparent 70%);
    pointer-events: none;
}
.badge {
    display: inline-block;
    background: rgba(74,222,128,0.12);
    border: 1px solid rgba(74,222,128,0.4);
    color: #4ade80;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 5px 14px;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}
.hero-title {
    font-family: 'Outfit', sans-serif;
    font-size: clamp(2rem, 4vw, 3.4rem);
    font-weight: 900;
    color: #ffffff;
    line-height: 1.08;
    letter-spacing: -1.5px;
    margin: 0 0 0.4rem 0;
}
.hero-title span { color: #4ade80; }
.hero-sub {
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-size: 1.1rem;
    font-weight: 300;
    color: #7ab38a;
    margin-top: 0.6rem;
    max-width: 700px;
    line-height: 1.6;
}
.hyp-row {
    display: flex;
    gap: 1rem;
    margin-top: 2rem;
    flex-wrap: wrap;
}
.hyp-pill {
    background: rgba(0,0,0,0.3);
    border: 1px solid #1e4a2a;
    border-radius: 10px;
    padding: 0.7rem 1.1rem;
    font-size: 0.82rem;
    color: #8abf9a;
    line-height: 1.5;
}
.hyp-pill strong { color: #4ade80; font-weight: 700; }

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin-bottom: 2.5rem;
}
.kpi-card {
    background: #0b1510;
    border: 1px solid #1a3321;
    border-radius: 14px;
    padding: 1.5rem 1.2rem 1.3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: border-color 0.25s, transform 0.2s;
}
.kpi-card:hover {
    border-color: #4ade80;
    transform: translateY(-2px);
}
.kpi-card::before {
    content: attr(data-icon);
    position: absolute;
    top: 0.6rem; right: 0.9rem;
    font-size: 1.4rem;
    opacity: 0.25;
}
.kpi-value {
    font-family: 'Outfit', sans-serif;
    font-size: 2.6rem;
    font-weight: 900;
    color: #4ade80;
    line-height: 1;
    letter-spacing: -1px;
}
.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #4a7a59;
    text-transform: uppercase;
    letter-spacing: 1.8px;
    margin-top: 0.35rem;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #2d5c3d;
    margin-top: 0.2rem;
}

/* ── Section headers ── */
.sec-header {
    font-family: 'Outfit', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #4ade80;
    border-left: 3px solid #4ade80;
    padding-left: 0.9rem;
    margin: 2.5rem 0 0.2rem 0;
    letter-spacing: -0.3px;
}
.sec-sub {
    font-size: 0.87rem;
    color: #4a7a59;
    padding-left: 1.2rem;
    margin-bottom: 1.2rem;
    font-style: italic;
}

/* ── Insight boxes ── */
.insight {
    background: linear-gradient(120deg, #0d1f14, #0f2a1a);
    border: 1px solid #1e4a2a;
    border-left: 4px solid #4ade80;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
    font-size: 0.9rem;
    color: #9fcfaf;
    line-height: 1.75;
}
.insight strong { color: #4ade80; }
.insight .verdict {
    font-family: 'Newsreader', serif;
    font-style: italic;
    font-size: 1.05rem;
    color: #daeade;
    margin-top: 0.5rem;
    display: block;
}

/* ── Stat blocks ── */
.stat-row {
    display: flex;
    gap: 1rem;
    margin: 1rem 0;
    flex-wrap: wrap;
}
.stat-block {
    flex: 1;
    min-width: 140px;
    background: #0b1510;
    border: 1px solid #1a3321;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.stat-block .val {
    font-size: 1.9rem;
    font-weight: 900;
    color: #4ade80;
    font-family: 'Outfit', sans-serif;
}
.stat-block .lbl {
    font-size: 0.72rem;
    color: #4a7a59;
    text-transform: uppercase;
    letter-spacing: 1.5px;
}
.stat-block.warn .val { color: #facc15; }
.stat-block.danger .val { color: #f87171; }
.stat-block.ok .val { color: #4ade80; }

/* ── Top/Flop table ── */
.ranking-table { width: 100%; border-collapse: collapse; font-size: 0.88rem; }
.ranking-table th {
    color: #4a7a59;
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 0.5rem 0.8rem;
    border-bottom: 1px solid #1a3321;
    text-align: left;
}
.ranking-table td {
    padding: 0.6rem 0.8rem;
    border-bottom: 1px solid #0f2018;
    color: #c0dcc8;
}
.ranking-table tr:hover td { background: #0d1f14; }
.rank-top td:first-child { color: #4ade80; font-weight: 700; }
.rank-flop td:first-child { color: #f87171; font-weight: 700; }
.bar-cell { display: flex; align-items: center; gap: 8px; }
.bar-inner { height: 6px; border-radius: 3px; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0b1510 !important;
    border-bottom: 1px solid #1a3321 !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    color: #4a7a59 !important;
    letter-spacing: 0.5px !important;
    padding: 0.7rem 1.4rem !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #4ade80 !important;
    border-bottom-color: #4ade80 !important;
    background: transparent !important;
}

/* ── Chart wrapper ── */
.chart-wrap {
    background: #0b1510;
    border: 1px solid #1a3321;
    border-radius: 14px;
    padding: 1.5rem;
    margin: 0.5rem 0;
}

/* ── Limit box ── */
.limit-box {
    background: rgba(250,204,21,0.05);
    border: 1px solid rgba(250,204,21,0.2);
    border-radius: 10px;
    padding: 1rem 1.3rem;
    font-size: 0.83rem;
    color: #a08030;
    line-height: 1.7;
}
.limit-box strong { color: #facc15; }

/* ── Footer ── */
.footer {
    text-align: center;
    color: #1a3321;
    font-size: 0.75rem;
    padding: 2.5rem 0 1.5rem;
    border-top: 1px solid #1a3321;
    margin-top: 3rem;
    letter-spacing: 1px;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #070d09; }
::-webkit-scrollbar-thumb { background: #1e4a2a; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #4ade80; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# MATPLOTLIB THEME
# ══════════════════════════════════════════════════════════════
BG = "#0b1510"
GRID = "#1a3321"
TEXT = "#8abf9a"
ACCENT = "#4ade80"
WARN = "#facc15"
DANGER = "#f87171"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "xtick.color": TEXT,
    "ytick.color": TEXT,
    "text.color": "#daeade",
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.facecolor": BG,
    "legend.edgecolor": GRID,
    "legend.labelcolor": TEXT,
    "font.family": "DejaVu Sans",
    "figure.dpi": 130,
})

CMAP_GREEN = plt.cm.Greens
PALETTE = [ACCENT, "#22c55e", "#16a34a", "#15803d", "#166534", "#14532d"]

# ══════════════════════════════════════════════════════════════
# POPULATION DATA (INSEE 2021)
# ══════════════════════════════════════════════════════════════
POPULATION = {
    "75001": 16266, "75002": 21174, "75003": 34248, "75004": 27769,
    "75005": 58850, "75006": 41022, "75007": 51367, "75008": 36808,
    "75009": 59555, "75010": 90372, "75011": 146824, "75012": 140115,
    "75013": 181553, "75014": 135964, "75015": 233392, "75016": 166361,
    "75017": 167288, "75018": 195233, "75019": 187015, "75020": 196217,
}

# ══════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🌿 Paramètres")
    st.markdown("---")
    n_clusters = st.slider("Clusters K-Means", min_value=3, max_value=8, value=5,
                           help="Nombre de zones géographiques à identifier")
    show_payant = st.checkbox("Inclure équipements payants", value=True)
    show_raw_data = st.checkbox("Afficher données brutes", value=False)
    st.markdown("---")
    st.markdown("""
    **📡 Source**  
    Open Data Paris  
    `ilots-de-fraicheur-equipements-activites`
    
    **👥 Population**  
    INSEE — Recensement 2021
    
    **🔬 Méthode**  
    CRISP-DM · Spearman · Chi² · Gini · KMeans
    """)
    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#1e4a2a; line-height:2;'>
    v1.0 · Paris Data Science Test<br>
    Données temps réel via API
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════
@st.cache_data(show_spinner=False, ttl=3600)
def load_data():
    all_data = []
    limit = 100
    offset = 0
    max_records = 2000
    base_url = (
        "https://opendata.paris.fr/api/explore/v2.1/catalog/datasets/"
        "ilots-de-fraicheur-equipements-activites/records"
    )
    while offset < max_records:
        try:
            r = requests.get(
                base_url,
                params={"limit": limit, "offset": offset, "select": "*"},
                timeout=15
            )
            data = r.json()
        except Exception:
            break
        if "results" not in data or not data["results"]:
            break
        all_data.extend(data["results"])
        if len(data["results"]) < limit:
            break
        offset += limit

    df = pd.json_normalize(all_data)
    return df


def prepare_data(df_raw, include_payant=True):
    cols_needed = {
        "nom": "nom",
        "type": "type",
        "arrondissement": "arrondissement",
        "payant": "payant",
        "statut_ouverture": "statut_ouverture",
        "geo_point_2d.lat": "lat",
        "geo_point_2d.lon": "lon",
    }
    existing = {k: v for k, v in cols_needed.items() if k in df_raw.columns}
    df = df_raw[list(existing.keys())].rename(columns=existing).copy()
    df["arrondissement"] = df["arrondissement"].astype(str).str.strip()
    df = df[df["arrondissement"].str.startswith("75")]
    df = df.dropna(subset=["lat", "lon", "arrondissement"])
    df["lat"] = pd.to_numeric(df["lat"], errors="coerce")
    df["lon"] = pd.to_numeric(df["lon"], errors="coerce")
    df = df.dropna(subset=["lat", "lon"])
    if not include_payant and "payant" in df.columns:
        df = df[df["payant"].str.lower().str.strip() != "oui"]
    return df


# Load
with st.spinner("🌿 Connexion à l'API Open Data Paris…"):
    df_raw = load_data()

df = prepare_data(df_raw, include_payant=show_payant)

if df.empty:
    st.error("⚠️ Impossible de charger les données. Vérifiez votre connexion Internet.")
    st.stop()

# ── Merge with population ──────────────────────────────
arr_counts = df["arrondissement"].value_counts().reset_index()
arr_counts.columns = ["arrondissement", "nb_ilots"]
arr_counts["population"] = arr_counts["arrondissement"].map(POPULATION)
arr_counts = arr_counts.dropna(subset=["population"])
arr_counts["population"] = arr_counts["population"].astype(int)
arr_counts["ratio"] = (arr_counts["nb_ilots"] / arr_counts["population"] * 10000).round(2)
arr_counts = arr_counts.sort_values("arrondissement")
arr_counts["arr_num"] = arr_counts["arrondissement"].str[-2:].astype(int)
arr_counts = arr_counts.sort_values("arr_num")

# ── KMeans ──────────────────────────────────────────────
coords = df[["lat", "lon"]].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)
kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
df["cluster"] = kmeans.fit_predict(coords_scaled)

# ── Statistical tests ────────────────────────────────────
rho, p_spearman = spearmanr(arr_counts["population"], arr_counts["nb_ilots"])
observed = arr_counts["nb_ilots"].values
total = observed.sum()
pops = arr_counts["population"].values
expected = pops / pops.sum() * total
chi2_stat = ((observed - expected) ** 2 / expected).sum()
from scipy.stats import chi2 as chi2_dist
p_chi2 = 1 - chi2_dist.cdf(chi2_stat, df=len(observed)-1)

# ── Gini ─────────────────────────────────────────────────
def gini(arr):
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return (2 * np.sum(idx * arr) / (n * arr.sum())) - (n + 1) / n

gini_brut = gini(arr_counts["nb_ilots"].values)
gini_norm = gini(arr_counts["ratio"].values)

# ══════════════════════════════════════════════════════════════
# HERO BANNER
# ══════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="hero">
    <div class="badge">🔬 Data Science · Open Data Paris · INSEE 2021</div>
    <div class="hero-title">Paris garantit-elle une<br><span>équité d'accès</span><br>aux îlots de fraîcheur ?</div>
    <div class="hero-sub">
        Analyse statistique de la répartition des {len(df):,} équipements de fraîcheur
        selon les {arr_counts.shape[0]} arrondissements parisiens, croisée avec les données de population.
    </div>
    <div class="hyp-row">
        <div class="hyp-pill"><strong>H₀</strong> — Distribution proportionnelle à la population (équité)</div>
        <div class="hyp-pill"><strong>H₁</strong> — Inégalité significative : certains arrondissements sont structurellement sous-équipés</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# KPI CARDS
# ══════════════════════════════════════════════════════════════
top1 = arr_counts.nlargest(1, "ratio").iloc[0]
flop1 = arr_counts.nsmallest(1, "ratio").iloc[0]
ecart = round(arr_counts["ratio"].max() / arr_counts["ratio"].min(), 1)

st.markdown(f"""
<div class="kpi-grid">
    <div class="kpi-card" data-icon="🌿">
        <div class="kpi-value">{len(df):,}</div>
        <div class="kpi-label">Équipements recensés</div>
        <div class="kpi-sub">via API Open Data Paris</div>
    </div>
    <div class="kpi-card" data-icon="📍">
        <div class="kpi-value">{arr_counts.shape[0]}</div>
        <div class="kpi-label">Arrondissements couverts</div>
        <div class="kpi-sub">sur 20 arrondissements</div>
    </div>
    <div class="kpi-card" data-icon="📊">
        <div class="kpi-value">{round(arr_counts['ratio'].mean(), 2)}</div>
        <div class="kpi-label">Ratio moyen Paris</div>
        <div class="kpi-sub">îlots / 10 000 habitants</div>
    </div>
    <div class="kpi-card" data-icon="⚖️">
        <div class="kpi-value">×{ecart}</div>
        <div class="kpi-label">Écart max / min</div>
        <div class="kpi-sub">{top1['arrondissement']} vs {flop1['arrondissement']}</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Distribution & Équité",
    "🔬 Tests statistiques",
    "🗺️ Géographie & Clusters",
    "🏆 Classements",
    "📋 Données brutes"
])

# ══════════════════════════════════════════════════════════════
# TAB 1 — DISTRIBUTION
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="sec-header">Répartition brute des équipements</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Nombre total d\'équipements par arrondissement — sans pondération par la population</div>', unsafe_allow_html=True)

    col_chart, col_info = st.columns([3, 1])
    with col_chart:
        fig, ax = plt.subplots(figsize=(12, 4.5))
        colors = [ACCENT if r >= arr_counts["nb_ilots"].mean() else "#1e4a2a"
                  for r in arr_counts["nb_ilots"]]
        bars = ax.bar(arr_counts["arr_num"].astype(str), arr_counts["nb_ilots"],
                      color=colors, width=0.72, zorder=3)
        ax.axhline(arr_counts["nb_ilots"].mean(), color=WARN, linewidth=1.5,
                   linestyle="--", label=f"Moyenne : {arr_counts['nb_ilots'].mean():.0f}", zorder=4)
        ax.set_xlabel("Arrondissement (75XXX)", fontsize=9)
        ax.set_ylabel("Nb d'équipements", fontsize=9)
        ax.set_title("Équipements par arrondissement", fontsize=11, fontweight="bold",
                     color=ACCENT, pad=12)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.4, zorder=0)
        ax.set_axisbelow(True)
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.3, str(int(h)),
                    ha="center", va="bottom", fontsize=6.5, color=TEXT)
        plt.xticks(rotation=0, fontsize=8)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

    with col_info:
        n_above = (arr_counts["nb_ilots"] >= arr_counts["nb_ilots"].mean()).sum()
        st.markdown(f"""
        <div class="stat-block ok" style="margin-bottom:0.8rem">
            <div class="val">{arr_counts['nb_ilots'].max()}</div>
            <div class="lbl">Maximum</div>
        </div>
        <div class="stat-block danger" style="margin-bottom:0.8rem">
            <div class="val">{arr_counts['nb_ilots'].min()}</div>
            <div class="lbl">Minimum</div>
        </div>
        <div class="stat-block warn">
            <div class="val">{20 - n_above}</div>
            <div class="lbl">Sous la moyenne</div>
        </div>
        """, unsafe_allow_html=True)

    # Ratio per 10k
    st.markdown('<div class="sec-header">Ratio équipements / 10 000 habitants</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Indicateur clé d\'équité — ce graphique révèle les inégalités réelles après pondération par la population</div>', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ratio_sorted = arr_counts.sort_values("ratio", ascending=True)
    bar_colors = []
    mean_ratio = ratio_sorted["ratio"].mean()
    for r in ratio_sorted["ratio"]:
        if r >= mean_ratio * 1.5:
            bar_colors.append(ACCENT)
        elif r >= mean_ratio:
            bar_colors.append("#22c55e")
        elif r >= mean_ratio * 0.7:
            bar_colors.append("#1e4a2a")
        else:
            bar_colors.append(DANGER)

    bars2 = ax2.barh(ratio_sorted["arr_num"].astype(str),
                     ratio_sorted["ratio"], color=bar_colors, height=0.7, zorder=3)
    ax2.axvline(mean_ratio, color=WARN, linewidth=1.5, linestyle="--",
                label=f"Moyenne : {mean_ratio:.2f}", zorder=4)
    ax2.set_xlabel("Îlots de fraîcheur / 10 000 hab.", fontsize=9)
    ax2.set_ylabel("Arrondissement", fontsize=9)
    ax2.set_title("Ratio normalisé par population (INSEE 2021)", fontsize=11,
                  fontweight="bold", color=ACCENT, pad=12)
    ax2.legend(fontsize=8)
    ax2.grid(axis="x", alpha=0.4, zorder=0)
    ax2.set_axisbelow(True)
    for bar, val in zip(bars2, ratio_sorted["ratio"]):
        ax2.text(val + 0.05, bar.get_y() + bar.get_height()/2,
                 f"{val:.2f}", va="center", fontsize=7, color=TEXT)
    patches = [
        mpatches.Patch(color=ACCENT, label=f"≥ 1.5× moy."),
        mpatches.Patch(color="#22c55e", label="≥ moy."),
        mpatches.Patch(color="#1e4a2a", label="< moy."),
        mpatches.Patch(color=DANGER, label="< 70% moy."),
    ]
    ax2.legend(handles=patches, fontsize=7.5, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    # Gini
    st.markdown('<div class="sec-header">Coefficient de Gini — Mesure de l\'inégalité</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Un Gini à 0 = parfaite égalité. Un Gini à 1 = inégalité totale. Référence : revenu France ≈ 0.29</div>', unsafe_allow_html=True)

    col_gini1, col_gini2, col_gini3 = st.columns(3)
    gini_label = "modérée" if gini_norm < 0.40 else ("significative" if gini_norm < 0.60 else "forte")
    h0_rej = p_chi2 < 0.05

    with col_gini1:
        st.markdown(f"""
        <div class="stat-block warn">
            <div class="val">{gini_brut:.3f}</div>
            <div class="lbl">Gini brut (nb absolu)</div>
        </div>""", unsafe_allow_html=True)
    with col_gini2:
        col = "danger" if gini_norm >= 0.40 else "warn"
        st.markdown(f"""
        <div class="stat-block {col}">
            <div class="val">{gini_norm:.3f}</div>
            <div class="lbl">Gini normalisé (ratio)</div>
        </div>""", unsafe_allow_html=True)
    with col_gini3:
        st.markdown(f"""
        <div class="stat-block {'danger' if h0_rej else 'ok'}">
            <div class="val">{'H₁ ✓' if h0_rej else 'H₀ ✓'}</div>
            <div class="lbl">Verdict hypothèse</div>
        </div>""", unsafe_allow_html=True)

    # Courbe de Lorenz
    fig3, ax3 = plt.subplots(figsize=(6, 5))
    ratios_sorted = np.sort(arr_counts["ratio"].values)
    lorenz = np.cumsum(ratios_sorted) / ratios_sorted.sum()
    lorenz = np.concatenate([[0], lorenz])
    n = len(ratios_sorted)
    x_pts = np.linspace(0, 1, n + 1)
    ax3.plot(x_pts, lorenz, color=ACCENT, linewidth=2.5, label=f"Lorenz (Gini={gini_norm:.3f})")
    ax3.plot([0, 1], [0, 1], color=WARN, linewidth=1.5, linestyle="--", label="Égalité parfaite")
    ax3.fill_between(x_pts, lorenz, x_pts, alpha=0.15, color=ACCENT)
    ax3.set_xlabel("Fraction des arrondissements", fontsize=9)
    ax3.set_ylabel("Fraction des îlots (cumulée)", fontsize=9)
    ax3.set_title("Courbe de Lorenz — Équité de distribution", fontsize=10,
                  fontweight="bold", color=ACCENT, pad=10)
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    plt.tight_layout()

    col_lorenz, col_lorenz_txt = st.columns([1, 1])
    with col_lorenz:
        st.pyplot(fig3)
        plt.close()
    with col_lorenz_txt:
        st.markdown(f"""
        <div class="insight">
        La courbe de Lorenz mesure l'écart entre la distribution réelle et une distribution parfaitement équitable (diagonale jaune).<br><br>
        Un Gini normalisé de <strong>{gini_norm:.3f}</strong> indique une inégalité <strong>{gini_label}</strong>.<br><br>
        <span class="verdict">
        {"→ Les 40% des arrondissements les moins équipés ne concentrent que " + str(round(lorenz[int(0.4*n)]*100, 1)) + "% des équipements totaux." if len(lorenz) > int(0.4*n) else ""}
        </span>
        </div>
        """, unsafe_allow_html=True)

        # Types
        if "type" in df.columns:
            st.markdown('<div class="sec-header" style="margin-top:1.5rem">Types d\'équipements</div>', unsafe_allow_html=True)
            type_counts = df["type"].value_counts().head(8)
            fig_t, ax_t = plt.subplots(figsize=(5, 4))
            cmap_t = plt.cm.Greens
            tc_colors = [cmap_t(0.4 + 0.5 * i / len(type_counts)) for i in range(len(type_counts))][::-1]
            ax_t.barh(type_counts.index[::-1], type_counts.values[::-1], color=tc_colors, height=0.65)
            ax_t.set_title("Top types d'équipements", fontsize=9, color=ACCENT, fontweight="bold")
            ax_t.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig_t)
            plt.close()

# ══════════════════════════════════════════════════════════════
# TAB 2 — STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="sec-header">Test de corrélation de Spearman</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">La population d\'un arrondissement prédit-elle le nombre d\'îlots de fraîcheur ?</div>', unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    rho_color = "ok" if abs(rho) > 0.5 else "warn"
    p_color_s = "danger" if p_spearman < 0.05 else "ok"
    with col_s1:
        st.markdown(f'<div class="stat-block {rho_color}"><div class="val">{rho:.3f}</div><div class="lbl">ρ de Spearman</div></div>', unsafe_allow_html=True)
    with col_s2:
        st.markdown(f'<div class="stat-block {p_color_s}"><div class="val">{p_spearman:.4f}</div><div class="lbl">p-value</div></div>', unsafe_allow_html=True)
    with col_s3:
        verdict_s = "Corrélation significative" if p_spearman < 0.05 else "Pas de corrélation significative"
        st.markdown(f'<div class="stat-block {"ok" if p_spearman < 0.05 else "warn"}"><div class="val">{"✓" if p_spearman < 0.05 else "✗"}</div><div class="lbl">{verdict_s}</div></div>', unsafe_allow_html=True)

    # Scatter plot Spearman
    fig_s, ax_s = plt.subplots(figsize=(8, 5))
    scatter_colors = [ACCENT if r >= mean_ratio else DANGER for r in arr_counts["ratio"]]
    ax_s.scatter(arr_counts["population"], arr_counts["nb_ilots"],
                 c=scatter_colors, s=90, alpha=0.85, zorder=5, edgecolors="#0b1510", linewidths=0.8)
    for _, row in arr_counts.iterrows():
        ax_s.annotate(str(row["arr_num"]), (row["population"], row["nb_ilots"]),
                      textcoords="offset points", xytext=(4, 3),
                      fontsize=7, color=TEXT, alpha=0.8)
    z = np.polyfit(arr_counts["population"], arr_counts["nb_ilots"], 1)
    p_line = np.poly1d(z)
    x_line = np.linspace(arr_counts["population"].min(), arr_counts["population"].max(), 100)
    ax_s.plot(x_line, p_line(x_line), color=WARN, linewidth=1.5, linestyle="--",
              label=f"Tendance linéaire", zorder=4)
    ax_s.set_xlabel("Population (INSEE 2021)", fontsize=9)
    ax_s.set_ylabel("Nombre d'îlots de fraîcheur", fontsize=9)
    ax_s.set_title(f"Population vs Nombre d'îlots — ρ = {rho:.3f} (p = {p_spearman:.4f})",
                   fontsize=10, fontweight="bold", color=ACCENT, pad=10)
    ax_s.legend(fontsize=8)
    ax_s.grid(alpha=0.3)
    ax_s.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}k"))
    plt.tight_layout()
    st.pyplot(fig_s)
    plt.close()

    corr_type = "positive modérée" if rho > 0.3 else "faible voire nulle"
    sig_text = "résultat statistiquement significatif" if p_spearman < 0.05 else "résultat non significatif (pas de preuve d'une corrélation)"
    interp_s = (
        f"ρ = {rho:.3f} : corrélation {corr_type} entre population et équipements. "
        f"p = {p_spearman:.4f} → {sig_text} (seuil α = 0.05)."
    )
    st.markdown(f'<div class="insight">🔬 <strong>Interprétation Spearman :</strong> {interp_s}</div>', unsafe_allow_html=True)

    # Chi2
    st.markdown('<div class="sec-header">Test du Chi² — Équité de distribution</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">La distribution réelle diffère-t-elle significativement d\'une distribution proportionnelle à la population ?</div>', unsafe_allow_html=True)

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        st.markdown(f'<div class="stat-block warn"><div class="val">{chi2_stat:.2f}</div><div class="lbl">Statistique χ²</div></div>', unsafe_allow_html=True)
    with col_c2:
        p_color_c = "danger" if p_chi2 < 0.05 else "ok"
        st.markdown(f'<div class="stat-block {p_color_c}"><div class="val">{p_chi2:.5f}</div><div class="lbl">p-value</div></div>', unsafe_allow_html=True)
    with col_c3:
        h0_text = "H₀ rejetée — Inégalité" if p_chi2 < 0.05 else "H₀ conservée — Équité"
        st.markdown(f'<div class="stat-block {"danger" if p_chi2 < 0.05 else "ok"}"><div class="val">{"✗ H₀" if p_chi2 < 0.05 else "✓ H₀"}</div><div class="lbl">{h0_text}</div></div>', unsafe_allow_html=True)

    # Observed vs Expected
    fig_chi, ax_chi = plt.subplots(figsize=(12, 5))
    x_pos = np.arange(len(arr_counts))
    width = 0.38
    obs_vals = arr_counts.set_index("arr_num").loc[arr_counts["arr_num"], "nb_ilots"].values
    exp_vals_plot = (arr_counts["population"].values / arr_counts["population"].sum() *
                     arr_counts["nb_ilots"].sum())
    ax_chi.bar(x_pos - width/2, arr_counts["nb_ilots"], width=width,
               color=ACCENT, alpha=0.85, label="Observé (réel)", zorder=3)
    ax_chi.bar(x_pos + width/2, exp_vals_plot, width=width,
               color=DANGER, alpha=0.6, label="Attendu (si équité parfaite)", zorder=3)
    ax_chi.set_xticks(x_pos)
    ax_chi.set_xticklabels(arr_counts["arr_num"].astype(str), rotation=0, fontsize=8)
    ax_chi.set_xlabel("Arrondissement", fontsize=9)
    ax_chi.set_ylabel("Nombre d'équipements", fontsize=9)
    ax_chi.set_title("Réel vs Attendu (distribution proportionnelle à la population)",
                     fontsize=10, fontweight="bold", color=ACCENT, pad=10)
    ax_chi.legend(fontsize=8)
    ax_chi.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig_chi)
    plt.close()

    if p_chi2 < 0.05:
        interp_chi = f"χ² = {chi2_stat:.2f}, p = {p_chi2:.5f} < 0.05 → <strong>H₀ rejetée</strong>. La distribution des îlots de fraîcheur n'est PAS proportionnelle à la population. Paris présente une inégalité d'accès statistiquement significative."
    else:
        interp_chi = f"χ² = {chi2_stat:.2f}, p = {p_chi2:.5f} ≥ 0.05 → <strong>H₀ conservée</strong>. On ne peut pas rejeter l'équité de distribution au seuil de 5%."
    st.markdown(f'<div class="insight">🔬 <strong>Interprétation Chi² :</strong> {interp_chi}</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 3 — GÉOGRAPHIE & CLUSTERS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="sec-header">K-Means — Méthode du coude</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Déterminer le nombre optimal de clusters géographiques</div>', unsafe_allow_html=True)

    # Elbow
    inertias = []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(coords_scaled)
        inertias.append(km.inertia_)

    fig_elbow, ax_elbow = plt.subplots(figsize=(8, 4))
    ax_elbow.plot(list(k_range), inertias, color=ACCENT, linewidth=2.5,
                  marker="o", markersize=7, markerfacecolor=WARN, zorder=5)
    ax_elbow.axvline(n_clusters, color=WARN, linewidth=1.5, linestyle="--",
                     label=f"k sélectionné = {n_clusters}")
    ax_elbow.set_xlabel("Nombre de clusters (k)", fontsize=9)
    ax_elbow.set_ylabel("Inertie", fontsize=9)
    ax_elbow.set_title("Méthode du coude — Choix optimal de k", fontsize=10,
                       fontweight="bold", color=ACCENT, pad=10)
    ax_elbow.legend(fontsize=8)
    ax_elbow.grid(alpha=0.3)
    plt.tight_layout()

    col_elbow, col_elbow_txt = st.columns([2, 1])
    with col_elbow:
        st.pyplot(fig_elbow)
        plt.close()
    with col_elbow_txt:
        st.markdown(f"""
        <div class="insight">
        Ajustez le curseur <strong>«Clusters K-Means»</strong> dans la barre latérale pour tester différentes valeurs de k.<br><br>
        Actuellement : <strong>k = {n_clusters}</strong><br><br>
        La méthode du coude identifie le point où ajouter un cluster n'améliore plus significativement la segmentation.
        </div>
        """, unsafe_allow_html=True)

    # Scatter géographique
    st.markdown('<div class="sec-header">Cartographie des clusters géographiques</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Chaque couleur représente une zone géographique naturelle identifiée par l\'algorithme</div>', unsafe_allow_html=True)

    CLUSTER_COLORS = [ACCENT, DANGER, WARN, "#60a5fa", "#c084fc", "#fb923c", "#34d399", "#f472b6"]

    fig_geo, ax_geo = plt.subplots(figsize=(10, 8))
    for c in range(n_clusters):
        mask = df["cluster"] == c
        ax_geo.scatter(df.loc[mask, "lon"], df.loc[mask, "lat"],
                       c=CLUSTER_COLORS[c], s=18, alpha=0.75, label=f"Cluster {c+1}",
                       zorder=4, edgecolors="none")
    # Centroids
    centers_orig = scaler.inverse_transform(kmeans.cluster_centers_)
    ax_geo.scatter(centers_orig[:, 1], centers_orig[:, 0],
                   c="white", s=120, marker="*", zorder=6, linewidths=1,
                   edgecolors="#0b1510", label="Centroïdes")
    ax_geo.set_xlabel("Longitude", fontsize=9)
    ax_geo.set_ylabel("Latitude", fontsize=9)
    ax_geo.set_title(f"Distribution géographique — {n_clusters} clusters (KMeans)",
                     fontsize=11, fontweight="bold", color=ACCENT, pad=12)
    ax_geo.legend(fontsize=8, markerscale=1.5)
    ax_geo.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig_geo)
    plt.close()

    # Cluster sizes
    cluster_sizes = df["cluster"].value_counts().sort_index()
    fig_cs, ax_cs = plt.subplots(figsize=(6, 3))
    ax_cs.bar([f"Cluster {i+1}" for i in cluster_sizes.index],
              cluster_sizes.values,
              color=[CLUSTER_COLORS[i] for i in cluster_sizes.index],
              width=0.65, zorder=3)
    ax_cs.set_title("Taille des clusters", fontsize=9, color=ACCENT, fontweight="bold")
    ax_cs.set_ylabel("Équipements", fontsize=8)
    ax_cs.grid(axis="y", alpha=0.3)
    for i, (bar, val) in enumerate(zip(ax_cs.patches, cluster_sizes.values)):
        ax_cs.text(bar.get_x() + bar.get_width()/2, val + 1, str(val),
                   ha="center", va="bottom", fontsize=8, color=TEXT)
    plt.tight_layout()

    col_cs, col_cs_txt = st.columns([1, 1])
    with col_cs:
        st.pyplot(fig_cs)
        plt.close()
    with col_cs_txt:
        st.markdown(f"""
        <div class="insight">
        Le clustering K-Means identifie <strong>{n_clusters} zones géographiques naturelles</strong>
        de concentration des équipements.<br><br>
        Les clusters déséquilibrés en taille révèlent des zones denses (centre) 
        et des zones éparses (périphérie), indépendamment des frontières administratives.<br><br>
        ⚠️ <em>Le clustering géographique ne tient pas compte des frontières d'arrondissements.</em>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 4 — CLASSEMENTS
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="sec-header">🏆 Classement complet des arrondissements</div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-sub">Trié par ratio d\'équipements / 10 000 habitants (indicateur d\'équité)</div>', unsafe_allow_html=True)

    full_rank = arr_counts.sort_values("ratio", ascending=False).reset_index(drop=True)
    full_rank["rank"] = range(1, len(full_rank) + 1)
    max_ratio = full_rank["ratio"].max()

    rows_html = ""
    for _, row in full_rank.iterrows():
        pct = row["ratio"] / max_ratio * 100
        rank_n = int(row["rank"])
        is_top = rank_n <= 3
        is_flop = rank_n > len(full_rank) - 3
        row_class = "rank-top" if is_top else ("rank-flop" if is_flop else "")
        bar_color = ACCENT if is_top else (DANGER if is_flop else "#1e4a2a")
        medal = "🥇" if rank_n == 1 else ("🥈" if rank_n == 2 else ("🥉" if rank_n == 3 else f"#{rank_n}"))
        rows_html += f"""
        <tr class="{row_class}">
            <td>{medal}</td>
            <td>{row['arrondissement']}</td>
            <td>{int(row['nb_ilots'])}</td>
            <td>{int(row['population']):,}</td>
            <td>
                <div class="bar-cell">
                    <div class="bar-inner" style="width:{pct:.0f}px; background:{bar_color};"></div>
                    <span style="color:{'#4ade80' if is_top else ('#f87171' if is_flop else '#4a7a59')}">{row['ratio']:.2f}</span>
                </div>
            </td>
        </tr>"""

    st.markdown(f"""
    <div class="chart-wrap">
    <table class="ranking-table">
    <thead><tr>
        <th>Rang</th>
        <th>Arrondissement</th>
        <th>Équipements</th>
        <th>Population</th>
        <th>Ratio / 10k hab.</th>
    </tr></thead>
    <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # Summary verdict
    top3_names = full_rank.head(3)["arrondissement"].tolist()
    flop3_names = full_rank.tail(3)["arrondissement"].tolist()
    st.markdown(f"""
    <div class="insight">
    🟢 <strong>TOP 3 les mieux dotés :</strong> {', '.join(top3_names)}<br>
    🔴 <strong>FLOP 3 les moins dotés :</strong> {', '.join(flop3_names)}<br>
    📊 <strong>Écart max/min :</strong> ×{ecart} — {top1['arrondissement']} dispose de {top1['ratio']:.2f} îlots/10k hab.
    contre seulement {flop1['ratio']:.2f} pour {flop1['arrondissement']}<br>
    ⚖️ <strong>Arrondissements sous la moyenne :</strong> {(full_rank['ratio'] < full_rank['ratio'].mean()).sum()} sur {len(full_rank)}<br>
    <span class="verdict">
    {"→ H₀ rejetée : la répartition n'est pas équitable. " if p_chi2 < 0.05 else "→ H₀ conservée : distribution compatible avec l'équité. "}
    Gini normalisé = {gini_norm:.3f} (inégalité {gini_label}).
    </span>
    </div>
    """, unsafe_allow_html=True)

    # Recommendations
    st.markdown('<div class="sec-header">💡 Recommandations</div>', unsafe_allow_html=True)
    recs = [
        ("🎯 Priorité haute", f"Renforcer {', '.join(flop3_names)} — les arrondissements les plus peuplés et les moins dotés en ratio."),
        ("🌿 Diversification", "Certains arrondissements dépendent d'un seul type d'équipement. Diversifier l'offre (fontaines, parcs, bâtiments publics)."),
        ("🗺️ Zones blanches", f"Les {n_clusters} clusters géographiques révèlent des zones sans équipements — à prioriser dans les futurs aménagements."),
        ("📈 Suivi annuel", "Mettre en place un indicateur annuel ratio/habitant par arrondissement pour mesurer la progression vers l'équité."),
    ]
    for title, body in recs:
        st.markdown(f'<div class="insight"><strong>{title}</strong><br>{body}</div>', unsafe_allow_html=True)

    # Limits
    st.markdown('<div class="sec-header">⚠️ Limites de l\'étude</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="limit-box">
    <strong>Données :</strong> Le dataset peut être incomplet — équipements non référencés dans l'Open Data non pris en compte.<br>
    <strong>Population :</strong> Source INSEE 2021 — stable mais ne reflète pas les mouvements récents de population.<br>
    <strong>Capacité :</strong> Un équipement ≠ un autre (une grande piscine ≠ une fontaine Wallace). L'analyse ne pondère pas la surface ou la capacité d'accueil.<br>
    <strong>Frontières :</strong> La proximité géographique des arrondissements signifie qu'un équipement peut servir plusieurs arrondissements adjacents.<br>
    <strong>Payant/Gratuit :</strong> Utilisez le filtre dans la barre latérale pour exclure les équipements payants.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# TAB 5 — RAW DATA
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown('<div class="sec-header">Données brutes — Tableau complet</div>', unsafe_allow_html=True)

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        search_term = st.text_input("🔍 Rechercher un équipement", placeholder="ex: piscine, fontaine, parc...")
    with col_f2:
        arr_filter = st.multiselect("Filtrer par arrondissement",
                                    options=sorted(df["arrondissement"].unique()),
                                    default=[])

    df_display = df.copy()
    if search_term and "nom" in df_display.columns:
        df_display = df_display[df_display["nom"].str.contains(search_term, case=False, na=False)]
    if arr_filter:
        df_display = df_display[df_display["arrondissement"].isin(arr_filter)]

    st.markdown(f"**{len(df_display):,} équipements** affichés")
    st.dataframe(
        df_display.drop(columns=["cluster"], errors="ignore"),
        use_container_width=True,
        height=450
    )

    # Merge table
    st.markdown('<div class="sec-header" style="margin-top:2rem">Tableau synthèse — Arrondissements</div>', unsafe_allow_html=True)
    st.dataframe(
        arr_counts.drop(columns=["arr_num"]).rename(columns={
            "arrondissement": "Arrondissement",
            "nb_ilots": "Nb équipements",
            "population": "Population (INSEE 2021)",
            "ratio": "Ratio / 10 000 hab."
        }).sort_values("Ratio / 10 000 hab.", ascending=False).reset_index(drop=True),
        use_container_width=True,
        height=350
    )

# ══════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════
st.markdown("""
<div class="footer">
    🌿 ÎLOTS DE FRAÎCHEUR · PARIS · DATA SCIENCE TEST<br>
    Source : Open Data Paris · Population : INSEE 2021 · Méthode : CRISP-DM
</div>
""", unsafe_allow_html=True)

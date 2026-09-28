"""
dashboard.py — Dashboard de détection de fraude (Streamlit + MongoDB)
Installation : pip install streamlit pymongo pandas plotly numpy
Lancement    : streamlit run dashboard.py
"""
import time
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from pymongo import MongoClient

MONGO_URI, DB_NAME = "mongodb://localhost:27017", "fraud_detection"
ROUGE, MARINE, GRIS, CLAIR = "#E94E5A", "#1B2A4A", "#C5CBD8", "#F4F6FB"

st.set_page_config(page_title="FraudGuard", page_icon="🛡️", layout="wide")

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {{ font-family:'Poppins',sans-serif; }}
.stApp {{ background:{CLAIR}; }}
header[data-testid="stHeader"] {{ background:transparent; }}
[data-testid="stSidebar"] {{ background:{MARINE}; }}
[data-testid="stSidebar"] * {{ color:#fff; }}
[data-testid="stSidebar"] input {{ color:{MARINE} !important; }}
[data-testid="stSidebar"] div[role="radiogroup"] label {{
  padding:10px 14px; border-radius:10px; margin-bottom:4px; transition:.2s; }}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{ background:rgba(255,255,255,.1); }}
.brand {{ font-size:1.4rem; font-weight:700; margin-bottom:.2rem; }}
.brand span {{ color:{ROUGE}; }}
.hero {{ background:linear-gradient(120deg,{MARINE} 0%,#2B3F6B 60%,{ROUGE} 130%);
  border-radius:18px; padding:28px 34px; color:#fff; margin-bottom:22px; }}
.hero small {{ color:{ROUGE}; font-weight:600; letter-spacing:.12em; text-transform:uppercase; }}
.hero h1 {{ color:#fff; font-size:1.9rem; margin:.2rem 0 0; padding:0; }}
.hero p {{ color:#D5DAE8; margin:.3rem 0 0; }}
.kpi {{ background:#fff; border-radius:14px; padding:16px 18px; border-top:4px solid {ROUGE};
  box-shadow:0 4px 14px rgba(27,42,74,.08); }}
.kpi .l {{ color:#6B7590; font-size:.78rem; font-weight:500; }}
.kpi .v {{ color:{MARINE}; font-size:1.7rem; font-weight:700; line-height:1.2; }}
.kpi .s {{ color:{ROUGE}; font-size:.75rem; font-weight:500; }}
[data-testid="stVerticalBlockBorderWrapper"] {{ background:#fff; border-radius:16px;
  border:1px solid #E6EAF2; box-shadow:0 4px 14px rgba(27,42,74,.06); }}
.ct {{ color:{MARINE}; font-weight:600; font-size:1.05rem; border-left:4px solid {ROUGE};
  padding-left:10px; margin-bottom:6px; }}
.defn {{ background:#FDF1F2; border-radius:10px; padding:10px 14px; color:#4A5470;
  font-size:.82rem; margin-top:4px; }}
.defn b {{ color:{ROUGE}; }}
</style>""", unsafe_allow_html=True)


@st.cache_data(ttl=5)
def charger():
    col = MongoClient(MONGO_URI)[DB_NAME]["toutes_transactions"]
    df = pd.DataFrame(list(col.find({}, {"_id": 0, "hour_sin": 0, "hour_cos": 0})))
    if not df.empty:
        df["heure"] = ((df["Time"] // 3600) % 24).astype(int)
        df["h_flux"] = (df["Time"] // 3600).astype(int)
    return df


def moment(t):
    return f"Jour {int(t // 86400) + 1} · {int(t % 86400 // 3600):02d}:{int(t % 3600 // 60):02d}"


def style(fig, h=340):
    fig.update_layout(template="plotly_white", height=h, margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Poppins", color=MARINE, size=12),
        legend=dict(orientation="h", y=1.12, x=0), bargap=.18)
    fig.update_xaxes(showgrid=False, linecolor=GRIS)
    fig.update_yaxes(gridcolor="#EDF0F5", zeroline=False)
    return fig


def carte(titre, fig, definition, extra=""):
    with st.container(border=True):
        st.markdown(f'<div class="ct">{titre}</div>', unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        if extra:
            st.markdown(f"**{extra}**")
        st.markdown(f'<div class="defn"><b>Définition —</b> {definition}</div>', unsafe_allow_html=True)


def kpi(l, v, s=""):
    return f'<div class="kpi"><div class="l">{l}</div><div class="v">{v}</div><div class="s">{s}&nbsp;</div></div>'


# ---------------- Sidebar ----------------
st.sidebar.markdown('<div class="brand">🛡️ Fraud<span>Guard</span></div>', unsafe_allow_html=True)
st.sidebar.caption("Détection de fraude bancaire")
page = st.sidebar.radio("Navigation", ["📊  Tableau de bord", "🕒  Alertes par heure",
                                       "📈  Distributions"], label_visibility="collapsed")
st.sidebar.markdown("---")
seuil = st.sidebar.slider("Seuil de décision", 0.05, 0.99, 0.50, 0.01)
montant_min = st.sidebar.number_input("Montant minimum", min_value=0.0, value=0.0)
auto = st.sidebar.checkbox("Rafraîchir toutes les 5 s")

df = charger()
if df.empty:
    st.warning("Aucune donnée dans toutes_transactions. Lancez le producer et le consumer.")
    st.stop()
df = df[df["Amount"] >= montant_min].copy()
df["alerte"] = (df["proba_fraude"] >= seuil).astype(int)

total, alertes = len(df), int(df["alerte"].sum())
vraies = int(((df.alerte == 1) & (df.Class == 1)).sum())
reelles = int((df.Class == 1).sum())
prec = vraies / alertes if alertes else 0
rec = vraies / reelles if reelles else 0

titres = {"📊": ("Vue d'ensemble", "Tableau de bord"), "🕒": ("Analyse temporelle", "Alertes par heure"),
          "📈": ("Analyse statistique", "Distributions")}
sur, tit = titres[page[0:1] if page[0] in titres else page[:2].strip()]
st.markdown(f'<div class="hero"><small>{sur}</small><h1>{tit}</h1>'
            f'<p>Surveillance en temps réel des transactions — seuil actuel : {seuil:.2f}</p></div>',
            unsafe_allow_html=True)

# ---------------- Page 1 ----------------
if page.startswith("📊"):
    k = st.columns(5)
    for c, (l, v, s) in zip(k, [
        ("Transactions traitées", f"{total:,}", ""),
        ("Alertes fraude", f"{alertes:,}", f"{alertes / total:.2%} du flux"),
        ("Montant alerté", f"{df.loc[df.alerte == 1, 'Amount'].sum():,.0f}", ""),
        ("Recall", f"{rec:.1%}", "vraies fraudes détectées"),
        ("Précision", f"{prec:.1%}", "alertes réellement frauduleuses")]):
        c.markdown(kpi(l, v, s), unsafe_allow_html=True)
    st.write("")
    g, d = st.columns([3, 2])
    with g:
        with st.container(border=True):
            st.markdown('<div class="ct">Dernières alertes</div>', unsafe_allow_html=True)
            t = df[df.alerte == 1].sort_values("Time", ascending=False).head(20)
            aff = pd.DataFrame({"Transaction": t.transaction_id.str[:8] + "…",
                                "Moment": t.Time.map(moment), "Montant": t.Amount,
                                "Probabilité": t.proba_fraude,
                                "Vérité": t.Class.map({1: "✅ Vraie fraude", 0: "⚠️ Fausse alerte"})})
            st.dataframe(aff, hide_index=True, use_container_width=True, height=390, column_config={
                "Montant": st.column_config.NumberColumn(format="%.2f"),
                "Probabilité": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
            st.markdown('<div class="defn"><b>Définition —</b> transactions dont la probabilité '
                        'de fraude dépasse le seuil ; la colonne « Vérité » compare avec la réalité.</div>',
                        unsafe_allow_html=True)
    with d:
        cm = pd.crosstab(df.Class, df.alerte).reindex(index=[0, 1], columns=[0, 1], fill_value=0).values
        fig = go.Figure(go.Heatmap(z=np.log1p(cm), text=cm, texttemplate="<b>%{text:,}</b>",
            textfont=dict(size=20), x=["Prédite normale", "Prédite fraude"],
            y=["Réelle normale", "Réelle fraude"], showscale=False, xgap=4, ygap=4,
            colorscale=[[0, "#FFFFFF"], [1, ROUGE]], hoverinfo="skip"))
        fig.update_yaxes(autorange="reversed")
        carte("Matrice de confusion", style(fig, 390),
              "compare les prédictions du modèle à la réalité : la diagonale = bonnes décisions, "
              "en bas à gauche = fraudes manquées, en haut à droite = fausses alertes.")

# ---------------- Page 2 ----------------
elif page.startswith("🕒"):
    h = df[df.alerte == 1].groupby("heure").size().reindex(range(24), fill_value=0)
    lab = [f"{i:02d}h" for i in h.index]
    fig = go.Figure(go.Bar(x=lab, y=h.values, marker_color=[ROUGE if v == h.max() else "#F3A0A6" for v in h.values],
                           hovertemplate="%{x} – %{y} alertes<extra></extra>"))
    fig.update_xaxes(title="Heure de la journée")
    fig.update_yaxes(title="Nombre d'alertes")
    pic = int(h.idxmax())
    carte("Alertes par heure de la journée", style(fig),
          "nombre d'alertes fraude selon l'heure à laquelle la transaction a eu lieu (00h = minuit). "
          "Il révèle les créneaux horaires les plus risqués.",
          f"Pic d'alertes : {pic:02d}h–{pic + 1:02d}h ({int(h.max())} alertes)")
    st.write("")
    f = df[df.alerte == 1].groupby("h_flux").size().reindex(range(int(df.h_flux.max()) + 1), fill_value=0)
    lab2 = [f"J{i // 24 + 1} · {i % 24:02d}h" for i in f.index]
    fig = go.Figure(go.Scatter(x=lab2, y=f.values, mode="lines+markers", line=dict(color=ROUGE, width=3),
        marker=dict(size=6, color=MARINE), fill="tozeroy", fillcolor="rgba(233,78,90,.12)",
        hovertemplate="%{x} – %{y} alertes<extra></extra>"))
    fig.update_xaxes(tickmode="array", tickvals=lab2[::max(1, len(lab2) // 12)], tickangle=-35, title="Jour · heure")
    fig.update_yaxes(title="Nombre d'alertes")
    carte("Évolution des alertes dans le temps", style(fig),
          "nombre d'alertes heure après heure depuis le début du flux (J1 = premier jour). "
          "Il permet de repérer les pics anormaux.")

# ---------------- Page 3 ----------------
else:
    df["Type"] = df.alerte.map({1: "Alerte fraude", 0: "Normale"})
    fig = px.histogram(df, x="Amount", color="Type", nbins=60, log_y=True, barmode="overlay", opacity=.8,
                       color_discrete_map={"Normale": GRIS, "Alerte fraude": ROUGE})
    fig.update_xaxes(title="Montant de la transaction")
    fig.update_yaxes(title="Nombre de transactions (échelle log)")
    carte("Distribution des montants", style(fig),
          "répartition des transactions selon leur montant, séparant les alertes fraude des transactions "
          "normales. L'échelle logarithmique rend visibles les fraudes, très minoritaires.")
    st.write("")
    fig = px.histogram(df, x="proba_fraude", nbins=50, log_y=True, color_discrete_sequence=[MARINE])
    fig.add_vline(x=seuil, line_dash="dash", line_color=ROUGE, line_width=2,
                  annotation_text=f"Seuil {seuil:.2f}", annotation_font_color=ROUGE)
    fig.update_xaxes(title="Probabilité de fraude prédite (0 à 1)")
    fig.update_yaxes(title="Nombre de transactions (échelle log)")
    carte("Distribution des probabilités de fraude", style(fig),
          "score de risque attribué par le modèle à chaque transaction. À droite du seuil (ligne rouge), "
          "la transaction déclenche une alerte ; plus les scores sont tranchés, plus le modèle est confiant.")

if auto:
    time.sleep(5)
    st.rerun()
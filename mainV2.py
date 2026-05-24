import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.corpus import stopwords
import re

nltk.download('stopwords', quiet=True)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taylor Swift Dashboard",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background-color: #fafafa; }
    h1 { font-weight: 700; letter-spacing: -0.5px; }
    h2 { font-weight: 600; color: #1a1a1a; }
    h3 { font-weight: 500; color: #333; }
    .era-card {
        border-radius: 12px;
        padding: 14px 16px;
        margin: 4px 0;
        border-left: 4px solid;
        transition: opacity 0.2s;
    }
    .era-title { font-size: 0.85rem; font-weight: 600; margin: 0; }
    .era-year  { font-size: 0.75rem; font-weight: 400; opacity: 0.7; margin: 2px 0 0 0; }
    .era-meta  { font-size: 0.72rem; opacity: 0.65; margin: 4px 0 0 0; }
    .info-box {
        background: #f0f4ff;
        border-left: 3px solid #4f7cff;
        padding: 10px 14px;
        border-radius: 6px;
        font-size: 0.85rem;
        color: #333;
        margin-bottom: 16px;
    }
    .stat-row { display: flex; gap: 24px; margin: 12px 0; }
    .stat-box { background: white; border: 1px solid #e8e8e8; border-radius: 10px; padding: 14px 20px; flex: 1; }
    .stat-num { font-size: 1.8rem; font-weight: 700; color: #1a1a1a; }
    .stat-label { font-size: 0.75rem; color: #888; margin-top: 2px; }
    section[data-testid="stSidebar"] { background: #fff; border-right: 1px solid #f0f0f0; }
    div[data-testid="stMetric"] { background: white; border: 1px solid #f0f0f0; border-radius: 10px; padding: 12px; }
    .stTabs [data-baseweb="tab"] { font-size: 0.85rem; }
</style>
""", unsafe_allow_html=True)

# ── Era data ──────────────────────────────────────────────────────────────────
ERA_COLORS = {
    "Taylor Swift":                      "#6B8E6B",
    "The Taylor Swift Holiday Collection":"#C84B4B",
    "Beautiful Eyes":                    "#8E6B8E",
    "Fearless":                          "#D4A843",
    "Speak Now":                         "#7B4B8E",
    "Red":                               "#B22222",
    "1989":                              "#7BA8C4",
    "reputation":                        "#3a3a3a",
    "Lover":                             "#E8709A",
    "folklore":                          "#7a7a6a",
    "evermore":                          "#8B6914",
    "Fearless (Taylor's Version)":       "#C49A2A",
    "Red (Taylor's Version)":            "#8B1A1A",
    "Midnights":                         "#2C3E6B",
    "Speak Now (Taylor's Version)":      "#5B2D7A",
    "1989 (Taylor's Version)":           "#5A98C4",
    "THE TORTURED POETS DEPARTMENT":     "#8a8078",
    "The Life of a Showgirl":            "#C8A882",
}

ERA_EMOJI = {
    "Taylor Swift": "🤠", "The Taylor Swift Holiday Collection": "🎄",
    "Beautiful Eyes": "👁️", "Fearless": "✨", "Speak Now": "💜",
    "Red": "❤️", "1989": "🌃", "reputation": "🐍", "Lover": "🌈",
    "folklore": "🌲", "evermore": "🍂", "Fearless (Taylor's Version)": "✨",
    "Red (Taylor's Version)": "❤️", "Midnights": "🌙",
    "Speak Now (Taylor's Version)": "💜", "1989 (Taylor's Version)": "🌃",
    "THE TORTURED POETS DEPARTMENT": "🖋️", "The Life of a Showgirl": "🎭",
}

TV_PAIRS = {
    "Fearless": "Fearless (Taylor's Version)",
    "Red":      "Red (Taylor's Version)",
    "Speak Now":"Speak Now (Taylor's Version)",
    "1989":     "1989 (Taylor's Version)",
}

AUDIO_FEATURES = ["danceability", "energy", "acousticness", "valence"]
FEATURE_LABELS = {
    "danceability": "Bailabilidad",
    "energy":       "Energía",
    "acousticness": "Acústica",
    "valence":      "Positividad",
    "loudness":     "Volumen",
    "tempo":        "Tempo",
}
FEATURE_DESC = {
    "danceability": "Qué tan bailable es la canción (0 = nada bailable, 1 = muy bailable)",
    "energy":       "Intensidad y actividad percibida (0 = suave, 1 = muy energética)",
    "acousticness": "Probabilidad de ser acústica (1 = completamente acústica)",
    "valence":      "Positividad musical (0 = triste/oscura, 1 = feliz/eufórica)",
}

def hex_to_rgba(hex_color, alpha=0.25):
    hex_color = hex_color.lstrip('#')
    if len(hex_color) == 3:
        hex_color = ''.join([c*2 for c in hex_color])
    r, g, b = int(hex_color[0:2],16), int(hex_color[2:4],16), int(hex_color[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    songs  = pd.read_csv("Taylor_Songs_Clean.csv")
    albums = pd.read_csv("Taylor Albums.csv")
    songs["album_release"]  = pd.to_datetime(songs["album_release"],  errors="coerce")
    albums["album_release"] = pd.to_datetime(albums["album_release"], errors="coerce")
    songs["color"]      = songs["album_name"].map(ERA_COLORS).fillna("#888")
    songs["has_lyrics"] = songs["lyrics"].notna() & (songs["lyrics"].astype(str).str.strip() != "") & (songs["lyrics"].astype(str) != "nan")
    return songs, albums

songs_df, albums_df = load_data()
ALBUM_ORDER = albums_df.sort_values("album_release")["album_name"].tolist()
DATA_ALBUMS = [a for a in ALBUM_ORDER if a in songs_df["album_name"].unique()]

# ── Helpers ───────────────────────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip(): return None
    return analyzer.polarity_scores(text)["compound"]

def lexical_diversity(text):
    if not isinstance(text, str) or not text.strip(): return None
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    return round(len(set(tokens)) / len(tokens), 3) if len(tokens) >= 10 else None

def clean_for_wc(text):
    stop = set(stopwords.words("english")) | {"oh","like","yeah","know","just","got","get","gonna","wanna","cause","let","say","ooh","ah","la","da","uh","mm","hey","baby","back","come","want","need"}
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return " ".join(t for t in tokens if t not in stop)

def make_radar(albums_data, feature_list):
    """Makes a radar chart; returns fig. albums_data = [(name, color, means_series)]"""
    fig = go.Figure()
    labels = [FEATURE_LABELS[f] for f in feature_list] + [FEATURE_LABELS[feature_list[0]]]
    for name, color, means in albums_data:
        vals = [means.get(f, 0) for f in feature_list] + [means.get(feature_list[0], 0)]
        fig.add_trace(go.Scatterpolar(
            r=vals, theta=labels, fill="toself", name=name,
            line=dict(color=color, width=2),
            opacity=0.75,
        ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1], gridcolor="#eee"),
                   angularaxis=dict(gridcolor="#eee")),
        showlegend=True, height=340,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center"),
        margin=dict(l=50,r=50,t=30,b=60),
    )
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎸 Taylor Swift")
    st.markdown("**Dashboard v2** · *All eras, all songs*")
    st.divider()
    section = st.radio("Ir a", [
        "🌟 Eras",
        "📊 Audio por álbum",
        "📝 Análisis de lyrics",
        "🔄 OG vs Taylor's Version",
        "🎙️ Podcast Swifting",
    ], label_visibility="collapsed")
    st.divider()
    st.caption(f"📀 {len(songs_df)} canciones · {songs_df['album_name'].nunique()} álbumes")
    st.caption(f"📝 {songs_df['has_lyrics'].sum()} con lyrics")

# ══════════════════════════════════════════════════════════════════════════════
# 1 — ERAS
# ══════════════════════════════════════════════════════════════════════════════
if section == "🌟 Eras":
    st.markdown("# 🌟 La Discografía de Taylor Swift")
    st.markdown("De pequeña ciudad en Pennsylvania a fenómeno global — 18 proyectos, 325 canciones, casi 20 años.")

    st.markdown("""
    <div class="info-box">
    Taylor Swift es la única artista en la historia con 6 álbumes que debutaron en #1 en Billboard 200 en la misma semana de lanzamiento.
    Cada era tiene una estética, sonido y paleta emocional completamente distinta.
    </div>
    """, unsafe_allow_html=True)

    # Grid de eras
    ordered = albums_df[albums_df["album_name"].isin(DATA_ALBUMS)].sort_values("album_release")
    cols_per_row = 6
    for i in range(0, len(ordered), cols_per_row):
        chunk = ordered.iloc[i:i+cols_per_row]
        cols = st.columns(len(chunk))
        for col, (_, alb) in zip(cols, chunk.iterrows()):
            name  = alb["album_name"]
            year  = pd.to_datetime(alb["album_release"]).year if pd.notna(alb["album_release"]) else "?"
            color = ERA_COLORS.get(name, "#888")
            emoji = ERA_EMOJI.get(name, "🎵")
            n_songs = len(songs_df[songs_df["album_name"] == name])
            score = alb.get("metacritic_score")
            score_str = f"⭐ {int(score)}" if pd.notna(score) else ""
            with col:
                st.markdown(f"""
                <div class="era-card" style="border-color:{color};background:{hex_to_rgba(color,0.06)}">
                    <div style="font-size:1.4rem">{emoji}</div>
                    <p class="era-title" style="color:{color}">{name}</p>
                    <p class="era-year">{year} · {n_songs} canciones</p>
                    <p class="era-meta">{score_str}</p>
                </div>
                """, unsafe_allow_html=True)

    st.divider()

    # Timeline
    st.markdown("### Cronología")
    st.caption("Cada círculo representa un álbum. El tamaño indica cuántas canciones tiene.")
    tl_data = []
    for _, alb in ordered.iterrows():
        n = len(songs_df[songs_df["album_name"] == alb["album_name"]])
        tl_data.append({"Álbum": alb["album_name"], "Fecha": alb["album_release"],
                         "Canciones": n, "Color": ERA_COLORS.get(alb["album_name"],"#888")})
    tl = pd.DataFrame(tl_data).dropna(subset=["Fecha"])
    fig_tl = px.scatter(
        tl, x="Fecha", y=[0]*len(tl), size="Canciones",
        color="Álbum", color_discrete_map={r["Álbum"]:r["Color"] for _,r in tl.iterrows()},
        hover_data={"Álbum":True,"Canciones":True,"Fecha":True},
        size_max=35, height=220,
    )
    fig_tl.update_yaxes(visible=False)
    fig_tl.update_layout(showlegend=False, plot_bgcolor="white",
                          paper_bgcolor="white", margin=dict(l=0,r=0,t=10,b=30),
                          xaxis=dict(showgrid=False, title=""))
    st.plotly_chart(fig_tl, use_container_width=True)

    # Metacritic
    mc = ordered[ordered["metacritic_score"].notna()].copy()
    mc["metacritic_score"] = pd.to_numeric(mc["metacritic_score"], errors="coerce")
    mc = mc.dropna(subset=["metacritic_score"])
    if not mc.empty:
        st.markdown("### Puntuaciones Metacritic")
        st.caption("Metacritic agrega reseñas de críticos profesionales. 80+ = excelente recepción.")
        fig_mc = px.bar(
            mc.sort_values("album_release"), x="album_name", y="metacritic_score",
            color="album_name",
            color_discrete_map={r["album_name"]:ERA_COLORS.get(r["album_name"],"#888") for _,r in mc.iterrows()},
            labels={"album_name":"","metacritic_score":"Score Metacritic"},
            height=360,
        )
        fig_mc.update_layout(showlegend=False, plot_bgcolor="white",
                              paper_bgcolor="white", xaxis_tickangle=-40,
                              xaxis=dict(tickfont=dict(size=10)))
        fig_mc.add_hline(y=80, line_dash="dot", line_color="#888", line_width=1,
                          annotation_text="80 · excelente", annotation_font_size=10)
        st.plotly_chart(fig_mc, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 2 — AUDIO POR ÁLBUM
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Audio por álbum":
    st.markdown("# 📊 Audio Features por Álbum")
    st.markdown("""
    <div class="info-box">
    Los <b>audio features</b> son métricas que Spotify calcula automáticamente para cada canción.
    Nos permiten describir el sonido de manera objetiva y comparar eras entre sí.
    </div>
    """, unsafe_allow_html=True)

    tab1, tab2 = st.tabs(["🔍 Un álbum", "⚖️ Comparar álbumes"])

    avail = [f for f in AUDIO_FEATURES if f in songs_df.columns]

    with tab1:
        sel = st.selectbox("Álbum", DATA_ALBUMS)
        sub = songs_df[songs_df["album_name"] == sel].copy()
        color = ERA_COLORS.get(sel, "#888")
        emoji = ERA_EMOJI.get(sel, "🎵")

        c1, c2, c3 = st.columns(3)
        alb_info = albums_df[albums_df["album_name"] == sel]
        year = pd.to_datetime(alb_info.iloc[0]["album_release"]).year if not alb_info.empty else "?"
        score = alb_info.iloc[0].get("metacritic_score") if not alb_info.empty else None
        c1.metric("📅 Año", year)
        c2.metric("🎵 Canciones", len(sub))
        if pd.notna(score): c3.metric("⭐ Metacritic", int(score))

        st.markdown("#### Perfil de sonido")
        st.caption("El radar muestra el promedio de cada feature para todo el álbum. Úsalo para entender la 'personalidad' sonora del disco.")

        means = sub[avail].mean()
        fig_r = make_radar([(sel, color, means)], avail)
        st.plotly_chart(fig_r, use_container_width=True)

        # Explicaciones de features
        with st.expander("ℹ️ ¿Qué significa cada feature?"):
            for f in avail:
                st.markdown(f"**{FEATURE_LABELS[f]}**: {FEATURE_DESC.get(f,'')}")

        st.markdown("#### Canciones del álbum")
        feat_sel = st.selectbox("Ver feature", avail, format_func=lambda x: FEATURE_LABELS[x])
        plot_df = sub[["track_name", feat_sel]].dropna().sort_values(feat_sel, ascending=True)

        fig_b = px.bar(
            plot_df, x=feat_sel, y="track_name", orientation="h",
            color_discrete_sequence=[color],
            labels={feat_sel: FEATURE_LABELS[feat_sel], "track_name": ""},
            height=max(280, len(plot_df)*26),
        )
        fig_b.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             margin=dict(l=0,r=20,t=10,b=20))
        st.plotly_chart(fig_b, use_container_width=True)

    with tab2:
        st.markdown("#### Comparar dos álbumes")
        st.caption("Selecciona dos álbumes para ver sus diferencias en el radar y en cada feature.")
        c1, c2 = st.columns(2)
        alb_a = c1.selectbox("Álbum A", DATA_ALBUMS, index=0)
        alb_b = c2.selectbox("Álbum B", DATA_ALBUMS, index=min(4, len(DATA_ALBUMS)-1))

        sub_a = songs_df[songs_df["album_name"] == alb_a]
        sub_b = songs_df[songs_df["album_name"] == alb_b]
        means_a = sub_a[avail].mean()
        means_b = sub_b[avail].mean()
        col_a = ERA_COLORS.get(alb_a, "#888")
        col_b = ERA_COLORS.get(alb_b, "#888")

        fig_cmp = make_radar([(alb_a, col_a, means_a), (alb_b, col_b, means_b)], avail)
        st.plotly_chart(fig_cmp, use_container_width=True)

        # Tabla diferencias
        diff_rows = []
        for f in avail:
            va, vb = round(means_a.get(f,0),3), round(means_b.get(f,0),3)
            diff_rows.append({"Feature": FEATURE_LABELS[f], alb_a: va, alb_b: vb,
                               "Diferencia": round(vb-va, 3)})
        diff_df = pd.DataFrame(diff_rows)
        st.dataframe(diff_df, hide_index=True, use_container_width=True)

        # Todos los álbumes — heatmap promedio
        st.markdown("#### Comparación global de todos los álbumes")
        st.caption("Cada fila es un álbum. Colores más intensos = valor más alto en esa feature.")
        heat = songs_df.groupby("album_name")[avail].mean().reindex(DATA_ALBUMS).dropna()
        heat.columns = [FEATURE_LABELS[c] for c in heat.columns]
        fig_h = px.imshow(
            heat.T, color_continuous_scale="RdPu",
            labels=dict(x="Álbum", y="Feature", color="Valor"),
            aspect="auto", height=260,
        )
        fig_h.update_layout(paper_bgcolor="white", margin=dict(l=80,r=20,t=10,b=80),
                             xaxis_tickangle=-40, xaxis=dict(tickfont=dict(size=9)))
        st.plotly_chart(fig_h, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3 — ANÁLISIS DE LYRICS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📝 Análisis de lyrics":
    st.markdown("# 📝 Análisis de Lyrics")
    lyrics_df = songs_df[songs_df["has_lyrics"]].copy()
    st.caption(f"✅ {len(lyrics_df)} canciones con lyrics · {len(songs_df)-len(lyrics_df)} sin lyrics")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "☁️ WordCloud", "😊 Sentimiento", "🧠 Complejidad", "🔍 Buscador", "📈 Evolución"
    ])

    # ── WordCloud ─────────────────────────────────────────────────────────────
    with tab1:
        st.markdown("#### Nube de palabras")
        st.markdown("""
        <div class="info-box">
        Las palabras más grandes son las que más se repiten (sin contar palabras vacías como "the", "and", etc.).
        Selecciona un álbum o ve el total de la discografía.
        </div>
        """, unsafe_allow_html=True)

        wc_album = st.selectbox("Álbum", ["🎵 Toda la discografía"] + DATA_ALBUMS, key="wc")
        if wc_album == "🎵 Toda la discografía":
            wc_sub = lyrics_df
            wc_color = "#C084A0"
        else:
            wc_sub = lyrics_df[lyrics_df["album_name"] == wc_album]
            wc_color = ERA_COLORS.get(wc_album, "#888")

        st.caption(f"Basado en {len(wc_sub)} canciones")
        wc_text = clean_for_wc(" ".join(wc_sub["lyrics"].dropna().astype(str)))
        if wc_text.strip():
            wc = WordCloud(width=900, height=380, background_color="white",
                            colormap="RdPu", max_words=100, collocations=False).generate(wc_text)
            fig_wc, ax = plt.subplots(figsize=(11, 4))
            ax.imshow(wc, interpolation="bilinear"); ax.axis("off")
            fig_wc.patch.set_facecolor("white")
            st.pyplot(fig_wc); plt.close()
        else:
            st.info("No hay suficientes lyrics para este álbum.")

    # ── Sentimiento ───────────────────────────────────────────────────────────
    with tab2:
        st.markdown("#### Análisis de sentimiento")
        st.markdown("""
        <div class="info-box">
        El <b>sentimiento VADER</b> es un algoritmo diseñado para texto en inglés que detecta positividad/negatividad.
        Va de <b>-1</b> (muy negativo) a <b>+1</b> (muy positivo). Un valor de 0 es neutro.
        <br>⚠️ Es una aproximación — el sarcasmo o la ironía pueden confundir al algoritmo.
        </div>
        """, unsafe_allow_html=True)

        lyrics_df["sentiment"] = lyrics_df["lyrics"].apply(get_sentiment)
        sent_df = lyrics_df.dropna(subset=["sentiment"])

        view = st.radio("Ver por", ["Álbum (promedio)", "Canción (individual)"], horizontal=True)

        if view == "Álbum (promedio)":
            sent_alb = sent_df.groupby("album_name")["sentiment"].mean().reset_index()
            sent_alb = sent_alb[sent_alb["album_name"].isin(DATA_ALBUMS)]
            sent_alb["color"] = sent_alb["album_name"].map(ERA_COLORS).fillna("#888")
            sent_alb = sent_alb.sort_values("sentiment")
            fig_s = px.bar(
                sent_alb, x="sentiment", y="album_name", orientation="h",
                color="album_name",
                color_discrete_map={r["album_name"]:r["color"] for _,r in sent_alb.iterrows()},
                labels={"sentiment":"Sentimiento promedio","album_name":""},
                height=460,
            )
            fig_s.add_vline(x=0, line_dash="dot", line_color="#ccc", line_width=1)
            fig_s.update_layout(showlegend=False, plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_s, use_container_width=True)
            st.caption("Los álbumes a la derecha del 0 tienen lyrics más positivas en promedio. A la izquierda, más oscuras.")
        else:
            alb_sel = st.selectbox("Álbum", DATA_ALBUMS, key="sent_alb")
            sub_s = sent_df[sent_df["album_name"] == alb_sel].sort_values("sentiment")
            color = ERA_COLORS.get(alb_sel, "#888")
            fig_s2 = px.bar(
                sub_s, x="sentiment", y="track_name", orientation="h",
                color_discrete_sequence=[color],
                labels={"sentiment":"Sentimiento","track_name":""},
                height=max(300, len(sub_s)*26),
            )
            fig_s2.add_vline(x=0, line_dash="dot", line_color="#ccc", line_width=1)
            fig_s2.update_layout(plot_bgcolor="white", paper_bgcolor="white")
            st.plotly_chart(fig_s2, use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**😊 Top 5 más positivas**")
            st.dataframe(sent_df.nlargest(5,"sentiment")[["track_name","album_name","sentiment"]].round(3), hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**😢 Top 5 más oscuras**")
            st.dataframe(sent_df.nsmallest(5,"sentiment")[["track_name","album_name","sentiment"]].round(3), hide_index=True, use_container_width=True)

    # ── Complejidad léxica ────────────────────────────────────────────────────
    with tab3:
        st.markdown("#### Complejidad léxica")
        st.markdown("""
        <div class="info-box">
        La <b>diversidad léxica</b> mide qué tan variado es el vocabulario de una canción.
        Se calcula como: <i>palabras únicas ÷ total de palabras</i>.
        <br>• <b>1.0</b> = todas las palabras son diferentes (muy variado)
        <br>• <b>0.3</b> = mucha repetición de las mismas palabras
        <br>Canciones con coros muy repetitivos tienden a tener valores bajos.
        </div>
        """, unsafe_allow_html=True)

        lyrics_df["lex"] = lyrics_df["lyrics"].apply(lexical_diversity)
        lex_df = lyrics_df.dropna(subset=["lex"])

        view_l = st.radio("Ver por", ["Álbum (promedio)", "Canción (individual)"], horizontal=True, key="lex_view")

        if view_l == "Álbum (promedio)":
            lex_alb = lex_df.groupby("album_name")["lex"].mean().reset_index()
            lex_alb = lex_alb[lex_alb["album_name"].isin(DATA_ALBUMS)]
            lex_alb["color"] = lex_alb["album_name"].map(ERA_COLORS).fillna("#888")
            fig_l = px.bar(
                lex_alb.sort_values("lex", ascending=False),
                x="album_name", y="lex", color="album_name",
                color_discrete_map={r["album_name"]:r["color"] for _,r in lex_alb.iterrows()},
                labels={"lex":"Diversidad léxica","album_name":""},
                height=380,
            )
            fig_l.update_layout(showlegend=False, plot_bgcolor="white",
                                 paper_bgcolor="white", xaxis_tickangle=-40,
                                 xaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig_l, use_container_width=True)
        else:
            alb_l = st.selectbox("Álbum", DATA_ALBUMS, key="lex_alb")
            sub_l = lex_df[lex_df["album_name"] == alb_l].sort_values("lex", ascending=False)
            color = ERA_COLORS.get(alb_l, "#888")
            fig_l2 = px.bar(
                sub_l, x="track_name", y="lex",
                color_discrete_sequence=[color],
                labels={"lex":"Diversidad léxica","track_name":""},
                height=340,
            )
            fig_l2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                   xaxis_tickangle=-40, xaxis=dict(tickfont=dict(size=9)))
            st.plotly_chart(fig_l2, use_container_width=True)

        # Top / bottom
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**🧠 Vocabulario más rico**")
            st.dataframe(lex_df.nlargest(5,"lex")[["track_name","album_name","lex"]], hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**🔁 Más repetitivas**")
            st.dataframe(lex_df.nsmallest(5,"lex")[["track_name","album_name","lex"]], hide_index=True, use_container_width=True)

    # ── Buscador ──────────────────────────────────────────────────────────────
    with tab4:
        st.markdown("#### Buscador de frases y palabras")
        st.markdown("""
        <div class="info-box">
        Busca cualquier palabra o frase en todas las lyrics. Puedes filtrar por álbum para ver en qué canciones aparece.
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([3,1])
        query = c1.text_input("🔍 Palabra o frase", placeholder='Ej: love, "shake it off", rain...')
        alb_filter = c2.selectbox("Filtrar álbum", ["Todos"] + DATA_ALBUMS, key="search_alb")

        if query:
            search_df = lyrics_df if alb_filter == "Todos" else lyrics_df[lyrics_df["album_name"] == alb_filter]
            pattern = re.compile(re.escape(query.strip('"')), re.IGNORECASE)
            results = []
            for _, row in search_df.iterrows():
                if isinstance(row["lyrics"], str):
                    matches = pattern.findall(row["lyrics"])
                    if matches:
                        lines = [l.strip() for l in row["lyrics"].split("\n") if pattern.search(l)]
                        results.append({
                            "Canción": row["track_name"],
                            "Álbum": row["album_name"],
                            "Menciones": len(matches),
                            "Contexto": " / ".join(lines[:2]),
                        })
            if results:
                res_df = pd.DataFrame(results).sort_values("Menciones", ascending=False)
                total_mentions = res_df["Menciones"].sum()
                st.success(f"**'{query}'** aparece en **{len(results)} canciones** con **{total_mentions} menciones** en total")
                st.dataframe(res_df, hide_index=True, use_container_width=True)
            else:
                st.warning(f"No se encontró **'{query}'** en las lyrics.")

    # ── Evolución emocional ───────────────────────────────────────────────────
    with tab5:
        st.markdown("#### Evolución emocional a lo largo de los años")
        st.markdown("""
        <div class="info-box">
        Este gráfico muestra el sentimiento promedio de las lyrics de cada álbum, ordenado cronológicamente.
        <br>• <b>Línea arriba del 0</b> = letras más positivas/alegres
        <br>• <b>Línea abajo del 0</b> = letras más oscuras/melancólicas
        <br>Puedes ver cómo el estado emocional de Taylor evolucionó con el tiempo.
        </div>
        """, unsafe_allow_html=True)

        evo = lyrics_df.copy()
        evo["sentiment"] = evo["lyrics"].apply(get_sentiment)
        evo = evo.dropna(subset=["sentiment","album_release"])
        evo["year"] = evo["album_release"].dt.year

        evo_alb = evo.groupby(["album_name","year"])["sentiment"].mean().reset_index()
        evo_alb = evo_alb[evo_alb["album_name"].isin(DATA_ALBUMS)].sort_values("year")
        evo_alb["color"] = evo_alb["album_name"].map(ERA_COLORS).fillna("#888")

        fig_evo = go.Figure()
        fig_evo.add_hline(y=0, line_dash="dot", line_color="#ccc", line_width=1,
                            annotation_text="neutro", annotation_font_size=10)
        for _, row in evo_alb.iterrows():
            fig_evo.add_trace(go.Scatter(
                x=[row["year"]], y=[row["sentiment"]],
                mode="markers+text",
                name=row["album_name"],
                text=[ERA_EMOJI.get(row["album_name"],"🎵")],
                textposition="top center",
                marker=dict(color=row["color"], size=16),
                hovertemplate=f"<b>{row['album_name']}</b><br>Año: {row['year']}<br>Sentimiento: {row['sentiment']:.3f}<extra></extra>",
            ))
        fig_evo.update_layout(
            showlegend=False, height=380,
            plot_bgcolor="white", paper_bgcolor="white",
            xaxis=dict(title="Año", showgrid=False),
            yaxis=dict(title="Sentimiento promedio", gridcolor="#f0f0f0"),
            margin=dict(l=40,r=20,t=20,b=40),
        )
        st.plotly_chart(fig_evo, use_container_width=True)
        st.caption("Cada emoji/punto es un álbum. Hover para ver el nombre y valor exacto.")

# ══════════════════════════════════════════════════════════════════════════════
# 4 — OG vs TAYLOR'S VERSION
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔄 OG vs Taylor's Version":
    st.markdown("# 🔄 OG vs Taylor's Version")
    st.markdown("""
    <div class="info-box">
    Desde 2021, Taylor ha estado regrabando sus primeros álbumes para recuperar los derechos de su música.
    Aquí comparamos el sonido original vs. la versión regrabada, y exploramos las canciones exclusivas del vault (canciones inéditas que no aparecían en el original).
    </div>
    """, unsafe_allow_html=True)

    tv_sel = st.selectbox("Álbum", list(TV_PAIRS.keys()))
    og_name = tv_sel
    tv_name = TV_PAIRS[tv_sel]
    og_col  = ERA_COLORS.get(og_name, "#888")
    tv_col  = ERA_COLORS.get(tv_name, "#C49A2A")

    og_songs = songs_df[songs_df["album_name"] == og_name]
    tv_songs = songs_df[songs_df["album_name"] == tv_name]
    avail    = [f for f in AUDIO_FEATURES if f in songs_df.columns]

    c1, c2 = st.columns(2)
    c1.metric(f"{ERA_EMOJI.get(og_name,'🎵')} {og_name}", f"{len(og_songs)} canciones")
    c2.metric(f"{ERA_EMOJI.get(tv_name,'✨')} {tv_name}", f"{len(tv_songs)} canciones")

    st.markdown("#### Perfil sonoro comparado")
    st.caption("El radar muestra el promedio de cada feature. Diferencias pequeñas son normales al regrabar.")

    means_og = og_songs[avail].mean()
    means_tv = tv_songs[avail].mean()
    fig_tv = make_radar([(og_name, og_col, means_og), (tv_name, tv_col, means_tv)], avail)
    st.plotly_chart(fig_tv, use_container_width=True)

    # Tabla de diferencias
    st.markdown("#### Diferencia por feature")
    diff_rows = []
    for f in avail:
        va, vb = round(means_og.get(f,0),3), round(means_tv.get(f,0),3)
        delta = round(vb - va, 3)
        diff_rows.append({"Feature": FEATURE_LABELS[f], "Original": va,
                            "Taylor's Version": vb, "Δ": delta})
    st.dataframe(pd.DataFrame(diff_rows), hide_index=True, use_container_width=True)
    st.caption("Δ positivo = la TV tiene más de esa feature. Δ negativo = el original tenía más.")

    # From The Vault
    def norm_title(t):
        t = re.sub(r"\s*[\(\[].*?[\)\]]","",str(t))
        return re.sub(r"[^a-z0-9\s]","",t.lower()).strip()

    og_norms = set(og_songs["track_name"].apply(norm_title))
    vault = tv_songs[~tv_songs["track_name"].apply(norm_title).isin(og_norms)]

    if not vault.empty:
        st.markdown(f"#### 🗄️ From The Vault — canciones nuevas en {tv_name}")
        st.caption("Estas canciones fueron escritas originalmente pero nunca publicadas en el álbum original.")
        vault_show = vault[["track_number","track_name"]].rename(columns={"track_number":"#","track_name":"Canción"})
        st.dataframe(vault_show, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 5 — PODCAST
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🎙️ Podcast Swifting":

    st.markdown('<div style="background:#111;border-radius:16px;padding:40px;text-align:center;margin-bottom:12px">', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.65rem;letter-spacing:4px;color:#E8521A;font-weight:700;margin:0">✦ &nbsp; PODCAST &nbsp; ✦</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-family:Georgia,serif;font-style:italic;font-size:3rem;color:white;margin:8px 0 4px">Swifting</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#888;font-style:italic;margin:0">Que Taylor (no) se entere de esto. 🧡</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div style="display:flex;gap:10px;margin:12px 0">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    col1.markdown('<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:1.5rem;font-weight:700;color:#E8521A">8</div><div style="font-size:0.68rem;color:#888;letter-spacing:1px;text-transform:uppercase">Temporadas</div></div>', unsafe_allow_html=True)
    col2.markdown('<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:1.5rem;font-weight:700;color:#E8521A">557</div><div style="font-size:0.68rem;color:#888;letter-spacing:1px;text-transform:uppercase">Posts</div></div>', unsafe_allow_html=True)
    col3.markdown('<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:1.5rem;font-weight:700;color:#E8521A">8.4K</div><div style="font-size:0.68rem;color:#888;letter-spacing:1px;text-transform:uppercase">Seguidores</div></div>', unsafe_allow_html=True)
    col4.markdown('<div style="background:#1a1a1a;border-radius:10px;padding:14px;text-align:center"><div style="font-size:1.5rem;font-weight:700;color:#E8521A">31M+</div><div style="font-size:0.68rem;color:#888;letter-spacing:1px;text-transform:uppercase">Threads</div></div>', unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.65rem;letter-spacing:3px;text-transform:uppercase;color:#E8521A;font-weight:600;margin:20px 0 8px">— Escúchanos —</p>', unsafe_allow_html=True)

    st.markdown(
        '<iframe src="https://open.spotify.com/embed/show/629KoqnqWiz79IOi5zjW8i" '
        'width="100%" height="232" frameborder="0" allowtransparency="true" '
        'allow="encrypted-media" style="border-radius:12px"></iframe>',
        unsafe_allow_html=True
    )

    st.markdown('<p style="font-size:0.65rem;letter-spacing:3px;text-transform:uppercase;color:#E8521A;font-weight:600;margin:20px 0 10px">— Síguenos —</p>', unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown('<a href="https://open.spotify.com/show/629KoqnqWiz79IOi5zjW8i" target="_blank" style="display:block;background:#1DB954;color:white;text-align:center;padding:12px;border-radius:10px;text-decoration:none;font-weight:600;font-size:0.85rem">🎧 Spotify</a>', unsafe_allow_html=True)
    c2.markdown('<a href="https://www.youtube.com/@swiftingpodcast" target="_blank" style="display:block;background:#FF0000;color:white;text-align:center;padding:12px;border-radius:10px;text-decoration:none;font-weight:600;font-size:0.85rem">▶️ YouTube</a>', unsafe_allow_html=True)
    c3.markdown('<a href="https://www.instagram.com/swiftingpodcast/" target="_blank" style="display:block;background:linear-gradient(45deg,#f09433,#dc2743,#bc1888);color:white;text-align:center;padding:12px;border-radius:10px;text-decoration:none;font-weight:600;font-size:0.85rem">📸 Instagram</a>', unsafe_allow_html=True)

    st.markdown('<div style="background:#1a1a1a;border-left:3px solid #E8521A;border-radius:0 10px 10px 0;padding:16px 18px;margin-top:20px"><p style="color:#E8521A;font-size:0.65rem;letter-spacing:2px;text-transform:uppercase;margin:0 0 5px;font-weight:600">Amamos</p><p style="color:#ccc;font-size:0.9rem;margin:0;font-style:italic">"el chismecito, hablar sin filtro y analizar intensamente cada una de sus canciones."</p></div>', unsafe_allow_html=True)

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re, math
from collections import Counter

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)

# ── Config ────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Taylor Swift Dashboard",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Era colors & covers ───────────────────────────────────────────────────────
ERA_COLORS = {
    "Taylor Swift":                      "#6B8E6B",
    "The Taylor Swift Holiday Collection":"#C84B4B",
    "Beautiful Eyes":                    "#8E6B8E",
    "Fearless":                          "#D4A843",
    "Speak Now":                         "#7B4B8E",
    "Red":                               "#B22222",
    "1989":                              "#A8C4D4",
    "reputation":                        "#2C2C2C",
    "Lover":                             "#E8A0B4",
    "folklore":                          "#8C8C7A",
    "evermore":                          "#8B6914",
    "Fearless (Taylor's Version)":       "#C49A2A",
    "Red (Taylor's Version)":            "#8B1A1A",
    "Midnights":                         "#1C2951",
    "Speak Now (Taylor's Version)":      "#5B2D7A",
    "1989 (Taylor's Version)":           "#7AA8C4",
    "THE TORTURED POETS DEPARTMENT":     "#C8C0B0",
    "The Life of a Showgirl":            "#C8A882",
}

ALBUM_COVERS = {
    "Taylor Swift":                      "https://upload.wikimedia.org/wikipedia/en/thumb/6/60/Taylor_Swift_-_Taylor_Swift.png/220px-Taylor_Swift_-_Taylor_Swift.png",
    "The Taylor Swift Holiday Collection":"https://upload.wikimedia.org/wikipedia/en/thumb/9/9a/Taylor_Swift_-_The_Taylor_Swift_Holiday_Collection.png/220px-Taylor_Swift_-_The_Taylor_Swift_Holiday_Collection.png",
    "Beautiful Eyes":                    "https://upload.wikimedia.org/wikipedia/en/thumb/6/6c/Taylor_Swift_-_Beautiful_Eyes.png/220px-Taylor_Swift_-_Beautiful_Eyes.png",
    "Fearless":                          "https://upload.wikimedia.org/wikipedia/en/thumb/3/3f/Taylor_Swift_-_Fearless.png/220px-Taylor_Swift_-_Fearless.png",
    "Speak Now":                         "https://upload.wikimedia.org/wikipedia/en/thumb/9/9a/Taylor_Swift_-_Speak_Now.png/220px-Taylor_Swift_-_Speak_Now.png",
    "Red":                               "https://upload.wikimedia.org/wikipedia/en/thumb/e/e8/Taylor_Swift_-_Red.png/220px-Taylor_Swift_-_Red.png",
    "1989":                              "https://upload.wikimedia.org/wikipedia/en/thumb/f/f6/Taylor_Swift_-_1989.png/220px-Taylor_Swift_-_1989.png",
    "reputation":                        "https://upload.wikimedia.org/wikipedia/en/thumb/3/3c/Taylor_Swift_-_Reputation_Album.png/220px-Taylor_Swift_-_Reputation_Album.png",
    "Lover":                             "https://upload.wikimedia.org/wikipedia/en/thumb/3/3e/Taylor_Swift_-_Lover.png/220px-Taylor_Swift_-_Lover.png",
    "folklore":                          "https://upload.wikimedia.org/wikipedia/en/thumb/f/f8/Taylor_Swift_-_Folklore.png/220px-Taylor_Swift_-_Folklore.png",
    "evermore":                          "https://upload.wikimedia.org/wikipedia/en/thumb/0/0f/Taylor_Swift_-_Evermore.png/220px-Taylor_Swift_-_Evermore.png",
    "Fearless (Taylor's Version)":       "https://upload.wikimedia.org/wikipedia/en/thumb/4/4e/Taylor_Swift_-_Fearless_%28Taylor%27s_Version%29.png/220px-Taylor_Swift_-_Fearless_%28Taylor%27s_Version%29.png",
    "Red (Taylor's Version)":            "https://upload.wikimedia.org/wikipedia/en/thumb/0/0e/Taylor_Swift_-_Red_%28Taylor%27s_Version%29.png/220px-Taylor_Swift_-_Red_%28Taylor%27s_Version%29.png",
    "Midnights":                         "https://upload.wikimedia.org/wikipedia/en/thumb/3/32/Taylor_Swift_-_Midnights.png/220px-Taylor_Swift_-_Midnights.png",
    "Speak Now (Taylor's Version)":      "https://upload.wikimedia.org/wikipedia/en/thumb/5/5a/Taylor_Swift_-_Speak_Now_%28Taylor%27s_Version%29.png/220px-Taylor_Swift_-_Speak_Now_%28Taylor%27s_Version%29.png",
    "1989 (Taylor's Version)":           "https://upload.wikimedia.org/wikipedia/en/thumb/d/d8/Taylor_Swift_-_1989_%28Taylor%27s_Version%29.png/220px-Taylor_Swift_-_1989_%28Taylor%27s_Version%29.png",
    "THE TORTURED POETS DEPARTMENT":     "https://upload.wikimedia.org/wikipedia/en/thumb/d/df/Taylor_Swift_-_The_Tortured_Poets_Department.png/220px-Taylor_Swift_-_The_Tortured_Poets_Department.png",
    "The Life of a Showgirl":            "https://upload.wikimedia.org/wikipedia/en/thumb/3/3b/Taylor_Swift_-_The_Life_of_a_Showgirl.png/220px-Taylor_Swift_-_The_Life_of_a_Showgirl.png",
}

TV_PAIRS = {
    "Fearless": "Fearless (Taylor's Version)",
    "Red":      "Red (Taylor's Version)",
    "Speak Now":"Speak Now (Taylor's Version)",
    "1989":     "1989 (Taylor's Version)",
}

PODCAST = {
    "name":     "Swifting",
    "youtube":  "",
    "spotify":  "https://open.spotify.com/show/swifting",
}

AUDIO_FEATURES = ["danceability","energy","acousticness","valence","loudness","tempo"]
FEATURE_LABELS = {
    "danceability": "Bailabilidad",
    "energy":       "Energía",
    "acousticness": "Acústica",
    "valence":      "Valencia (positividad)",
    "loudness":     "Volumen",
    "tempo":        "Tempo (BPM)",
}

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    songs   = pd.read_csv("Taylor_Songs_Clean.csv")
    albums  = pd.read_csv("Taylor_Albums.csv")
    songs["album_release"] = pd.to_datetime(songs["album_release"], errors="coerce")
    albums["album_release"] = pd.to_datetime(albums["album_release"], errors="coerce")
    songs["color"] = songs["album_name"].map(ERA_COLORS).fillna("#888888")
    songs["has_lyrics"] = songs["lyrics"].notna() & (songs["lyrics"].astype(str).str.strip() != "")
    return songs, albums

songs_df, albums_df = load_data()
ALBUM_ORDER = albums_df.sort_values("album_release")["album_name"].tolist()

# ── Helpers ───────────────────────────────────────────────────────────────────
analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return None
    return analyzer.polarity_scores(text)["compound"]

def lexical_complexity(text):
    if not isinstance(text, str) or not text.strip():
        return None
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    if len(tokens) < 10:
        return None
    return len(set(tokens)) / len(tokens)

def clean_for_wordcloud(text):
    stop = set(stopwords.words("english")) | {"oh","like","yeah","know","just","got","get","gonna","wanna","cause","let","say","ooh","ah","la","da"}
    tokens = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    return " ".join(t for t in tokens if t not in stop)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b1/Taylor_Swift_at_the_2023_MTV_Video_Music_Awards_3.png/220px-Taylor_Swift_at_the_2023_MTV_Video_Music_Awards_3.png", use_container_width=True)
    st.markdown("## 🎸 Taylor Swift Dashboard v2")
    st.markdown("*All eras · All songs · All the feels*")
    st.divider()
    section = st.radio("Navegar a", [
        "🌟 Línea del tiempo",
        "📊 Quick Look por álbum",
        "📝 Análisis de lyrics",
        "🔄 OG vs Taylor's Version",
        "🎙️ Podcast Swifting",
    ])
    st.divider()
    st.caption(f"📀 {len(songs_df)} canciones · {songs_df['album_name'].nunique()} álbumes")

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — LÍNEA DEL TIEMPO
# ══════════════════════════════════════════════════════════════════════════════
if section == "🌟 Línea del tiempo":
    st.title("🌟 La Era de Taylor Swift")
    st.markdown("Una discografía completa, de 2006 a 2025.")

    ordered = albums_df[albums_df["album_name"].isin(ALBUM_ORDER)].sort_values("album_release")

    cols_per_row = 6
    rows = [ordered.iloc[i:i+cols_per_row] for i in range(0, len(ordered), cols_per_row)]

    for row_df in rows:
        cols = st.columns(len(row_df))
        for col, (_, alb) in zip(cols, row_df.iterrows()):
            name  = alb["album_name"]
            year  = pd.to_datetime(alb["album_release"]).year if pd.notna(alb["album_release"]) else "?"
            color = ERA_COLORS.get(name, "#888")
            cover = ALBUM_COVERS.get(name, "")
            with col:
                if cover:
                    st.image(cover, use_container_width=True)
                st.markdown(
                    f"<div style='background:{color}22;border-left:3px solid {color};"
                    f"padding:6px 8px;border-radius:6px;font-size:0.72rem;font-weight:600;"
                    f"color:{color};margin-top:4px'>{name}<br>"
                    f"<span style='font-weight:400;color:#888'>{year}</span></div>",
                    unsafe_allow_html=True
                )
                score = alb.get("metacritic_score")
                if pd.notna(score):
                    st.caption(f"⭐ Metacritic: {int(score)}")

    st.divider()

    # Timeline Plotly
    st.subheader("Cronología de lanzamientos")
    timeline_data = []
    for _, alb in ordered.iterrows():
        n_songs = len(songs_df[songs_df["album_name"] == alb["album_name"]])
        timeline_data.append({
            "Álbum": alb["album_name"],
            "Fecha": alb["album_release"],
            "Canciones": n_songs,
            "Color": ERA_COLORS.get(alb["album_name"], "#888"),
        })
    tl = pd.DataFrame(timeline_data).dropna(subset=["Fecha"])

    fig = px.scatter(
        tl, x="Fecha", y=[0]*len(tl), size="Canciones",
        color="Álbum", color_discrete_map={r["Álbum"]: r["Color"] for _, r in tl.iterrows()},
        hover_data={"Álbum":True,"Canciones":True,"Fecha":True},
        size_max=40, height=280,
    )
    fig.update_yaxes(visible=False)
    fig.update_layout(
        showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", margin=dict(l=0,r=0,t=20,b=20),
        xaxis=dict(showgrid=False),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metacritic bar
    mc = ordered[ordered["metacritic_score"].notna()].copy()
    mc["metacritic_score"] = pd.to_numeric(mc["metacritic_score"], errors="coerce")
    mc = mc.dropna(subset=["metacritic_score"])
    if not mc.empty:
        st.subheader("Puntuaciones Metacritic")
        fig2 = px.bar(
            mc, x="album_name", y="metacritic_score",
            color="album_name",
            color_discrete_map={r["album_name"]: ERA_COLORS.get(r["album_name"],"#888") for _,r in mc.iterrows()},
            labels={"album_name":"Álbum","metacritic_score":"Score"},
            height=380,
        )
        fig2.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                           paper_bgcolor="rgba(0,0,0,0)", xaxis_tickangle=-45)
        fig2.add_hline(y=80, line_dash="dot", line_color="gold", annotation_text="80 — excelente")
        st.plotly_chart(fig2, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — QUICK LOOK POR ÁLBUM
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📊 Quick Look por álbum":
    st.title("📊 Quick Look por álbum")

    selected_album = st.selectbox(
        "Selecciona un álbum",
        options=ALBUM_ORDER,
        format_func=lambda x: x
    )

    alb_songs = songs_df[songs_df["album_name"] == selected_album].copy()
    color     = ERA_COLORS.get(selected_album, "#888")
    cover     = ALBUM_COVERS.get(selected_album, "")

    c1, c2 = st.columns([1, 3])
    with c1:
        if cover:
            st.image(cover, width=180)
        alb_info = albums_df[albums_df["album_name"] == selected_album]
        if not alb_info.empty:
            row = alb_info.iloc[0]
            st.metric("Año", pd.to_datetime(row["album_release"]).year if pd.notna(row["album_release"]) else "?")
            if pd.notna(row.get("metacritic_score")):
                st.metric("Metacritic", int(row["metacritic_score"]))
        st.metric("Canciones", len(alb_songs))

    with c2:
        # Audio features radar
        avail = [f for f in ["danceability","energy","acousticness","valence"] if f in alb_songs.columns]
        means = alb_songs[avail].mean()

        fig_radar = go.Figure(go.Scatterpolar(
            r=means.values.tolist() + [means.values[0]],
            theta=[FEATURE_LABELS[f] for f in avail] + [FEATURE_LABELS[avail[0]]],
            fill="toself",
            fillcolor=color + "44",
            line=dict(color=color, width=2),
        ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0,1])),
            showlegend=False, height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=40,r=40,t=40,b=40),
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # Features por canción
    st.subheader("Audio features por canción")
    feat_sel = st.selectbox("Feature", avail, format_func=lambda x: FEATURE_LABELS[x])
    plot_df  = alb_songs[["track_name", feat_sel]].dropna().sort_values(feat_sel, ascending=True)

    fig_bar = px.bar(
        plot_df, x=feat_sel, y="track_name", orientation="h",
        color_discrete_sequence=[color], height=max(300, len(plot_df)*28),
        labels={feat_sel: FEATURE_LABELS[feat_sel], "track_name": ""},
    )
    fig_bar.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Comparar álbumes
    st.divider()
    st.subheader("Comparar álbumes — audio features promedio")
    feat_cmp = st.multiselect("Features a comparar", avail,
                               default=avail[:3],
                               format_func=lambda x: FEATURE_LABELS[x])
    if feat_cmp:
        cmp_df = songs_df.groupby("album_name")[feat_cmp].mean().reset_index()
        cmp_df = cmp_df[cmp_df["album_name"].isin(ALBUM_ORDER)]
        cmp_melt = cmp_df.melt(id_vars="album_name", var_name="Feature", value_name="Valor")
        cmp_melt["Feature"] = cmp_melt["Feature"].map(FEATURE_LABELS)

        fig_cmp = px.bar(
            cmp_melt, x="album_name", y="Valor", color="Feature",
            barmode="group", height=420,
            labels={"album_name":"", "Valor":""},
        )
        fig_cmp.update_layout(xaxis_tickangle=-45,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cmp, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 3 — ANÁLISIS DE LYRICS
# ══════════════════════════════════════════════════════════════════════════════
elif section == "📝 Análisis de lyrics":
    st.title("📝 Análisis de lyrics")

    lyrics_df = songs_df[songs_df["has_lyrics"]].copy()
    st.caption(f"{len(lyrics_df)} canciones con lyrics disponibles de {len(songs_df)} totales")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "☁️ WordCloud", "😊 Sentimiento", "🧠 Complejidad léxica",
        "🔍 Buscador de frases", "📈 Evolución emocional"
    ])

    # ── Tab 1: WordCloud ──────────────────────────────────────────────────────
    with tab1:
        wc_album = st.selectbox("Álbum para WordCloud", ["Toda la discografía"] + ALBUM_ORDER, key="wc_alb")
        if wc_album == "Toda la discografía":
            wc_lyrics = " ".join(lyrics_df["lyrics"].dropna().astype(str))
            wc_color  = "#C084A0"
        else:
            sub       = lyrics_df[lyrics_df["album_name"] == wc_album]
            wc_lyrics = " ".join(sub["lyrics"].dropna().astype(str))
            wc_color  = ERA_COLORS.get(wc_album, "#888")

        if wc_lyrics.strip():
            wc_text = clean_for_wordcloud(wc_lyrics)
            wc = WordCloud(
                width=800, height=400, background_color="white",
                colormap="RdPu", max_words=120,
            ).generate(wc_text)
            fig_wc, ax = plt.subplots(figsize=(10, 4))
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig_wc)
            plt.close()
        else:
            st.info("No hay lyrics disponibles para este álbum.")

    # ── Tab 2: Sentimiento ────────────────────────────────────────────────────
    with tab2:
        lyrics_df["sentiment"] = lyrics_df["lyrics"].apply(get_sentiment)
        sent_df = lyrics_df.dropna(subset=["sentiment"])

        # Promedio por álbum
        sent_alb = sent_df.groupby("album_name")["sentiment"].mean().reset_index()
        sent_alb = sent_alb[sent_alb["album_name"].isin(ALBUM_ORDER)]
        sent_alb["color"] = sent_alb["album_name"].map(ERA_COLORS).fillna("#888")
        sent_alb = sent_alb.sort_values("sentiment")

        fig_sent = px.bar(
            sent_alb, x="sentiment", y="album_name", orientation="h",
            color="album_name",
            color_discrete_map={r["album_name"]: r["color"] for _, r in sent_alb.iterrows()},
            labels={"sentiment": "Sentimiento promedio (VADER)", "album_name": ""},
            height=480,
        )
        fig_sent.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_sent.update_layout(showlegend=False, plot_bgcolor="rgba(0,0,0,0)",
                                paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_sent, use_container_width=True)

        # Top 5 más positivas / negativas
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**😊 Top 5 más positivas**")
            top_pos = sent_df.nlargest(5, "sentiment")[["track_name","album_name","sentiment"]]
            top_pos["sentiment"] = top_pos["sentiment"].round(3)
            st.dataframe(top_pos, hide_index=True, use_container_width=True)
        with c2:
            st.markdown("**😢 Top 5 más negativas**")
            top_neg = sent_df.nsmallest(5, "sentiment")[["track_name","album_name","sentiment"]]
            top_neg["sentiment"] = top_neg["sentiment"].round(3)
            st.dataframe(top_neg, hide_index=True, use_container_width=True)

    # ── Tab 3: Complejidad léxica ─────────────────────────────────────────────
    with tab3:
        lyrics_df["lex"] = lyrics_df["lyrics"].apply(lexical_complexity)
        lex_df = lyrics_df.dropna(subset=["lex"])

        lex_alb = lex_df.groupby("album_name")["lex"].mean().reset_index()
        lex_alb = lex_alb[lex_alb["album_name"].isin(ALBUM_ORDER)]
        lex_alb["color"] = lex_alb["album_name"].map(ERA_COLORS).fillna("#888")

        fig_lex = px.bar(
            lex_alb.sort_values("lex", ascending=False),
            x="album_name", y="lex",
            color="album_name",
            color_discrete_map={r["album_name"]: r["color"] for _, r in lex_alb.iterrows()},
            labels={"lex": "Diversidad léxica (type-token ratio)", "album_name": ""},
            height=400,
        )
        fig_lex.update_layout(showlegend=False, xaxis_tickangle=-45,
                               plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_lex, use_container_width=True)
        st.caption("La diversidad léxica mide qué tan variado es el vocabulario: 1 = todas las palabras únicas, 0 = mucha repetición.")

    # ── Tab 4: Buscador de frases ─────────────────────────────────────────────
    with tab4:
        query = st.text_input("🔍 Busca una palabra o frase en todas las lyrics", placeholder='Ej: love, "shake it off", rain...')
        if query:
            pattern = re.compile(re.escape(query.strip('"')), re.IGNORECASE)
            results = []
            for _, row in lyrics_df.iterrows():
                if isinstance(row["lyrics"], str):
                    matches = pattern.findall(row["lyrics"])
                    if matches:
                        # Extraer contexto
                        lines = [l for l in row["lyrics"].split("\n") if pattern.search(l)]
                        results.append({
                            "Canción":    row["track_name"],
                            "Álbum":      row["album_name"],
                            "Menciones":  len(matches),
                            "Contexto":   " / ".join(lines[:2]),
                        })
            if results:
                res_df = pd.DataFrame(results).sort_values("Menciones", ascending=False)
                st.success(f"'{query}' aparece en **{len(results)}** canciones, **{res_df['Menciones'].sum()}** veces en total")
                st.dataframe(res_df, hide_index=True, use_container_width=True)
            else:
                st.warning(f"No se encontró '{query}' en ninguna canción.")

    # ── Tab 5: Evolución emocional ────────────────────────────────────────────
    with tab5:
        st.subheader("Evolución emocional a lo largo de los años")
        evo_df = lyrics_df.copy()
        evo_df["sentiment"] = evo_df["lyrics"].apply(get_sentiment)
        evo_df = evo_df.dropna(subset=["sentiment","album_release"])
        evo_df["year"] = pd.to_datetime(evo_df["album_release"]).dt.year

        evo_alb = evo_df.groupby(["album_name","year"])["sentiment"].mean().reset_index()
        evo_alb = evo_alb[evo_alb["album_name"].isin(ALBUM_ORDER)].sort_values("year")
        evo_alb["color"] = evo_alb["album_name"].map(ERA_COLORS).fillna("#888")

        fig_evo = px.line(
            evo_alb, x="year", y="sentiment", markers=True,
            color="album_name",
            color_discrete_map={r["album_name"]: r["color"] for _, r in evo_alb.iterrows()},
            labels={"year":"Año","sentiment":"Sentimiento promedio","album_name":"Álbum"},
            height=420,
        )
        fig_evo.add_hline(y=0, line_dash="dot", line_color="gray", annotation_text="neutro")
        fig_evo.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_evo, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 4 — OG vs TAYLOR'S VERSION
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🔄 OG vs Taylor's Version":
    st.title("🔄 OG vs Taylor's Version")
    st.markdown("Comparación de audio features entre los 4 álbumes regrabados y sus originales.")

    tv_sel = st.selectbox("Elige el álbum", list(TV_PAIRS.keys()))
    og_name = tv_sel
    tv_name = TV_PAIRS[tv_sel]

    og_songs = songs_df[songs_df["album_name"] == og_name].copy()
    tv_songs = songs_df[songs_df["album_name"] == tv_name].copy()

    og_col = ERA_COLORS.get(og_name, "#888")
    tv_col = ERA_COLORS.get(tv_name, "#C49A2A")

    features_avail = [f for f in ["danceability","energy","acousticness","valence"] if f in songs_df.columns]

    # Radar comparativo
    og_means = og_songs[features_avail].mean()
    tv_means = tv_songs[features_avail].mean()
    labels   = [FEATURE_LABELS[f] for f in features_avail] + [FEATURE_LABELS[features_avail[0]]]

    fig_tv = go.Figure()
    for means, name, color in [(og_means, og_name, og_col), (tv_means, tv_name, tv_col)]:
        vals = means.values.tolist() + [means.values[0]]
        fig_tv.add_trace(go.Scatterpolar(
            r=vals, theta=labels, fill="toself", name=name,
            fillcolor=color + "33", line=dict(color=color, width=2),
        ))
    fig_tv.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0,1])),
        showlegend=True, height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig_tv, use_container_width=True)

    # Barras comparativas por feature
    st.subheader("Diferencia por feature")
    diff_data = []
    for f in features_avail:
        diff_data.append({
            "Feature": FEATURE_LABELS[f],
            "OG":      og_means[f],
            "TV":      tv_means[f],
            "Δ":       tv_means[f] - og_means[f],
        })
    diff_df = pd.DataFrame(diff_data)
    diff_df["color"] = diff_df["Δ"].apply(lambda x: "#2ecc71" if x > 0 else "#e74c3c")

    fig_diff = go.Figure()
    fig_diff.add_trace(go.Bar(name=og_name, x=diff_df["Feature"], y=diff_df["OG"],
                               marker_color=og_col, opacity=0.85))
    fig_diff.add_trace(go.Bar(name=tv_name,  x=diff_df["Feature"], y=diff_df["TV"],
                               marker_color=tv_col, opacity=0.85))
    fig_diff.update_layout(barmode="group", height=340,
                            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_diff, use_container_width=True)

    # Canciones exclusivas (From The Vault)
    og_titles = set(og_songs["track_name"].str.lower().str.replace(r"\s*[\(\[].*", "", regex=True).str.strip())
    vault = tv_songs[~tv_songs["track_name"].str.lower().str.replace(r"\s*[\(\[].*","",regex=True).str.strip().isin(og_titles)]
    if not vault.empty:
        st.subheader(f"🗄️ From The Vault — canciones nuevas en {tv_name}")
        st.dataframe(vault[["track_number","track_name"]].rename(
            columns={"track_number":"#","track_name":"Canción"}
        ), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 5 — PODCAST SWIFTING
# ══════════════════════════════════════════════════════════════════════════════
elif section == "🎙️ Podcast Swifting":
    st.title("🎙️ Podcast Swifting")
    st.markdown("El podcast definitivo para Swifties.")

    if PODCAST["youtube"]:
        video_id = PODCAST["youtube"].split("v=")[-1].split("&")[0] if "v=" in PODCAST["youtube"] else PODCAST["youtube"].split("/")[-1]
        st.video(f"https://www.youtube.com/watch?v={video_id}")
    elif PODCAST["spotify"]:
        st.markdown(
            f'<iframe src="{PODCAST["spotify"].replace("open.spotify.com","open.spotify.com/embed").replace("/show/","/embed/show/")}"'
            ' width="100%" height="352" frameborder="0" allowtransparency="true" allow="encrypted-media"></iframe>',
            unsafe_allow_html=True
        )
    else:
        st.info("Configura la URL del podcast en la variable PODCAST al inicio del archivo.")

    st.divider()
    st.markdown("""
    **Swifting** es un podcast dedicado a analizar en profundidad la música, letras y eras de Taylor Swift.
    Perfectamente maridado con este dashboard. 🎵
    """)

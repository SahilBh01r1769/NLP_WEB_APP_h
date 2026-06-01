"""
NLP Multi-Feature Analyzer — Streamlit UI
HF Spaces deployment: calls nlp_core directly (no Flask needed).
Local dev: same file works, just run alongside app.py if you want the API too.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Analyzer",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]{background:#030712;color:#e2e8f0;}
[data-testid="stSidebar"]{background:#0d1117;border-right:1px solid rgba(255,255,255,0.07);}
[data-testid="stSidebar"] *{color:#e2e8f0 !important;}
h1,h2,h3{color:#e2e8f0 !important;}
.main-title{font-size:2.4rem;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#818cf8,#c084fc);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:0.2rem;}
.sub-title{color:#64748b;font-size:1rem;margin-bottom:2rem;}
.metric-card{background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:20px 24px;margin:8px 0;}
.result-card{background:rgba(255,255,255,0.03);border:1px solid rgba(99,102,241,0.25);border-radius:16px;padding:24px;margin:12px 0;}
.answer-box{background:rgba(99,102,241,0.08);border-left:3px solid #6366f1;border-radius:0 12px 12px 0;padding:16px 20px;font-size:1.1rem;font-weight:500;color:#e2e8f0;margin:12px 0;}
.keyword-pill{display:inline-block;background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);border-radius:100px;padding:5px 14px;margin:4px;font-size:0.85rem;color:#cbd5e1;}
.sentence-row{padding:10px 14px;border-radius:10px;margin:6px 0;border-left:3px solid;font-size:0.92rem;color:#e2e8f0;}
textarea,input{background:rgba(255,255,255,0.04) !important;border:1px solid rgba(255,255,255,0.1) !important;border-radius:10px !important;color:#e2e8f0 !important;}
.stTabs [data-baseweb="tab-list"]{background:rgba(255,255,255,0.03);border-radius:12px;padding:4px;gap:4px;}
.stTabs [data-baseweb="tab"]{background:transparent;color:#64748b;border-radius:8px;font-weight:500;}
.stTabs [aria-selected="true"]{background:rgba(99,102,241,0.2) !important;color:#a5b4fc !important;}
.stButton>button{background:#6366f1;color:white;border:none;border-radius:10px;font-weight:600;padding:10px 28px;transition:all 0.2s;width:100%;}
.stButton>button:hover{background:#4f46e5;transform:translateY(-1px);}
hr{border-color:rgba(255,255,255,0.07) !important;}
::-webkit-scrollbar{width:5px;}::-webkit-scrollbar-track{background:#030712;}::-webkit-scrollbar-thumb{background:#1e293b;border-radius:10px;}
.entities{background:transparent !important;font-size:15px;line-height:2;}
mark.entity{border-radius:6px !important;padding:3px 8px !important;font-size:14px;}
</style>
""", unsafe_allow_html=True)

# ── Load NLP core (cached so models only load once per session) ───────────────
@st.cache_resource(show_spinner="Loading NLP models — first run takes ~60s...")
def load_core():
    import nlp_core
    return nlp_core

core = load_core()

# ── Chart helpers ─────────────────────────────────────────────────────────────
def sentiment_color(label):
    return {"Positive": "#22c55e", "Negative": "#ef4444", "Neutral": "#94a3b8"}.get(label, "#94a3b8")

def make_gauge(value, label, color):
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        title={"text": label, "font": {"color": "#94a3b8", "size": 14}},
        number={"suffix": "%", "font": {"color": color, "size": 28}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#334155", "tickfont": {"color": "#475569"}},
            "bar": {"color": color},
            "bgcolor": "rgba(0,0,0,0)",
            "bordercolor": "rgba(255,255,255,0.05)",
            "steps": [
                {"range": [0, 33],   "color": "rgba(239,68,68,0.08)"},
                {"range": [33, 66],  "color": "rgba(148,163,184,0.08)"},
                {"range": [66, 100], "color": "rgba(34,197,94,0.08)"}
            ]
        }
    ))
    fig.update_layout(height=200, margin=dict(l=20,r=20,t=40,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#e2e8f0"})
    return fig

def make_entity_chart(breakdown):
    labels = [b["label"] for b in breakdown]
    counts = [b["count"] for b in breakdown]
    fig = go.Figure(go.Bar(
        x=labels, y=counts,
        marker_color=px.colors.qualitative.Pastel[:len(labels)],
        text=counts, textposition="outside", textfont={"color": "#e2e8f0"}
    ))
    fig.update_layout(height=280, margin=dict(l=0,r=0,t=20,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#94a3b8"},
                      xaxis={"gridcolor": "rgba(255,255,255,0.04)", "tickfont": {"color": "#94a3b8"}},
                      yaxis={"gridcolor": "rgba(255,255,255,0.04)", "tickfont": {"color": "#94a3b8"}})
    return fig

def make_keyword_chart(keywords):
    top    = keywords[:12]
    words  = [k["word"]  for k in reversed(top)]
    scores = [k["score"] for k in reversed(top)]
    fig = go.Figure(go.Bar(
        x=scores, y=words, orientation="h",
        marker=dict(color=scores, colorscale=[[0,"rgba(99,102,241,0.3)"],[1,"#a5b4fc"]], showscale=False),
        text=[f"{s:.4f}" for s in scores], textposition="outside",
        textfont={"color": "#94a3b8", "size": 11}
    ))
    fig.update_layout(height=max(300, len(top)*28), margin=dict(l=0,r=60,t=10,b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font={"color": "#94a3b8"},
                      xaxis={"gridcolor": "rgba(255,255,255,0.04)", "title": "TF-IDF Score"},
                      yaxis={"gridcolor": "rgba(0,0,0,0)"})
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 NLP Analyzer")
    st.markdown("---")
    st.markdown("**Features**")
    st.markdown("🎭 Sentiment Analysis\n\n📝 Text Summarization\n\n🏷️ Named Entity Recognition\n\n🔑 Keyword Extraction\n\n❓ Question Answering\n\n🌳 Dependency Parse")
    st.markdown("---")
    st.markdown("**Settings**")
    summary_sentences = st.slider("Summary length (sentences)", 1, 6, 3)
    keyword_count     = st.slider("Max keywords", 5, 20, 12)
    st.markdown("---")
    st.markdown("**Sample Texts**")

    samples = {
        "Tech Article":   "Apple Inc. announced its latest iPhone 16 in San Francisco on September 10, 2024. The new device features a 48MP camera system, an A18 chip built on TSMC's 3nm process, and a new Action button. CEO Tim Cook called it the most powerful iPhone ever created. The starting price is $799, and pre-orders begin September 13.",
        "Finance News":   "The Federal Reserve raised interest rates by 25 basis points on Wednesday, bringing the benchmark rate to its highest level in 22 years. Fed Chair Jerome Powell signaled that further hikes remain possible if inflation does not cool sufficiently. Stock markets fell sharply in response, with the S&P 500 dropping 1.4% and the Nasdaq declining 1.8%.",
        "Science Text":   "Researchers at MIT have developed a new artificial intelligence system capable of predicting protein folding structures with unprecedented accuracy. The system, trained on a dataset of 200 million protein sequences, outperforms existing methods by 35% on benchmark tests. The findings, published in Nature, could accelerate drug discovery significantly."
    }

    for label, content in samples.items():
        if st.button(f"📄 {label}", use_container_width=True):
            st.session_state["sample_text"] = content

    st.markdown("---")
    st.caption("spaCy · HuggingFace · Plotly · Streamlit")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">NLP Multi-Feature Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Sentiment · Summarization · NER · Keywords · QA · Dependency Parse</p>', unsafe_allow_html=True)

default_text = st.session_state.get("sample_text", "")
text_input   = st.text_area(
    "Enter your text",
    value=default_text,
    height=180,
    placeholder="Paste any article, paragraph, or document here...",
    label_visibility="collapsed"
)
word_count = len(text_input.split()) if text_input.strip() else 0
st.caption(f"📊 {word_count} words · {len(text_input)} characters")
st.markdown("---")

qa_question = st.text_input(
    "❓ Question for Q&A feature",
    placeholder="e.g. Who announced the iPhone 16?  |  What did the Fed do?"
)
st.markdown("---")

col_btn, col_clear = st.columns([3, 1])
with col_btn:
    analyze = st.button("🚀 Analyze Text", use_container_width=True)
with col_clear:
    if st.button("Clear", use_container_width=True):
        st.session_state["sample_text"] = ""
        st.rerun()

# ── Results ───────────────────────────────────────────────────────────────────
if analyze:
    if not text_input.strip() or word_count < 5:
        st.warning("Please enter at least 5 words.")
        st.stop()

    tabs = st.tabs(["🎭 Sentiment","📝 Summary","🏷️ NER","🔑 Keywords","❓ Q&A","🌳 Dependency"])

    # ── Sentiment ─────────────────────────────────────────────────────────────
    with tabs[0]:
        with st.spinner("Analysing sentiment..."):
            try:
                res   = core.analyse_sentiment(text_input)
                color = sentiment_color(res["label"])
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.plotly_chart(make_gauge(res["confidence"], "Confidence", color),
                                    use_container_width=True, config={"displayModeBar": False})
                with c2:
                    st.markdown(f'<div class="metric-card" style="text-align:center;"><div style="font-size:2.5rem;font-weight:800;color:{color};">{res["label"]}</div><div style="color:#64748b;font-size:0.85rem;margin-top:6px;">Overall Sentiment</div></div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card" style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#a5b4fc;">{res["polarity"]:+.3f}</div><div style="color:#64748b;font-size:0.85rem;margin-top:4px;">Polarity (−1 to +1)</div></div>', unsafe_allow_html=True)
                with c3:
                    st.markdown(f'<div class="metric-card" style="text-align:center;"><div style="font-size:1.8rem;font-weight:700;color:#c084fc;">{res["subjectivity"]:.2f}</div><div style="color:#64748b;font-size:0.85rem;margin-top:4px;">Subjectivity (0=obj · 1=subj)</div></div>', unsafe_allow_html=True)

                if res.get("sentences"):
                    st.markdown("#### Sentence-Level Breakdown")
                    for s in res["sentences"]:
                        sc = sentiment_color(s["label"])
                        st.markdown(f'<div class="sentence-row" style="border-color:{sc};background:rgba(0,0,0,0.2);"><span style="color:{sc};font-weight:600;font-size:0.8rem;">{s["label"]} ({s["polarity"]:+.3f})</span><br>{s["text"]}</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Sentiment analysis failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    with tabs[1]:
        if word_count < 30:
            st.warning("Provide at least 30 words for a meaningful summary.")
        else:
            with st.spinner("Generating summary — BART may take 20–30s on first run..."):
                try:
                    res = core.summarize_text(text_input, num_sentences=summary_sentences)
                    st.markdown(f'<div class="result-card"><div style="font-size:1rem;line-height:1.8;color:#e2e8f0;">{res["summary"]}</div></div>', unsafe_allow_html=True)
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Method",         res["method"])
                    m2.metric("Original Words", res["original_words"])
                    m3.metric("Summary Words",  res["summary_words"])
                    m4.metric("Compression",    f'{res["compression_rate"]}%')
                except Exception as e:
                    st.error(f"Summarization failed: {e}")

    # ── NER ───────────────────────────────────────────────────────────────────
    with tabs[2]:
        with st.spinner("Running NER..."):
            try:
                res = core.run_ner(text_input)
                if not res["entities"]:
                    st.info("No named entities found.")
                else:
                    st.markdown(f"**{res['total']} entities detected**")
                    st.markdown("#### Entity Highlights")
                    st.markdown(f'<div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:20px;font-size:15px;line-height:2.2;">{res["html"]}</div>', unsafe_allow_html=True)

                    col_chart, col_table = st.columns(2)
                    with col_chart:
                        st.markdown("#### Entity Type Breakdown")
                        st.plotly_chart(make_entity_chart(res["breakdown"]),
                                        use_container_width=True, config={"displayModeBar": False})
                    with col_table:
                        st.markdown("#### Entity List")
                        df = pd.DataFrame(res["entities"])[["text","label","desc"]]
                        df.columns = ["Entity","Type","Description"]
                        st.dataframe(df, use_container_width=True, height=280)
            except Exception as e:
                st.error(f"NER failed: {e}")

    # ── Keywords ──────────────────────────────────────────────────────────────
    with tabs[3]:
        with st.spinner("Extracting keywords..."):
            try:
                res = core.extract_keywords(text_input, top_n=keyword_count)
                if not res["keywords"]:
                    st.info("No keywords found. Try a longer text.")
                else:
                    st.markdown("#### Top Keywords")
                    pills = "".join(f'<span class="keyword-pill">{k["word"]} <span style="color:#6366f1;font-size:0.75rem;">×{k["count"]}</span></span>' for k in res["keywords"])
                    st.markdown(f'<div style="margin-bottom:20px;">{pills}</div>', unsafe_allow_html=True)
                    st.markdown("#### TF-IDF Scores")
                    st.plotly_chart(make_keyword_chart(res["keywords"]),
                                    use_container_width=True, config={"displayModeBar": False})
                    st.caption(f"Analysed {res['total_words']} tokens")
            except Exception as e:
                st.error(f"Keyword extraction failed: {e}")

    # ── Q&A ───────────────────────────────────────────────────────────────────
    with tabs[4]:
        if not qa_question.strip():
            st.info("Enter a question in the **Q&A question** field above, then click Analyze.")
        elif word_count < 10:
            st.warning("Provide more context text for Q&A.")
        else:
            with st.spinner("Finding answer (RoBERTa) — first run ~30s..."):
                try:
                    res = core.answer_question(text_input, qa_question)
                    conf_color = "#22c55e" if res["confidence"] > 70 else "#f59e0b" if res["confidence"] > 40 else "#ef4444"
                    st.markdown(f'<div style="color:#64748b;font-size:0.85rem;margin-bottom:6px;">Question</div><div style="font-size:1rem;font-weight:600;color:#a5b4fc;margin-bottom:16px;">{res["question"]}</div><div style="color:#64748b;font-size:0.85rem;margin-bottom:6px;">Answer</div><div class="answer-box">{res["answer"]}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="metric-card" style="display:inline-block;margin-top:12px;"><span style="color:#64748b;font-size:0.8rem;">Confidence: </span><span style="color:{conf_color};font-weight:700;font-size:1rem;">{res["confidence"]}%</span></div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"QA failed: {e}")

    # ── Dependency ────────────────────────────────────────────────────────────
    with tabs[5]:
        with st.spinner("Generating dependency parse..."):
            try:
                res = core.dependency_parse(text_input)
                st.markdown(f"**Parsing:** *{res['sentence']}*")
                st.markdown("#### Dependency Parse Tree")
                st.markdown(f'<div style="background:#0d1117;border:1px solid rgba(255,255,255,0.07);border-radius:14px;padding:16px;overflow-x:auto;">{res["svg"]}</div>', unsafe_allow_html=True)
                st.markdown("#### Token Details")
                df = pd.DataFrame(res["tokens"])
                df.columns = ["Token","POS Tag","Dependency","Head"]
                st.dataframe(df, use_container_width=True, height=260)
                st.caption("Only the first sentence is parsed. Full-text dependency trees are unreadable.")
            except Exception as e:
                st.error(f"Dependency parse failed: {e}")

else:
    st.markdown('<div style="text-align:center;padding:60px 20px;color:#475569;"><div style="font-size:3rem;margin-bottom:16px;">🧠</div><h3 style="color:#64748b;font-weight:500;">Paste your text above and click Analyze</h3><p style="color:#334155;margin-top:8px;">Or try one of the sample texts from the sidebar →</p></div>', unsafe_allow_html=True)

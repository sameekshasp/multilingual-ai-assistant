"""
Streamlit App — Multilingual AI Assistant (Friendly Redesign)
=============================================================
Clean tab-based layout. App loads instantly — no startup blocker.
Models load on first use of each tab with a simple inline spinner.

Run with:
    streamlit run frontend/app.py
"""

import sys, os
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import torch
from transformers import pipeline, BlenderbotTokenizer, BlenderbotForConditionalGeneration
from langdetect import detect_langs, LangDetectException

# ── Language maps ──────────────────────────────────────────────────────────
LANGUAGE_CODE_MAP = {
    "English": "eng_Latn", "French": "fra_Latn", "Spanish": "spa_Latn",
    "German": "deu_Latn", "Italian": "ita_Latn", "Portuguese": "por_Latn",
    "Dutch": "nld_Latn", "Russian": "rus_Cyrl",
    "Chinese (Simplified)": "zho_Hans", "Chinese (Traditional)": "zho_Hant",
    "Japanese": "jpn_Jpan", "Korean": "kor_Hang", "Arabic": "arb_Arab",
    "Hindi": "hin_Deva", "Bengali": "ben_Beng", "Urdu": "urd_Arab",
    "Turkish": "tur_Latn", "Vietnamese": "vie_Latn", "Polish": "pol_Latn",
    "Swedish": "swe_Latn", "Norwegian": "nob_Latn", "Danish": "dan_Latn",
    "Finnish": "fin_Latn", "Greek": "ell_Grek", "Hebrew": "heb_Hebr",
    "Thai": "tha_Thai", "Indonesian": "ind_Latn", "Malay": "zsm_Latn",
    "Swahili": "swh_Latn", "Marathi": "mar_Deva", "Tamil": "tam_Taml",
    "Telugu": "tel_Telu", "Kannada": "kan_Knda", "Gujarati": "guj_Gujr",
    "Punjabi": "pan_Guru",
}
SUPPORTED_LANGUAGES = list(LANGUAGE_CODE_MAP.keys())

ISO_TO_FRIENDLY = {
    "en":"English","fr":"French","es":"Spanish","de":"German","it":"Italian",
    "pt":"Portuguese","nl":"Dutch","ru":"Russian","zh-cn":"Chinese (Simplified)",
    "zh-tw":"Chinese (Traditional)","ja":"Japanese","ko":"Korean","ar":"Arabic",
    "hi":"Hindi","bn":"Bengali","ur":"Urdu","tr":"Turkish","vi":"Vietnamese",
    "pl":"Polish","sv":"Swedish","no":"Norwegian","da":"Danish","fi":"Finnish",
    "el":"Greek","he":"Hebrew","th":"Thai","id":"Indonesian","ms":"Malay",
    "sw":"Swahili","mr":"Marathi","ta":"Tamil","te":"Telugu","kn":"Kannada",
    "gu":"Gujarati","pa":"Punjabi",
}

DEVICE = 0 if torch.cuda.is_available() else -1
CHAT_MODEL = "facebook/blenderbot-400M-distill"

# ── Cached model loaders ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_translator():
    return pipeline("translation", model="facebook/nllb-200-distilled-600M",
                    device=DEVICE)

@st.cache_resource(show_spinner=False)
def load_sentiment():
    return pipeline("sentiment-analysis",
                    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
                    top_k=1, device=DEVICE)

@st.cache_resource(show_spinner=False)
def load_chatbot():
    tok = BlenderbotTokenizer.from_pretrained(CHAT_MODEL)
    mdl = BlenderbotForConditionalGeneration.from_pretrained(CHAT_MODEL)
    mdl.eval()
    return tok, mdl

# ── Inference helpers ──────────────────────────────────────────────────────
def run_detect(text):
    try:
        res = detect_langs(text[:1000])
        top = res[0]
        code = str(top.lang)
        name = ISO_TO_FRIENDLY.get(code, code.upper())
        return code, name, round(top.prob, 4)
    except LangDetectException:
        return "unknown", "Unknown", 0.0

def run_translate(text, target_lang):
    code, src_name, _ = run_detect(text)
    src_code = LANGUAGE_CODE_MAP.get(ISO_TO_FRIENDLY.get(code, "English"), "eng_Latn")
    tgt_code = LANGUAGE_CODE_MAP.get(target_lang, "eng_Latn")
    model = load_translator()
    with torch.no_grad():
        result = model(text, src_lang=src_code, tgt_lang=tgt_code, max_length=200)
    return result[0]["translation_text"], src_name

def run_sentiment(text):
    model = load_sentiment()
    with torch.no_grad():
        res = model(text[:512])
    top = res[0][0] if isinstance(res[0], list) else res[0]
    label = top["label"].lower()
    label_map = {"positive":"Positive","negative":"Negative","neutral":"Neutral",
                 "label_0":"Negative","label_1":"Neutral","label_2":"Positive"}
    norm = label_map.get(label, label.capitalize())
    emoji = {"Positive":"😊","Negative":"😞","Neutral":"😐"}.get(norm,"🤔")
    return norm, emoji, round(top["score"], 4)

def run_chat(text):
    tok, mdl = load_chatbot()
    inputs = tok([text], return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        ids = mdl.generate(**inputs, max_new_tokens=80, num_beams=1, do_sample=False)
    return tok.batch_decode(ids, skip_special_tokens=True)[0].strip()

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(page_title="Multilingual AI Assistant", page_icon="🌐",
                   layout="centered")

# ── CSS ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

*, html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

/* White background */
.stApp, .main { background-color: #F8FAFC !important; }

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2rem !important; max-width: 760px !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #F1F5F9;
    border-radius: 12px;
    padding: 4px;
    border: none !important;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 8px 20px !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: #64748B !important;
    background: transparent !important;
    border: none !important;
    transition: all 0.2s ease !important;
}
.stTabs [aria-selected="true"] {
    background: #FFFFFF !important;
    color: #4F46E5 !important;
    font-weight: 600 !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08) !important;
}

/* ── Text area ── */
.stTextArea textarea {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 12px !important;
    font-size: 0.95rem !important;
    color: #1E293B !important;
    background: #FFFFFF !important;
    padding: 12px 14px !important;
    transition: border-color 0.2s !important;
}
.stTextArea textarea:focus {
    border-color: #4F46E5 !important;
    box-shadow: 0 0 0 3px rgba(79,70,229,0.1) !important;
}
.stTextArea textarea::placeholder { color: #94A3B8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: #4F46E5 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.5rem !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: background 0.2s, transform 0.15s !important;
    width: 100% !important;
}
.stButton > button:hover {
    background: #4338CA !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Selectbox ── */
.stSelectbox > div > div {
    border: 1.5px solid #E2E8F0 !important;
    border-radius: 10px !important;
    background: #FFFFFF !important;
    color: #1E293B !important;
}

/* ── Result box ── */
.result-box {
    background: #FFFFFF;
    border: 1.5px solid #E2E8F0;
    border-radius: 14px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
    color: #1E293B;
    font-size: 0.95rem;
    line-height: 1.65;
    animation: pop 0.25s ease;
}
@keyframes pop {
    from { opacity:0; transform:translateY(6px); }
    to   { opacity:1; transform:translateY(0); }
}
.result-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    color: #94A3B8;
    margin-bottom: 6px;
}
.result-value { font-size: 1.05rem; font-weight: 600; color: #1E293B; }

/* Sentiment chips */
.chip { display:inline-block; border-radius:999px; padding:4px 14px;
        font-size:0.85rem; font-weight:600; }
.chip-pos { background:#DCFCE7; color:#166534; }
.chip-neg { background:#FEE2E2; color:#991B1B; }
.chip-neu { background:#F1F5F9; color:#475569; }

/* Confidence bar */
.bar-wrap { background:#F1F5F9; border-radius:999px; height:6px; margin-top:10px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:999px; transition:width 0.5s ease; }

/* Chat bubbles */
.bubble-user {
    background: #4F46E5; color: #fff;
    border-radius: 16px 16px 4px 16px;
    padding: 10px 14px; margin: 6px 0 6px auto;
    max-width: 80%; font-size: 0.9rem; line-height: 1.5;
    display: table; margin-left: auto;
}
.bubble-ai {
    background: #F1F5F9; color: #1E293B;
    border-radius: 16px 16px 16px 4px;
    padding: 10px 14px; margin: 6px auto 6px 0;
    max-width: 80%; font-size: 0.9rem; line-height: 1.5;
    display: table;
}
.bubble-label { font-size:0.72rem; color:#94A3B8; margin-bottom:2px; font-weight:500; }

label { color: #374151 !important; font-size: 0.85rem !important;
        font-weight: 500 !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-bottom:1.5rem;">
    <div style="font-size:2rem; margin-bottom:6px;">🌐</div>
    <h1 style="font-size:1.6rem; font-weight:700; color:#1E293B; margin:0;">
        Multilingual AI Assistant
    </h1>
    <p style="color:#64748B; font-size:0.9rem; margin:6px 0 0;">
        Detect language · Translate · Analyse sentiment · Chat with AI
    </p>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔍  Detect Language",
    "🌍  Translate",
    "💭  Sentiment",
    "🤖  Chat with AI",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Language Detection
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("#### Detect the language of any text")
    st.caption("Supports 55+ languages. Works instantly — no AI model needed.")

    text1 = st.text_area("Enter text", placeholder="Type or paste your text here…",
                          height=130, key="t1", label_visibility="collapsed")

    char1 = len(text1)
    st.markdown(
        f'<div style="text-align:right;font-size:0.75rem;color:{"#94A3B8" if char1<800 else "#EF4444"}'
        f';margin-top:-10px;margin-bottom:8px;">{char1} / 1000</div>',
        unsafe_allow_html=True)

    if st.button("🔍 Detect Language", key="b1"):
        if not text1.strip():
            st.warning("Please enter some text first.", icon="⚠️")
        else:
            with st.spinner("Detecting…"):
                code, name, conf = run_detect(text1)
            if code == "unknown":
                st.error("Could not detect language. Try entering more text.", icon="❌")
            else:
                pct = int(conf * 100)
                bar_col = "#4F46E5" if conf > 0.8 else "#F59E0B"
                st.markdown(f"""
                <div class="result-box">
                    <div class="result-label">Detected Language</div>
                    <div class="result-value" style="font-size:1.3rem;">{name}</div>
                    <div style="font-size:0.78rem;color:#94A3B8;margin-top:2px;">
                        ISO code: <code style="background:#F1F5F9;padding:1px 7px;
                        border-radius:5px;font-size:0.78rem;">{code}</code>
                    </div>
                    <div style="margin-top:12px;">
                        <div style="display:flex;justify-content:space-between;
                                    font-size:0.75rem;color:#94A3B8;">
                            <span>Confidence</span><span>{pct}%</span>
                        </div>
                        <div class="bar-wrap">
                            <div class="bar-fill"
                                 style="width:{pct}%;background:{bar_col};"></div>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Translation
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("#### Translate text to any language")
    st.caption("Powered by Facebook NLLB-200. Loads on first use (~1–2 min).")

    col_a, col_b = st.columns([3, 2])
    with col_a:
        text2 = st.text_area("Text to translate",
                              placeholder="Enter text in any language…",
                              height=130, key="t2", label_visibility="collapsed")
    with col_b:
        st.markdown('<p style="font-size:0.83rem;font-weight:500;color:#374151;margin-bottom:4px;">Translate to</p>', unsafe_allow_html=True)
        target_lang = st.selectbox("Target", SUPPORTED_LANGUAGES,
                                   label_visibility="collapsed", key="tl")

    if st.button("🌍 Translate", key="b2"):
        if not text2.strip():
            st.warning("Please enter some text to translate.", icon="⚠️")
        else:
            with st.spinner("Translating… (first use loads the model — ~1 min)"):
                try:
                    translated, src_name = run_translate(text2, target_lang)
                    st.markdown(f"""
                    <div class="result-box">
                        <div style="display:flex;justify-content:space-between;
                                    align-items:center;margin-bottom:10px;">
                            <div class="result-label" style="margin:0;">Translation</div>
                            <div style="font-size:0.78rem;color:#64748B;">
                                {src_name} → <strong>{target_lang}</strong>
                            </div>
                        </div>
                        <div style="background:#F8FAFC;border:1px solid #E2E8F0;
                                    border-radius:10px;padding:12px 14px;
                                    font-size:1rem;color:#1E293B;line-height:1.65;">
                            {translated}
                        </div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Translation failed: {e}", icon="❌")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Sentiment Analysis
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("#### Analyse the sentiment of your text")
    st.caption("Supports English, French, Spanish, German, Arabic, Hindi, Italian, Portuguese.")

    text3 = st.text_area("Text to analyse", placeholder="e.g. I love this product!",
                          height=130, key="t3", label_visibility="collapsed")

    if st.button("💭 Analyse Sentiment", key="b3"):
        if not text3.strip():
            st.warning("Please enter some text to analyse.", icon="⚠️")
        else:
            with st.spinner("Analysing… (first use loads the model — ~30 sec)"):
                try:
                    label, emoji, conf = run_sentiment(text3)
                    chip_class = {"Positive":"chip-pos","Negative":"chip-neg","Neutral":"chip-neu"}.get(label,"chip-neu")
                    bar_col = {"Positive":"#22C55E","Negative":"#EF4444","Neutral":"#94A3B8"}.get(label,"#94A3B8")
                    pct = int(conf * 100)
                    st.markdown(f"""
                    <div class="result-box">
                        <div class="result-label">Sentiment Result</div>
                        <div style="display:flex;align-items:center;gap:12px;margin:8px 0;">
                            <span style="font-size:2rem;">{emoji}</span>
                            <span class="chip {chip_class}">{label}</span>
                        </div>
                        <div style="margin-top:10px;">
                            <div style="display:flex;justify-content:space-between;
                                        font-size:0.75rem;color:#94A3B8;">
                                <span>Confidence</span><span>{pct}%</span>
                            </div>
                            <div class="bar-wrap">
                                <div class="bar-fill"
                                     style="width:{pct}%;background:{bar_col};"></div>
                            </div>
                        </div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Analysis failed: {e}", icon="❌")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — Chat with AI
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("#### Chat with the AI")
    st.caption("Powered by BlenderBot. Best in English. Loads on first use (~30 sec).")

    text4 = st.text_area("Your message",
                          placeholder="Say something… e.g. What is your favourite hobby?",
                          height=100, key="t4", label_visibility="collapsed")

    if st.button("🤖 Send Message", key="b4"):
        if not text4.strip():
            st.warning("Please type a message first.", icon="⚠️")
        else:
            with st.spinner("Generating response…"):
                try:
                    reply = run_chat(text4)
                    st.markdown(f"""
                    <div style="margin-top:1rem;">
                        <div class="bubble-label">You</div>
                        <div class="bubble-user">{text4}</div>
                        <div class="bubble-label" style="margin-top:10px;">AI</div>
                        <div class="bubble-ai">{reply}</div>
                    </div>""", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Chat failed: {e}", icon="❌")


# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;margin-top:3rem;padding-top:1rem;
            border-top:1px solid #E2E8F0;">
    <p style="font-size:0.75rem;color:#94A3B8;">
        Powered by 🤗 Hugging Face Transformers &amp; Streamlit &nbsp;·&nbsp;
        Runs 100% locally
    </p>
</div>
""", unsafe_allow_html=True)

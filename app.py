import html
import os

import pandas as pd
import streamlit as st

from model import get_training_metrics
from scoring import run_analysis
from storage import log_analysis

st.set_page_config(page_title="News Credibility Checker", page_icon="📰", layout="wide")

CSS = """
<style>
:root{--card:rgba(255,255,255,.045);--line:rgba(255,255,255,.09);--muted:#94A3B8;--accent:#4F8CFF;
--good:#34D399;--mid:#FBBF24;--bad:#F87171;--na:#64748B;}
.stApp{background:radial-gradient(1100px 500px at 15% -10%,#16264a 0%,transparent 60%),#0A1220;}
[data-testid="stToolbar"],[data-testid="stDecoration"],#MainMenu,footer{display:none;}
.block-container{padding-top:2rem;max-width:1180px;}
.title{font-size:2rem;font-weight:800;letter-spacing:-.02em;margin:0;}
.sub{color:var(--muted);margin:.2rem 0 1.4rem;}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.3rem 1.4rem;margin-bottom:.9rem;}
.hero{display:flex;gap:1.4rem;align-items:center;}
.ring{position:relative;width:120px;height:120px;flex:none;}
.ring svg{transform:rotate(-90deg);}
.ring b{position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:2.1rem;font-weight:800;}
.pill{display:inline-block;padding:.35rem .9rem;border-radius:999px;font-weight:700;font-size:.95rem;margin-bottom:.5rem;}
.pill.good{background:rgba(52,211,153,.15);color:var(--good)}.pill.mid{background:rgba(251,191,36,.15);color:var(--mid)}.pill.bad{background:rgba(248,113,113,.15);color:var(--bad)}
.hero p{margin:0;color:#CBD5E1;line-height:1.5;}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:.7rem;margin-bottom:.9rem;}
.tile{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:.9rem 1rem;}
.tile span{color:var(--muted);font-size:.8rem;}
.tile div{font-weight:600;margin-top:.15rem;display:flex;align-items:center;gap:.5rem;}
.dot{width:9px;height:9px;border-radius:50%;background:var(--na);flex:none;}
.dot.good{background:var(--good)}.dot.mid{background:var(--mid)}.dot.bad{background:var(--bad)}
mark.flag{background:rgba(251,191,36,.28);color:#fff;padding:0 3px;border-radius:4px;}
.reading{line-height:1.7;color:#E2E8F0;}
.match{padding:.5rem 0;border-bottom:1px solid var(--line);font-size:.92rem;}
.match:last-child{border:0}.match a{color:#8DB4FF;text-decoration:none}.match small{color:var(--muted);display:block}
.badge{background:rgba(52,211,153,.15);color:var(--good);font-size:.7rem;padding:1px 7px;border-radius:9px;margin-left:6px;font-weight:700;}
.badge.bad{background:rgba(248,113,113,.15);color:var(--bad);}
.empty{color:var(--muted);line-height:1.7;}
div[data-testid="stForm"]{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:1.1rem 1.2rem;}
.stButton>button,.stFormSubmitButton>button{border-radius:10px;font-weight:600;}
@media(max-width:700px){.hero{flex-direction:column;align-items:flex-start}.grid{grid-template-columns:1fr}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def h(s: str) -> str:
    return "".join(line.strip() for line in s.splitlines())


def esc(x) -> str:
    return html.escape(str(x))


EXAMPLES = {
    "Real report": ("ISRO schedules Earth observation satellite launch",
                    "ISRO officials said on Monday that the launch window remains on schedule after final technical checks. "
                    "A spokesperson confirmed the payload will be integrated this week, according to a statement from the agency.", "Reuters"),
    "Miracle cure": ("SHOCKING: One secret herb cures all diseases instantly, doctors hate it!!",
                     "Leaked report EXPOSES the ancient secret big pharma is hiding. 100% guaranteed results overnight. Share this before it's deleted!", "ViralNewsHub"),
    "Calm fake": ("Drinking hot lemon water every morning cures cancer, researchers say",
                  "Researchers said that a daily glass of hot lemon water cures cancer completely, with no treatment needed.", ""),
}


def load_example(name: str):
    st.session_state.in_title, st.session_state.in_text, st.session_state.in_source = EXAMPLES[name]


def secret_key() -> str:
    try:
        return st.secrets.get("GOOGLE_FACTCHECK_API_KEY", "")
    except Exception:
        return ""


@st.cache_data(ttl=1800, show_spinner=False)
def cached_analysis(title, text, source, use_web, key):
    return run_analysis(title, text, source, use_web, key or None)


with st.sidebar:
    st.subheader("Settings")
    use_web = st.toggle("Cross-check on the web", value=True)
    key = st.text_input("Fact-check API key (optional)", type="password", value=os.getenv("GOOGLE_FACTCHECK_API_KEY") or secret_key())
    tm = get_training_metrics()
    st.caption(f"Classifier trained on {tm.get('unique_articles', '?')} articles. Small dataset, so it has little say in the result.")

st.markdown('<p class="title">News Credibility Checker</p><p class="sub">Checks the source, the wording, and what trusted outlets are reporting.</p>', unsafe_allow_html=True)

left, right = st.columns([5, 6], gap="large")

with left:
    cols = st.columns(len(EXAMPLES))
    for col, name in zip(cols, EXAMPLES):
        col.button(name, on_click=load_example, args=(name,), width="stretch")
    with st.form("check", border=False):
        st.text_input("Headline", key="in_title", placeholder="Paste the headline")
        st.text_area("Article text", key="in_text", height=200, placeholder="Paste the article (optional but more accurate)")
        st.text_input("Source", key="in_source", placeholder="Outlet name or article URL")
        submitted = st.form_submit_button("Check credibility", type="primary", width="stretch")
    if submitted:
        t, x, s = (st.session_state.get(k, "").strip() for k in ("in_title", "in_text", "in_source"))
        if not t and not x:
            st.warning("Enter a headline or some article text.")
        else:
            with st.spinner("Checking..."):
                result = cached_analysis(t, x, s, use_web, key)
            log_analysis(result, t, s)
            st.session_state.result = result

with right:
    res = st.session_state.get("result")
    if not res:
        st.markdown('<div class="card empty">Your result will appear here.<br><br>Try one of the examples, or paste your own headline.</div>', unsafe_allow_html=True)
    else:
        band = {"Likely reliable": "good", "Needs verification": "mid", "Likely misleading": "bad"}[res["verdict"]]
        color = f"var(--{band})"
        circ = 2 * 3.14159 * 52
        off = circ * (1 - res["score"] / 100)
        st.markdown(h(f"""
        <div class="card hero">
          <div class="ring"><svg width="120" height="120"><circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,.1)" stroke-width="10"/>
          <circle cx="60" cy="60" r="52" fill="none" stroke="{color}" stroke-width="10" stroke-linecap="round" stroke-dasharray="{circ:.1f}" stroke-dashoffset="{off:.1f}"/></svg><b>{res['score']}</b></div>
          <div><span class="pill {band}">{esc(res['verdict'])}</span><p>{esc(res['summary'])}</p></div>
        </div>
        <div class="grid">""" + "".join(
            f'<div class="tile"><span>{esc(c["label"])}</span><div><i class="dot {c["band"]}"></i>{esc(c["short"] or c["summary"][:40])}</div></div>' for c in res["components"]
        ) + "</div>"), unsafe_allow_html=True)

        with st.expander("Why this result"):
            for c in res["components"]:
                st.markdown(f"**{c['label']}**: {c['summary']}")
            st.markdown("**Next steps**")
            for s_ in res["next_steps"]:
                st.markdown(f"- {s_}")

        with st.expander("Highlighted text"):
            st.markdown(h(f'<div class="reading">{res["highlighted_html"]}</div>'), unsafe_allow_html=True)
            for f in res["language"]["findings"]:
                st.caption(f"{f['title']}: {f['detail']}")

        web = res["web"]
        with st.expander(f"Web matches ({len(web.get('matches', []))})"):
            st.caption(web.get("summary", ""))
            rows = "".join(
                f'<div class="match"><a href="{esc(m["url"])}" target="_blank" rel="noopener">{esc(m["title"])}</a>'
                f'{"<span class=badge>trusted</span>" if m["trusted"] else ""}{"<span class=\'badge bad\'>disputes</span>" if m.get("debunk") else ""}'
                f'<small>{esc(m["source"])} {esc(m["published"])}</small></div>' for m in web.get("matches", [])[:8])
            if rows:
                st.markdown(h(f'<div class="card">{rows}</div>'), unsafe_allow_html=True)
            for fc in web.get("fact_checks", []):
                st.markdown(f"**{fc['publisher']}** rated *{fc['rating']}*: {fc['claim']} ([source]({fc['url']}))")
            if web.get("fact_check_note"):
                st.caption(web["fact_check_note"])

        with st.expander("Classifier details"):
            m = res["model"]
            st.write(f"Leans **{m['prediction']}** ({m['confidence']:.0%}). Recognised {m['vocab_coverage']:.0%} of the words.")
            if m["top_terms"]:
                st.dataframe(pd.DataFrame(m["top_terms"])[["term", "toward", "push"]], hide_index=True, width="stretch")

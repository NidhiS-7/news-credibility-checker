import os

import pandas as pd
import streamlit as st

from model import DATASET_PATH, dataset_quality, get_training_metrics, load_dataset, train_model
from storage import DATA_DIR, load_metrics, read_history

st.set_page_config(page_title="Admin | News Credibility Checker", layout="wide")
st.title("Admin")

password = os.getenv("ADMIN_PASSWORD", "")
if password and st.text_input("Admin password", type="password") != password:
    st.info("Enter the admin password to continue. Set the ADMIN_PASSWORD environment variable to change it.")
    st.stop()

metrics, tm = load_metrics(), get_training_metrics()
history = read_history()

c1, c2, c3, c4 = st.columns(4)
c1.metric("Articles analysed", metrics.get("articles_analyzed", 0))
c2.metric("Unique training articles", tm.get("unique_articles", 0))
c3.metric("Cross-validated accuracy", f"{tm.get('accuracy', 0):.1%}", f"± {tm.get('accuracy_std', 0):.1%}", delta_color="off")
c4.metric("Transformer branch", "On" if tm.get("transformer_enabled") else "Off")

tab_model, tab_data, tab_hist = st.tabs(["Model", "Dataset", "Activity"])

with tab_model:
    st.caption(f"Last trained {tm.get('trained_at', 'unknown')}. Method: {tm.get('method', 'n/a')}.")
    for w in tm.get("warnings", []):
        st.warning(w)
    cm = tm.get("confusion_matrix")
    if cm:
        st.dataframe(pd.DataFrame(cm, index=["Actually real", "Actually fake"], columns=["Predicted real", "Predicted fake"]), width="stretch")
    st.write({k: tm.get(k) for k in ("precision", "recall", "f1", "roc_auc")})
    use_t = st.checkbox("Include transformer embeddings (needs sentence-transformers)", value=False)
    if st.button("Retrain model", type="primary"):
        with st.spinner("Training..."):
            try:
                train_model(use_transformer=use_t)
                st.success("Model retrained.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

with tab_data:
    if DATASET_PATH.exists():
        q = dataset_quality()
        a, b, c = st.columns(3)
        a.metric("Rows", q["rows"])
        b.metric("Unique articles", q["unique_articles"])
        c.metric("Class split", " / ".join(f"{k}: {v}" for k, v in q["label_counts"].items()))
        for w in q["warnings"]:
            st.warning(w)
        st.dataframe(load_dataset().drop(columns=["combined_text"]).head(100), width="stretch", hide_index=True)
    else:
        st.warning("No dataset found at data/dataset.csv.")
    up = st.file_uploader("Replace dataset (CSV with title, text, source, label columns)", type="csv")
    if up is not None and st.button("Use uploaded dataset"):
        try:
            tmp = DATA_DIR / "_upload.csv"
            tmp.write_bytes(up.getvalue())
            load_dataset(tmp)  # validates columns and labels
            tmp.replace(DATASET_PATH)
            st.success("Dataset replaced. Retrain the model on the Model tab.")
        except Exception as exc:
            st.error(f"Could not use that file: {exc}")

with tab_hist:
    if history:
        df = pd.DataFrame(history)
        st.bar_chart(df["verdict"].value_counts())
        st.dataframe(df.iloc[::-1], width="stretch", hide_index=True)
    else:
        st.caption("No analyses recorded yet. Only the headline, source, verdict and score are stored; never the article body.")

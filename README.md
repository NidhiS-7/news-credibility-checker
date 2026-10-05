# News Credibility Checker

Run:  `pip install -r requirements.txt` then `streamlit run app.py`
(the first run trains the classifier automatically; the Admin page is in the sidebar).

Signals: source (name or URL domain) · writing style · live web cross-check (Google News RSS, no key needed)
· optional Google Fact Check API (`GOOGLE_FACTCHECK_API_KEY`) · text classifier (low weight, scaled by training size).

Edit `data/trusted_sources.json` to extend the trusted / unreliable outlet lists.
Replace `data/dataset.csv` with a real corpus (e.g. LIAR, ISOT, FakeNewsNet) and retrain from the Admin page.
Set `ADMIN_PASSWORD` to protect the Admin page.

"""
Urban Regeneration Platform — Streamlit App
Run: streamlit run apps/streamlit/app.py --server.port=8501 --server.address=0.0.0.0
"""

import streamlit as st

st.set_page_config(
    page_title="Urban Regeneration Platform",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Navigation ───────────────────────────────────────────────────────────────

PAGES = {
    "EDA": {
        "🗺 Heatmap":          "eda_heatmap",
        "📊 Correlation":      "eda_correlation",
        "📈 Parallel Trends":  "eda_parallel",
    },
    "Causal Inference": {
        "🔥 Uplifts":               "ci_uplifts",
        "🔬 CATE":                  "ci_cate",
        "🏆 Policy Recommendations": "ci_policy",
    },
}

with st.sidebar:
    st.title("🏙️ Urban Regeneration")
    st.divider()

    st.subheader("Section")
    section = st.radio("", list(PAGES.keys()), label_visibility="collapsed")
    st.subheader("Dashboard")
    page_label = st.radio("", list(PAGES[section].keys()), label_visibility="collapsed")
    st.divider()

    interactive = st.toggle("Interactive map", value=True)
    st.caption("Interactive = Folium (st_folium). Static = matplotlib (exportable PNG).")

# ── Route ────────────────────────────────────────────────────────────────────

module_name = PAGES[section][page_label]

if module_name == "eda_heatmap":
    from apps.streamlit.views.eda_heatmap import render
elif module_name == "eda_correlation":
    from apps.streamlit.views.eda_correlation import render
elif module_name == "eda_parallel":
    from apps.streamlit.views.eda_parallel import render
elif module_name == "ci_uplifts":
    from apps.streamlit.views.ci_uplifts import render
elif module_name == "ci_cate":
    from apps.streamlit.views.ci_cate import render
elif module_name == "ci_policy":
    from apps.streamlit.views.ci_policy import render
else:
    def render(interactive=True):
        st.error(f"Unknown page: {module_name}")

render(interactive=interactive)

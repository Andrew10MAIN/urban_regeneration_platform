"""
Export helpers: PNG bytes from matplotlib, HTML bytes from folium.
"""

from __future__ import annotations

import io
import folium
import matplotlib.pyplot as plt


def map_to_html_bytes(m: folium.Map) -> bytes:
    buf = io.BytesIO()
    html = m._repr_html_()
    buf.write(html.encode("utf-8"))
    buf.seek(0)
    return buf.read()


def map_to_html_str(m: folium.Map) -> str:
    return m._repr_html_()


def fig_to_png_bytes(fig: plt.Figure, dpi: int = 150) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.read()


def download_button_map(m: folium.Map, filename: str = "map.html") -> None:
    """Streamlit download button for a folium map."""
    import streamlit as st
    st.download_button(
        label="⬇ Download map (HTML)",
        data=map_to_html_bytes(m),
        file_name=filename,
        mime="text/html",
    )


def download_button_fig(fig: plt.Figure, filename: str = "chart.png", dpi: int = 150) -> None:
    """Streamlit download button for a matplotlib figure."""
    import streamlit as st
    st.download_button(
        label="⬇ Download image (PNG)",
        data=fig_to_png_bytes(fig, dpi=dpi),
        file_name=filename,
        mime="image/png",
    )

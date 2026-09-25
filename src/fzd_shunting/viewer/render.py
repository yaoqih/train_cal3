"""One offline HTML document, shared by Streamlit and downloadable replay."""

import json
from pathlib import Path

ASSETS = Path(__file__).parent / "assets"


def render_html(view):
    payload = json.dumps(
        view, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    )
    payload = (
        payload.replace("<", "\\u003c")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
    template = (ASSETS / "viewer.html").read_text()
    return (
        template.replace("/*__STYLE__*/", (ASSETS / "viewer.css").read_text())
        .replace("/*__SCRIPT__*/", (ASSETS / "viewer.js").read_text())
        .replace("/*__DATA__*/", payload)
    )

"""Run with: python -m streamlit run app.py"""

import json
import sys
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from fzd_shunting.viewer import build_view, render_html

st.set_page_config(page_title="福州东 · 调车查看器", page_icon="🚆", layout="wide")
st.markdown(
    """<style>
    .block-container {padding: .4rem 1rem 1rem; max-width: 1800px;}
    [data-testid="stHeader"] {background: transparent; height: 2.4rem;}
    [data-testid="stSidebar"] {background: #f8fafc; border-right: 1px solid #e3e9ef;}
    [data-testid="stSidebar"] .block-container {padding-top: 1.5rem;}
    [data-testid="stSidebar"] h1 {font-size: 1.35rem;}
    #MainMenu, footer {visibility: hidden;}
</style>""",
    unsafe_allow_html=True,
)


def read_json(path):
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8-sig"))


def project_picker(key):
    kind = st.radio("文件类型", ["请求", "计划"], horizontal=True, key=key + "_kind")
    if kind == "请求":
        variants = [
            name
            for name in ("normalized", "augmented")
            if (ROOT / "data/point_to_area" / name).is_dir()
        ]
        labels = {"normalized": "规范请求", "augmented": "含顺序约束"}
        if variants:
            variant = st.selectbox(
                "请求版本", variants, format_func=labels.get, key=key + "_variant"
            )
            files = sorted((ROOT / "data/point_to_area" / variant).glob("*.json"))
        else:
            files = sorted((ROOT / "scenarios").glob("*.json"))
    else:
        files = []
        for path in sorted((ROOT / "runs").glob("*plan*.json")):
            try:
                data = read_json(path)
            except (OSError, ValueError):
                continue
            if isinstance(data, dict) and data.get("schema_version") == 2:
                files.append(path)
    if not files:
        st.caption("这里还没有 JSON 文件，可改用上传或本地路径。")
        return None, None
    path = st.selectbox(
        "选择文件", files, format_func=lambda p: p.name, key=key + "_file"
    )
    return read_json(path), path.stem


def input_picker(key, companion=False):
    choices = (
        ["上传 JSON", "项目文件", "本地路径"]
        if companion
        else ["示例回放", "项目文件", "上传 JSON", "本地路径"]
    )
    source = st.selectbox(
        "数据来源" if not companion else "配套文件来源", choices, key=key + "_source"
    )
    if source == "示例回放":
        return read_json(ROOT / "scenarios/viewer-demo.json"), "示例 · 前场车辆取送"
    if source == "项目文件":
        return project_picker(key)
    if source == "上传 JSON":
        upload = st.file_uploader(
            "规范请求、schema-2 计划或 request / plan 合集",
            type=["json"],
            key=key + "_upload",
        )
        if upload is not None:
            return (
                json.loads(upload.getvalue().decode("utf-8-sig")),
                Path(upload.name).stem,
            )
        return None, None
    path = st.text_input(
        "JSON 文件路径", placeholder="/root/.../request.json", key=key + "_path"
    )
    if path.strip():
        return read_json(path.strip()), Path(path.strip()).stem
    return None, None


try:
    with st.sidebar:
        st.title("打开数据")
        primary, title = input_picker("primary")
        secondary = None
        with st.expander("搭配另一份请求或计划", expanded=False):
            st.caption(
                "只有动作的计划需要初始请求；已有快照的计划可补充请求来显示车辆属性与目标。"
            )
            enabled = st.checkbox("启用配套文件")
            if enabled:
                secondary, secondary_title = input_picker("secondary", companion=True)
                if secondary is not None:
                    title = (title or "") + " + " + secondary_title
        st.divider()
        st.caption("仅查看与回放，不启动调度求解。")
    if primary is None:
        st.info(
            "从左侧打开 train_cal3 规范请求、schema-2 勾计划或 request / plan 合集。"
        )
        st.stop()
    view = build_view(primary, secondary, title=title)
    if view["needs_request"]:
        st.info(
            "这份计划只有动作记录。请在左侧展开“搭配另一份请求或计划”，打开对应的初始请求。"
        )
        st.stop()
    document = render_html(view)
    with st.sidebar:
        st.download_button(
            "下载交互回放 HTML",
            document,
            file_name="shunting-replay.html",
            mime="text/html",
            use_container_width=True,
            help="无需启动服务，浏览器打开即可离线查看和播放。",
        )
        with st.expander("查看器说明"):
            st.caption(
                "车序从北向南，机车车序从靠近机车到远离机车。Get 取北端前缀，Put 放机车尾部。"
            )
            st.caption(
                "有快照时显示原始快照；没有快照时按取放记录重建。数据提示会标注不一致。回放本身不代表计划已经通过物理合法性校验。"
            )
    components.html(document, height=1060, scrolling=True)
except (OSError, ValueError, TypeError, KeyError, AttributeError) as exc:
    st.error("无法打开这份数据：" + str(exc))
    st.caption(
        "请求应含 StartStatus，TargetLines 为目标股道映射；计划使用 schema_version=2 与 actions/events。"
    )

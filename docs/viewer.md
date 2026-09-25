# 调车查看器

使用 `./run_viewer.sh` 启动，默认 `http://127.0.0.1:8502`。本机使用独立 Python 3.10 环境运行 Streamlit；新环境安装 `.[ui]`。查看器只读，不启动调度训练或求解。

## 唯一输入格式

| 输入 | 字段 | 是否需要配套请求 |
| --- | --- | --- |
| 规范请求 | `StartStatus`；目标为 `TargetLines[股道].ForceTargetPosition` | 否 |
| 带快照计划 | `schema_version: 2`、`events[].before/after` | 可选，补充长度、目标等属性 |
| 纯动作计划 | `schema_version: 2`、`actions[].line/operation/count` | 是 |
| 合集 | `{"request": 请求, "plan": 计划}` | 内嵌 |

不支持旧 `Operations`、`Request/Response`、动作裸数组、目标字符串数组或车辆顶层 `ForceTargetPosition`。计划和事件版本不匹配直接报错；项目计划列表只列出当前格式。

默认示例 `scenarios/viewer-demo.json` 不依赖旧项目。项目请求入口只有规范请求和含顺序约束的请求；`raw` 仅为来源档案。

## 交互

站场图、车序表、详情、勾列表与机车车列共享一个时间轴。支持逐勾、播放、拖动、勾列表定位；搜索车辆或股道，点击查看目标及保护信息。手动选择对象后关闭“跟随作业”，便于持续观察。

车序为北→南；机车车列为近→远。保留每勾前后的挂入／摘下变化、车辆数量、含机车总长。只有一个主要数据视图，不重复铺开全部表格。

支持地图平移、缩放、窄屏车序表；方向键逐步、空格播放、Home/End到首末态。可导出包含相同交互的独立HTML，无需Python服务或网络。

## 展示与校验边界

有快照时展示原始记录，并核对动作推演、相邻状态和车辆集合；矛盾会标注，不在后台改写记录。没有快照时只按单股道Get／Put重建；来源不足或动作不合法则停止并保留有效前缀。

中途继续的计划以计划快照为起点，保留全局勾号；原始请求继续提供保护锚点和车辆属性。只有快照、没有请求时，未知长度显示未知。未配置几何布局的股道出现在车序表和地图补充区域。

查看器用于检查记录，不认证完整进路或业务合法性。严格校验请运行 `python -m fzd_shunting replay REQUEST PLAN --rules RULES`；不能把文件自报的 `complete` 当作独立验收。

## 验证

```bash
python -m pytest -q tests/test_viewer.py
python tests/browser_viewer_smoke.py --browser /path/to/chrome
```

浏览器检查覆盖播放、车列变化、搜索、保护车、地图／表格、勾列表跳转、缩放和窄屏，输出到 `runs/ui-checks/`。代码入口为 `viewer/model.py`、`viewer/render.py`、`viewer/assets/` 与 `app.py`。

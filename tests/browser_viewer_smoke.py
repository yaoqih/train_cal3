"""Optional real-browser check: python tests/browser_viewer_smoke.py --browser /path/to/chrome.

Requires playwright; does not download a browser or run a planning solver.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fzd_shunting.viewer import build_view, render_html
from playwright.sync_api import sync_playwright


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser", default=None)
    args = parser.parse_args()
    out = ROOT / "runs/ui-checks"
    out.mkdir(parents=True, exist_ok=True)
    bundle = json.loads((ROOT / "scenarios/viewer-demo.json").read_text())
    page_path = out / "demo.html"
    page_path.write_text(render_html(build_view(bundle, title="示例 · 前场车辆取送")))
    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=args.browser,
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1440, "height": 1100})
        errors = []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.goto(page_path.as_uri())
        page.locator("#next").click()
        assert "5243217" in page.locator("#train-body .train-row").last.inner_text()
        page.screenshot(path=str(out / "desktop.png"), full_page=True)
        page.locator("#next").click()
        assert page.locator("#train-body .removed").inner_text() == "5240771"
        page.locator("#search").fill("1503303")
        page.locator("#search-results [data-car]").click()
        assert "保护车" in page.locator("#details").inner_text()
        assert not page.locator("#follow").is_checked()
        page.locator("#table-tab").click()
        assert page.locator("#table-view").is_visible()
        page.locator("#plan-tab").click()
        page.locator('#plan-list [data-step="4"]').click()
        assert page.locator("#scrubber").input_value() == "4"
        page.locator("#last").click()
        assert page.locator("#next").is_disabled()
        page.locator("#first").click()
        page.locator("#speed").select_option("450")
        page.locator("#play").click()
        page.wait_for_function("document.getElementById('scrubber').value === '6'")
        assert page.locator("#play").inner_text() == "播放"
        page.locator("#scrubber").evaluate(
            "e => {e.value=3; e.dispatchEvent(new Event('input',{bubbles:true}));}"
        )
        assert "第 3 勾" in page.locator("#action-title").inner_text()
        page.locator("#map-tab").click()
        page.locator("#zoom-in").click()
        assert page.locator("#yard-svg").get_attribute("viewBox") != "40 45 2160 1040"
        page.locator("#reset").click()
        page.locator('#yard-svg [data-line="存1线"] .track-hit').click(force=True)
        assert "存1线" in page.locator("#details").inner_text()
        page.set_viewport_size({"width": 390, "height": 844})
        page.goto(page_path.as_uri())
        page.locator("#next").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        page.screenshot(path=str(out / "mobile.png"), full_page=True)
        assert not errors, errors
        browser.close()
    print(
        "Browser checks passed: playback, train changes, search, protection, map/table, plan jump, zoom, mobile."
    )


if __name__ == "__main__":
    main()

(() => {
  "use strict";
  const D = JSON.parse(document.getElementById("viewer-data").textContent);
  const $ = (id) => document.getElementById(id),
    esc = (v) =>
      String(v ?? "").replace(
        /[&<>"']/g,
        (c) =>
          ({
            "&": "&amp;",
            "<": "&lt;",
            ">": "&gt;",
            '"': "&quot;",
            "'": "&#39;",
          })[c],
      );
  let step = 0,
    playing = null,
    mode = window.matchMedia("(max-width:650px)").matches ? "table" : "map",
    panel = "details",
    selected = null,
    follow = true;
  let box = [40, 45, 2160, 1040],
    drag = null,
    dragged = false;
  const frames = D.frames,
    last = frames.length - 1,
    frame = () => frames[step],
    previous = () => frames[Math.max(0, step - 1)];
  const car = (id) => D.vehicles[id] || { no: id, targets: {} },
    track = (name) => D.tracks[name] || {},
    carsAt = (f, line) =>
      line === "__loco__" ? f.train : f.stacks[line] || [];
  const opName = (op) =>
    ({ get: "取", put: "放" })[op];
  const lineName = (line) => (line === "__loco__" ? "机车" : line);
  const kindName = (kind) =>
    ({
      storage: "存车线",
      temporary: "临停线",
      operation: "作业线",
      transit: "走行线路",
      unknown: "自定义股道",
    })[kind] || "股道";
  function location(id, f = frame()) {
    if (f.train.includes(id)) return ["__loco__", f.train.indexOf(id) + 1];
    for (const [line, ids] of Object.entries(f.stacks))
      if (ids.includes(id)) return [line, ids.indexOf(id) + 1];
    return ["未出现在此状态", null];
  }
  function length(ids) {
    let n = 0;
    for (const id of ids) {
      const v = car(id).length;
      if (v == null) return null;
      n += v;
    }
    return Math.round(n * 10) / 10;
  }
  const metres = (v) => (v == null ? "长度未知" : `${v.toFixed(1)} m`);
  const targets = (id) =>
    Object.keys(car(id).targets || {}).join(" / ") || "目标未提供";
  function actionTitle(f, i) {
    if (!f.action) return "初始状态";
    const a = f.action;
    return `第 ${f.hook} 勾 · ${a.line} ${opName(a.operation)}${a.count ? " " + a.count + " 辆" : ""}`;
  }
  function notice(items) {
    return items.length
      ? `<div class="notice">${items.length === 1 ? esc(items[0]) : `<details><summary>${items.length} 条数据提示 · 点击展开</summary><ul>${items.map((x) => `<li>${esc(x)}</li>`).join("")}</ul></details>`}</div>`
      : "";
  }
  function chips(ids, { added = [], removed = [] } = {}) {
    if (!ids.length) return '<span class="muted">空</span>';
    return ids
      .map(
        (id) =>
          `<button class="chip ${added.includes(id) ? "added" : ""} ${removed.includes(id) ? "removed" : ""} ${car(id).protected ? "protected" : ""} ${selected?.car === id ? "selected" : ""}" data-car="${esc(id)}" title="${esc(id + " · " + targets(id) + (car(id).protected ? " · 保护车" : ""))}">${esc(id)}</button>`,
      )
      .join("");
  }
  function setSelected(selection) {
    selected = selection;
    follow = false;
    $("follow").checked = false;
    panel = "details";
    updatePanels();
    render();
  }
  function moveTo(n) {
    step = Math.max(0, Math.min(last, n));
    if (follow && frame().action) selected = { line: frame().action.line };
    if (step === last) pause();
    render();
  }
  function pause() {
    if (playing) clearInterval(playing);
    playing = null;
    $("play").textContent = "播放";
    $("play").setAttribute("aria-label", "播放");
  }
  function play() {
    if (playing) {
      pause();
      return;
    }
    if (step === last) moveTo(0);
    playing = setInterval(() => moveTo(step + 1), Number($("speed").value));
    $("play").textContent = "暂停";
    $("play").setAttribute("aria-label", "暂停");
  }
  function pointAt(points, t) {
    const lengths = points
      .slice(1)
      .map((p, i) => Math.hypot(p[0] - points[i][0], p[1] - points[i][1]));
    let distance = lengths.reduce((a, b) => a + b, 0) * t;
    for (let i = 0; i < lengths.length; i++) {
      if (distance <= lengths[i] || i === lengths.length - 1) {
        const q = lengths[i] ? distance / lengths[i] : 0;
        return [
          points[i][0] + q * (points[i + 1][0] - points[i][0]),
          points[i][1] + q * (points[i + 1][1] - points[i][1]),
        ];
      }
      distance -= lengths[i];
    }
    return points[0];
  }
  function renderMap() {
    const f = frame(),
      a = f.action;
    const selectedLine = selected?.car
      ? location(selected.car)[0]
      : selected?.line;
    let svg = "";
    for (const [name, t] of Object.entries(D.tracks)) {
      if (!t.points) continue;
      const ids = carsAt(f, name),
        active = a?.line === name,
        route = a?.route.includes(name),
        sel = selectedLine === name;
      const pts = t.points.map((p) => p.join(",")).join(" "),
        label = t.labelAnchor || pointAt(t.points, 0.5);
      svg += `<g data-line="${esc(name)}"><polyline class="track-path ${ids.length ? "occupied" : ""} ${route ? "route" : ""} ${sel ? "selected" : ""} ${active ? "active" : ""}" points="${pts}"/><polyline class="track-hit" points="${pts}"><title>${esc(name)} · ${ids.length} 辆</title></polyline><text class="track-label ${t.kind === "transit" ? "transit" : ""}" x="${label[0]}" y="${label[1] - 3}" text-anchor="middle">${esc(name)}${ids.length ? " · " + ids.length : ""}</text>`;
      ids.forEach((id, i) => {
        const p = pointAt(
          t.points,
          0.06 + ((i + 0.5) / Math.max(ids.length, 5)) * 0.87,
        );
        svg += `<circle class="car-marker ${a?.vehicles.includes(id) ? "changed" : ""} ${car(id).protected ? "protected" : ""}" cx="${p[0]}" cy="${p[1]}" r="${ids.length > 25 ? 4 : 6}" data-car="${esc(id)}"><title>${esc(id + " · " + targets(id))}</title></circle>`;
      });
      svg += "</g>";
    }
    const locoTrack = track(f.loco_line);
    if (locoTrack.points) {
      const p = pointAt(locoTrack.points, f.loco_end === "South" ? 0.96 : 0.04);
      svg += `<g transform="translate(${p[0]},${p[1] + 24})"><rect class="loco-marker" x="-21" y="-11" width="42" height="23" rx="5"/><text class="loco-text" text-anchor="middle" y="6">机</text></g>`;
    }
    $("yard-svg").innerHTML = svg;
    $("yard-svg").setAttribute("viewBox", box.join(" "));
    const unmapped = Object.keys(f.stacks).filter(
      (k) => !track(k).points && f.stacks[k].length,
    );
    $("unmapped").innerHTML = unmapped.length
      ? "未配置图形的股道：" +
        unmapped
          .map(
            (k) =>
              `<button data-line="${esc(k)}">${esc(k)} · ${f.stacks[k].length}</button>`,
          )
          .join("")
      : "";
  }
  function renderTable() {
    const f = frame(),
      a = f.action,
      only = $("occupied-only").checked;
    const names = Object.keys(D.tracks)
      .filter((n) => track(n).kind !== "transit" || carsAt(f, n).length)
      .filter((n) => !only || carsAt(f, n).length || a?.line === n);
    $("track-table").innerHTML =
      names
        .map((name) => {
          const ids = carsAt(f, name),
            prev = carsAt(previous(), name);
          return `<div class="track-row ${a?.line === name ? "active" : ""}"><button class="track-name" data-line="${esc(name)}">${esc(name)}<small>${ids.length} 辆 · ${metres(length(ids))}</small></button><div class="chips">${chips(ids, { added: ids.filter((c) => !prev.includes(c)) })}</div></div>`;
        })
        .join("") || '<div class="empty">当前没有停留车辆</div>';
  }
  function carRows(ids, added = [], removed = []) {
    return ids.length
      ? `<div class="car-list">${ids.map((id, i) => `<button class="car-row ${added.includes(id) ? "added" : ""} ${removed.includes(id) ? "removed" : ""}" data-car="${esc(id)}"><span class="ordinal">${i + 1}</span><span class="car-content">${esc(id)}<span class="target">→ ${esc(targets(id))}</span></span>${car(id).protected ? '<span class="lock">保护</span>' : ""}</button>`).join("")}</div>`
      : '<div class="empty">空股道</div>';
  }
  function renderDetails() {
    const f = frame(),
      a = f.action;
    if (!selected) {
      $("selection-kind").textContent = "";
      $("details").innerHTML =
        '<div class="empty">点击股道或车辆查看详情</div>';
      return;
    }
    if (selected.car) {
      const id = selected.car,
        c = car(id),
        [line, pos] = location(id),
        bool = (v) => (v == null ? "未知" : v ? "是" : "否");
      $("selection-kind").textContent = "车辆";
      $("details").innerHTML =
        `<button class="text-button" data-line="${esc(line)}">← ${esc(lineName(line))}</button><h2 style="margin-top:14px">${esc(id)}</h2>${c.protected ? '<span class="lock">保护车 · 不能移动</span>' : ""}<dl class="kv"><dt>当前位置</dt><dd>${esc(lineName(line))}${pos ? " · 第 " + pos + " 辆" : ""}</dd><dt>车型 / 修程</dt><dd>${esc(c.type || "未知")} / ${esc(c.repair || "未知")}</dd><dt>长度</dt><dd>${metres(c.length)}</dd><dt>属性</dt><dd>重车 ${bool(c.heavy)} · 称重 ${bool(c.weigh)}<br>关门车 ${bool(c.closed)}</dd><dt>初始位置</dt><dd>${esc(lineName(c.initial_line || "未知"))}${c.initial_position ? " · " + c.initial_position : ""}</dd></dl><h3>目标 · 任一股道可选</h3>${
          Object.entries(c.targets || {})
            .map(
              ([name, t]) =>
                `<div class="target-card"><button class="text-button" data-line="${esc(name)}">${esc(name)}</button><br><span class="muted">${t.ForceTargetPosition?.length ? "顺序位置 " + esc(t.ForceTargetPosition.join(" / ")) : "不限制顺序位置"}</span></div>`,
            )
            .join("") ||
          '<p class="hint">此文件未提供目标信息。可搭配原始请求补全。</p>'
        }${Object.values(c.targets || {}).some((t) => t.ForceTargetPosition?.length) ? '<p class="hint">位置表示允许空位平移的前后关系，不要求等于当前连续序号。</p>' : ""}<details style="margin-top:20px"><summary>原始车辆字段</summary><pre class="raw">${esc(JSON.stringify(c.raw || {}, null, 2))}</pre></details>`;
      return;
    }
    const name = selected.line,
      t = track(name),
      ids = carsAt(f, name),
      prev = carsAt(previous(), name),
      len = length(ids),
      capacity = t.length_mm ? t.length_mm / 1000 : null,
      changed = JSON.stringify(ids) !== JSON.stringify(prev);
    $("selection-kind").textContent = kindName(t.kind);
    $("details").innerHTML =
      `<h2>${esc(lineName(name))}</h2>${t.protected_no_buffer ? '<p class="lock">保护库内 · 禁止临时缓存</p>' : ""}<div class="statline"><span>${ids.length} 辆</span><span>${metres(len)}${capacity ? " / " + capacity + " m" : ""}</span></div>${capacity && len != null ? `<div class="meter ${len > capacity ? "over" : ""}"><i style="width:${Math.min(100, (len / capacity) * 100)}%"></i></div><p class="hint">${len > capacity ? "超出临停长度 " + (len - capacity).toFixed(1) + " m" : "长度余量 " + Math.max(0, capacity - len).toFixed(1) + " m"}${t.terminal_length_mm && t.terminal_length_mm !== t.length_mm ? " · 终停范围 " + t.terminal_length_mm / 1000 + " m" : ""}</p>` : ""}<h3>${name === "__loco__" ? "靠近机车 → 远离机车" : "北端 → 南端"}${changed ? " · 本步之后" : ""}</h3>${carRows(
        ids,
        ids.filter((c) => !prev.includes(c)),
      )}${
        changed
          ? `<details style="margin-top:15px"><summary>查看本步之前 · ${prev.length} 辆</summary><div style="margin-top:8px">${carRows(
              prev,
              [],
              prev.filter((c) => !ids.includes(c)),
            )}</div></details>`
          : ""
      }${a?.line === name ? `<h3>本步记录</h3><p class="hint">${f.source === "snapshot" ? "状态来自计划快照" : "状态按取放记录重建"}</p>${a.route.length ? `<div class="route-text">${a.route.map(esc).join(" → ")}</div>` : '<p class="hint">此步未提供进路。</p>'}` : ""}`;
  }
  function renderPlan() {
    if (!D.has_plan) {
      $("plan-tab").hidden = true;
      return;
    }
    $("plan-list").innerHTML = frames
      .map(
        (f, i) =>
          `<button class="plan-item ${step === i ? "current" : ""}" data-step="${i}" aria-current="${step === i ? "step" : "false"}"><span class="plan-number">${i === 0 ? "始" : f.hook}</span><span class="plan-copy">${esc(i === 0 ? "初始状态" : f.action.line + " " + opName(f.action.operation) + (f.action.count ? " " + f.action.count + " 辆" : ""))}<small>${i === 0 ? "请求 / 初始快照" : f.action.vehicles.length ? esc(f.action.vehicles.slice(0, 2).join("、")) + (f.action.vehicles.length > 2 ? " 等" : "") : "车辆未记录"}</small></span>${f.issues.length ? '<span class="issue-mark" title="有数据提示">!</span>' : ""}</button>`,
      )
      .join("");
    if (D.stopped)
      $("plan-list").innerHTML +=
        '<p class="hint">后续步骤无法重建，详见数据提示。</p>';
  }
  function renderTrain() {
    const f = frame(),
      ids = f.train,
      prev = previous().train,
      changed = JSON.stringify(ids) !== JSON.stringify(prev),
      len = length(ids),
      total = len == null ? null : len + 15;
    $("loco-location").textContent =
      `${f.loco_line} · ${f.loco_end === "South" ? "南端" : "北端"}`;
    $("train-stats").textContent =
      `${ids.length} / 20 辆 · ${total == null ? "总长未知" : total.toFixed(1) + " / 193 m（含机车）"}`;
    $("train-stats").className =
      ids.length > 20 || (total != null && total > 193) ? "over" : "";
    const row = (list, before) =>
      `<div class="train-row ${before ? "before" : ""}">${changed ? `<span class="train-row-label">${before ? "之前" : "之后"}</span>` : ""}<span class="engine">机车</span><div class="chips">${chips(list, { added: before ? [] : ids.filter((c) => !prev.includes(c)), removed: before ? prev.filter((c) => !ids.includes(c)) : [] })}</div></div>`;
    $("train-body").innerHTML =
      (changed ? row(prev, true) : "") + row(ids, false);
  }
  function updatePanels() {
    $("map-view").hidden = mode !== "map";
    $("table-view").hidden = mode !== "table";
    for (const v of ["map", "table"]) {
      $(v + "-tab").classList.toggle("active", mode === v);
      $(v + "-tab").setAttribute("aria-pressed", String(mode === v));
    }
    $("details").hidden = panel !== "details";
    $("plan-list").hidden = panel !== "plan";
    $("detail-tab").classList.toggle("active", panel === "details");
    $("plan-tab").classList.toggle("active", panel === "plan");
  }
  function render() {
    const f = frame();
    $("scrubber").value = step;
    $("action-title").textContent = actionTitle(f, step);
    $("step-count").textContent = `${step} / ${D.total_steps} 步`;
    $("first").disabled = $("prev").disabled = step === 0;
    $("next").disabled = $("last").disabled = step === last;
    $("step-notice").innerHTML = notice(f.issues);
    renderMap();
    renderTable();
    renderDetails();
    renderPlan();
    renderTrain();
    if (panel === "plan")
      $("plan-list")
        .querySelector(".current")
        ?.scrollIntoView({ block: "nearest" });
  }
  function search() {
    const q = $("search").value.trim().toLowerCase(),
      out = $("search-results");
    if (!q) {
      out.hidden = true;
      return;
    }
    let matches = [];
    for (const name of Object.keys(D.tracks)) {
      if (name.toLowerCase().includes(q)) {
        matches.push(
          `<button data-line="${esc(name)}">${esc(name)}<small>股道 · 当前 ${carsAt(frame(), name).length} 辆</small></button>`,
        );
      }
    }
    for (const id of Object.keys(D.vehicles)) {
      if (id.toLowerCase().includes(q)) {
        const loc = location(id);
        matches.push(
          `<button data-car="${esc(id)}">${esc(id)}<small>${esc(lineName(loc[0]))}${loc[1] ? " · 第 " + loc[1] + " 辆" : ""}</small></button>`,
        );
      }
    }
    out.innerHTML =
      matches.slice(0, 30).join("") ||
      '<div class="empty">没有匹配的车号或股道</div>';
    out.hidden = false;
  }
  function zoom(factor) {
    const [x, y, w, h] = box;
    const nw = Math.max(300, Math.min(4300, w * factor)),
      nh = (nw * 1040) / 2160;
    box = [x + (w - nw) / 2, y + (h - nh) / 2, nw, nh];
    $("yard-svg").setAttribute("viewBox", box.join(" "));
  }
  function init() {
    const statuses = {
      complete: ["完整计划", ""],
      search_limit: ["部分计划", "warn"],
      cycle_detected: ["循环停步", "warn"],
      invalid_request: ["请求未通过校验", "warn"],
      dead_end: ["部分计划", "warn"],
    };
    const s = D.stopped
      ? ["回放中断", "warn"]
      : D.has_plan
        ? statuses[D.status] || [
            D.status ? "计划 · " + D.status : "勾计划",
            "neutral",
          ]
        : ["调车请求", "neutral"];
    $("status").textContent = s[0];
    $("status").className = "badge " + s[1];
    $("source").textContent =
      `${D.title} · ${Object.keys(D.vehicles).length} 辆车${D.has_plan ? " · " + D.total_hooks + " 勾" + (D.total_steps !== D.total_hooks ? " / " + D.total_steps + " 步" : "") : ""}`;
    $("replay-note").textContent = D.replay_note || "";
    $("notices").innerHTML = notice(D.issues);
    if (!frames.length) {
      document.querySelector(".workspace").hidden = true;
      document.querySelector(".train").hidden = true;
      $("playback").hidden = true;
      return;
    }
    selected = { line: frame().loco_line };
    if (Object.values(frame().stacks).some((ids) => ids.length))
      selected = {
        line: Object.keys(frame().stacks).find((k) => frame().stacks[k].length),
      };
    $("scrubber").max = last;
    $("playback").hidden = !D.has_plan;
    $("play").disabled = last === 0;
    $("scrubber").disabled = last === 0;
    $("first").onclick = () => {
      pause();
      moveTo(0);
    };
    $("last").onclick = () => {
      pause();
      moveTo(last);
    };
    $("prev").onclick = () => {
      pause();
      moveTo(step - 1);
    };
    $("next").onclick = () => {
      pause();
      moveTo(step + 1);
    };
    $("play").onclick = play;
    $("scrubber").oninput = (e) => {
      pause();
      moveTo(Number(e.target.value));
    };
    $("speed").onchange = () => {
      if (playing) {
        pause();
        play();
      }
    };
    $("follow").onchange = (e) => {
      follow = e.target.checked;
      if (follow) {
        selected = { line: frame().action?.line || frame().loco_line };
        render();
      }
    };
    $("map-tab").onclick = () => {
      mode = "map";
      updatePanels();
    };
    $("table-tab").onclick = () => {
      mode = "table";
      updatePanels();
    };
    $("detail-tab").onclick = () => {
      panel = "details";
      updatePanels();
    };
    $("plan-tab").onclick = () => {
      panel = "plan";
      updatePanels();
      $("plan-list")
        .querySelector(".current")
        ?.scrollIntoView({ block: "nearest" });
    };
    $("occupied-only").onchange = renderTable;
    $("search").oninput = search;
    $("search").onkeydown = (e) => {
      if (e.key === "Escape") $("search-results").hidden = true;
      if (e.key === "Enter")
        $("search-results").querySelector("button")?.click();
    };
    document.addEventListener("click", (e) => {
      if (!e.target.closest(".search-wrap")) $("search-results").hidden = true;
      const el = e.target.closest("[data-car],[data-line],[data-step]");
      if (!el || dragged) {
        dragged = false;
        return;
      }
      if (el.dataset.step != null) {
        pause();
        moveTo(Number(el.dataset.step));
      } else if (el.dataset.car) {
        setSelected({ car: el.dataset.car });
      } else setSelected({ line: el.dataset.line });
      $("search-results").hidden = true;
    });
    document.addEventListener("keydown", (e) => {
      if (
        ["INPUT", "SELECT", "TEXTAREA", "BUTTON", "SUMMARY"].includes(
          e.target.tagName,
        ) ||
        !D.has_plan
      )
        return;
      if (e.key === "ArrowRight") {
        pause();
        moveTo(step + 1);
      } else if (e.key === "ArrowLeft") {
        pause();
        moveTo(step - 1);
      } else if (e.key === "Home") {
        pause();
        moveTo(0);
      } else if (e.key === "End") {
        pause();
        moveTo(last);
      } else if (e.code === "Space") {
        e.preventDefault();
        if (last) play();
      }
    });
    $("zoom-in").onclick = () => zoom(0.8);
    $("zoom-out").onclick = () => zoom(1.25);
    $("reset").onclick = () => {
      box = [40, 45, 2160, 1040];
      renderMap();
    };
    const svg = $("yard-svg");
    svg.addEventListener(
      "wheel",
      (e) => {
        if (e.ctrlKey || e.metaKey) {
          e.preventDefault();
          zoom(e.deltaY > 0 ? 1.12 : 0.88);
        }
      },
      { passive: false },
    );
    svg.addEventListener("pointerdown", (e) => {
      drag = { x: e.clientX, y: e.clientY, box: [...box] };
      dragged = false;
    });
    svg.addEventListener("pointermove", (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x,
        dy = e.clientY - drag.y;
      if (Math.hypot(dx, dy) > 4) {
        dragged = true;
        svg.classList.add("dragging");
        svg.setPointerCapture(e.pointerId);
        const scale = Math.min(
          svg.clientWidth / box[2],
          svg.clientHeight / box[3],
        );
        box = [
          drag.box[0] - dx / scale,
          drag.box[1] - dy / scale,
          drag.box[2],
          drag.box[3],
        ];
        svg.setAttribute("viewBox", box.join(" "));
      }
    });
    svg.addEventListener("pointerup", () => {
      drag = null;
      svg.classList.remove("dragging");
    });
    svg.addEventListener("pointercancel", () => {
      drag = null;
      dragged = false;
      svg.classList.remove("dragging");
    });
    updatePanels();
    render();
  }
  init();
})();

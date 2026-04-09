import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

let pollInterval = 500;
const FADE_TICKS = 6;
const GRAPH_POINTS = 120;

const execState = { running: false, node: null, progress: null };
let peakVramUsed = 0;

// color palette — dark chrome, colored data
const C = {
    vram:       "#e67e22",
    torch:      "#2ecc71",
    pinned:     "#4a9eff",
    unloaded:   "#3a3a3a",
    other:      "#505050",
    text:       "#b0b0b0",
    textDim:    "#707070",
    running:    "#b0b0b0",
    bg:         "#181818",
    headerBg:   "#202020",
    border:     "#2a2a2a",
    btn:        "#2a2a2a",
    btnText:    "#888",
    graphBg:    "#0e0e0e",
    gridLine:   "#1e1e1e",
    totalLine:  "#d0d0d0",
    capLine:    "#555",
    barBg:      "#222",
    fadeInFrom:  [255, 220, 0],
    fadeInTo:    [230, 126, 34],
    fadeOutFrom: [200, 60, 60],
    fadeOutTo:   [58, 58, 58],
};


function escHtml(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

function formatBytes(bytes) {
    if (bytes == null) return "?";
    if (bytes >= 1024 ** 3) return (bytes / 1024 ** 3).toFixed(1) + " GB";
    if (bytes >= 1024 ** 2) return (bytes / 1024 ** 2).toFixed(0) + " MB";
    return (bytes / 1024).toFixed(0) + " KB";
}

// rolling history — ring buffer to avoid shift()
const history = {
    torch_active: new Float64Array(GRAPH_POINTS),
    aimdo_usage: new Float64Array(GRAPH_POINTS),
    free_vram: new Float64Array(GRAPH_POINTS),
    total_vram: 1,
    head: 0,
    len: 0,
};

function pushHistory(data) {
    history.total_vram = data.total_vram;
    const i = history.head;
    history.torch_active[i] = data.torch_active;
    history.aimdo_usage[i] = data.aimdo_usage;
    history.free_vram[i] = data.free_vram;
    history.head = (i + 1) % GRAPH_POINTS;
    if (history.len < GRAPH_POINTS) history.len++;
}

function historyGet(arr, idx) {
    // idx 0 = oldest, idx len-1 = newest
    return arr[(history.head - history.len + idx + GRAPH_POINTS) % GRAPH_POINTS];
}

function drawGraph(ctx, w, h) {
    const total = history.total_vram;
    const len = history.len;
    if (len < 2) return;

    ctx.clearRect(0, 0, w, h);

    ctx.strokeStyle = C.gridLine;
    ctx.lineWidth = 1;
    for (const pct of [0.25, 0.5, 0.75]) {
        const y = Math.round(h - h * pct) + 0.5;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
    }

    const stepX = w / (GRAPH_POINTS - 1);
    const yFor = val => h - (val / total) * h;

    // aimdo area
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < len; i++) {
        ctx.lineTo((GRAPH_POINTS - len + i) * stepX, yFor(historyGet(history.aimdo_usage, i)));
    }
    ctx.lineTo((GRAPH_POINTS - 1) * stepX, h);
    ctx.closePath();
    ctx.fillStyle = "rgba(230,126,34,0.35)";
    ctx.fill();

    // torch area stacked
    ctx.beginPath();
    ctx.moveTo(0, h);
    for (let i = 0; i < len; i++) {
        const x = (GRAPH_POINTS - len + i) * stepX;
        ctx.lineTo(x, yFor(historyGet(history.aimdo_usage, i) + historyGet(history.torch_active, i)));
    }
    for (let i = len - 1; i >= 0; i--) {
        ctx.lineTo((GRAPH_POINTS - len + i) * stepX, yFor(historyGet(history.aimdo_usage, i)));
    }
    ctx.closePath();
    ctx.fillStyle = "rgba(46,204,113,0.4)";
    ctx.fill();

    // total used line
    ctx.beginPath();
    ctx.strokeStyle = C.totalLine;
    ctx.lineWidth = 1.5;
    for (let i = 0; i < len; i++) {
        const x = (GRAPH_POINTS - len + i) * stepX;
        const y = yFor(total - historyGet(history.free_vram, i));
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // capacity line
    ctx.strokeStyle = C.capLine;
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(0, yFor(total));
    ctx.lineTo(w, yFor(total));
    ctx.stroke();
    ctx.setLineDash([]);
}

// per-model residency diff state
const modelState = {};

function diffResidency(key, residency) {
    let st = modelState[key];
    if (!st || st.prev.length !== residency.length) {
        st = { prev: new Uint8Array(residency), changeAge: new Uint8Array(residency.length) };
        modelState[key] = st;
        return st;
    }

    for (let i = 0; i < residency.length; i++) {
        if (residency[i] !== st.prev[i]) {
            st.changeAge[i] = FADE_TICKS;
        } else if (st.changeAge[i] > 0) {
            st.changeAge[i]--;
        }
        st.prev[i] = residency[i];
    }
    return st;
}

// draw page grid to canvas — much faster than 700 DOM divs
function drawPageGrid(ctx, w, residency, changeAge) {
    const cellSize = 6;
    const gap = 1;
    const step = cellSize + gap;
    const cols = Math.floor((w + gap) / step);
    const rows = Math.ceil(residency.length / cols);
    const h = rows * step;

    ctx.canvas.height = h || 1;
    ctx.canvas.style.height = (h || 1) + "px";
    ctx.clearRect(0, 0, w, h);

    // batch: draw all static vram cells, then all static unloaded, then animated individually
    const animated = [];

    ctx.fillStyle = C.vram;
    for (let i = 0; i < residency.length; i++) {
        if (changeAge[i] > 0) { animated.push(i); continue; }
        if (!(residency[i] & 1)) continue;
        ctx.fillRect((i % cols) * step, Math.floor(i / cols) * step, cellSize, cellSize);
    }

    ctx.fillStyle = C.unloaded;
    for (let i = 0; i < residency.length; i++) {
        if (changeAge[i] > 0 || (residency[i] & 1)) continue;
        ctx.fillRect((i % cols) * step, Math.floor(i / cols) * step, cellSize, cellSize);
    }

    // animated cells need individual colors
    for (const i of animated) {
        const resident = residency[i] & 1;
        const t = changeAge[i] / FADE_TICKS;
        const [fr, fg, fb] = resident ? C.fadeInFrom : C.fadeOutFrom;
        const [tr, tg, tb] = resident ? C.fadeInTo : C.fadeOutTo;
        ctx.fillStyle = `rgb(${Math.round(fr * t + tr * (1 - t))},${Math.round(fg * t + tg * (1 - t))},${Math.round(fb * t + tb * (1 - t))})`;
        ctx.fillRect((i % cols) * step, Math.floor(i / cols) * step, cellSize, cellSize);
    }
}

function createPanel() {
    const panel = document.createElement("div");
    panel.id = "aimdo-viz-panel";
    panel.style.cssText = `
        position: fixed; bottom: 10px; right: 10px;
        background: ${C.bg}; color: ${C.text};
        border: 1px solid ${C.border}; border-radius: 8px;
        padding: 0; font-family: monospace; font-size: 12px;
        z-index: 10000; min-width: 280px; width: 340px; max-height: 90vh;
        box-shadow: 0 4px 12px rgba(0,0,0,0.7);
        user-select: none; resize: horizontal; overflow-y: auto;
    `;

    const header = document.createElement("div");
    header.style.cssText = `
        display: flex; justify-content: space-between; align-items: center;
        padding: 6px 10px; background: ${C.headerBg};
        border-radius: 8px 8px 0 0; cursor: move;
    `;
    const titleSpan = document.createElement("span");
    titleSpan.style.cssText = `font-weight:bold;color:${C.text};`;
    titleSpan.textContent = "VRAM";
    header.appendChild(titleSpan);

    const headerRight = document.createElement("div");
    headerRight.style.cssText = "display:flex;align-items:center;gap:6px;";

    const intervalSelect = document.createElement("select");
    intervalSelect.style.cssText = `font-size:9px;background:${C.btn};color:${C.btnText};border:none;border-radius:2px;padding:1px 2px;cursor:pointer;`;
    for (const ms of [100, 250, 500, 1000, 2000, 5000]) {
        const opt = document.createElement("option");
        opt.value = ms;
        opt.textContent = ms < 1000 ? `${ms}ms` : `${ms/1000}s`;
        if (ms === pollInterval) opt.selected = true;
        intervalSelect.appendChild(opt);
    }
    intervalSelect.addEventListener("change", () => { pollInterval = parseInt(intervalSelect.value); });

    const unloadBtn = document.createElement("span");
    unloadBtn.textContent = "unload";
    unloadBtn.style.cssText = `cursor:pointer;font-size:10px;padding:1px 6px;background:${C.btn};border-radius:3px;color:${C.btnText};`;
    unloadBtn.addEventListener("click", async (e) => {
        e.stopPropagation();
        unloadBtn.textContent = "...";
        try {
            await api.fetchApi("/aimdo/unload_all", { method: "POST" });
            peakVramUsed = 0;  // #7: reset peak on unload all
        } finally {
            unloadBtn.textContent = "unload";
        }
    });

    const toggleBtn = document.createElement("span");
    toggleBtn.textContent = "\u2212";
    toggleBtn.style.cssText = `cursor:pointer;font-size:16px;padding:0 4px;color:${C.btnText};`;

    const body = document.createElement("div");
    body.id = "aimdo-viz-body";
    body.style.cssText = "padding: 8px 10px;";

    let collapsed = false;
    toggleBtn.addEventListener("click", (e) => {
        e.stopPropagation();
        collapsed = !collapsed;
        body.style.display = collapsed ? "none" : "block";
        toggleBtn.textContent = collapsed ? "+" : "\u2212";
    });

    headerRight.appendChild(intervalSelect);
    headerRight.appendChild(unloadBtn);
    headerRight.appendChild(toggleBtn);
    header.appendChild(headerRight);
    panel.appendChild(header);
    panel.appendChild(body);

    let dragging = false, dx = 0, dy = 0;
    header.addEventListener("mousedown", (e) => {
        dragging = true;
        dx = e.clientX - panel.offsetLeft;
        dy = e.clientY - panel.offsetTop;
    });
    document.addEventListener("mousemove", (e) => {
        if (!dragging) return;
        panel.style.left = (e.clientX - dx) + "px";
        panel.style.top = (e.clientY - dy) + "px";
        panel.style.right = "auto";
        panel.style.bottom = "auto";
    });
    document.addEventListener("mouseup", () => { dragging = false; });

    document.body.appendChild(panel);
    body._titleSpan = titleSpan;
    return body;
}

// persistent DOM refs to avoid re-querying / re-creating
let refs = null;

function ensureStructure(body) {
    if (refs) return refs;

    body.innerHTML = "";

    const contentDiv = document.createElement("div");
    contentDiv.id = "aimdo-content";
    body.appendChild(contentDiv);

    const graphCanvas = document.createElement("canvas");
    graphCanvas.width = 300;
    graphCanvas.height = 80;
    graphCanvas.style.cssText = `width:100%;height:80px;border-radius:3px;background:${C.graphBg};`;
    body.appendChild(graphCanvas);

    const modelsDiv = document.createElement("div");
    modelsDiv.id = "aimdo-models";
    body.appendChild(modelsDiv);

    refs = {
        contentDiv,
        graphCanvas,
        graphCtx: graphCanvas.getContext("2d"),
        modelsDiv,
        pageCanvases: {},   // keyed by model name
        pageCtxs: {},
    };
    return refs;
}

function renderData(body, data) {
    if (!data.enabled) {
        body.innerHTML = `<div style="color:${C.textDim};">not available</div>`;
        refs = null;
        return;
    }

    const r = ensureStructure(body);
    body._titleSpan.textContent = data.aimdo_active ? "VRAM (aimdo)" : "VRAM";
    pushHistory(data);

    const used = data.total_vram - data.free_vram;
    if (used > peakVramUsed) peakVramUsed = used;
    const aimdoPct = data.total_vram > 0 ? (data.aimdo_usage / data.total_vram * 100).toFixed(0) : 0;
    const torchPct = data.total_vram > 0 ? (data.torch_active / data.total_vram * 100).toFixed(0) : 0;
    const otherUsed = Math.max(0, used - data.aimdo_usage - data.torch_active);
    const otherPct = data.total_vram > 0 ? (otherUsed / data.total_vram * 100).toFixed(0) : 0;

    r.contentDiv.innerHTML = `<div style="margin-bottom:4px;">
        <div style="display:flex;justify-content:space-between;margin-bottom:2px;">
            <span>VRAM</span>
            <span>${formatBytes(used)} / ${formatBytes(data.total_vram)}</span>
        </div>
        <div style="background:${C.barBg};border-radius:3px;height:8px;overflow:hidden;display:flex;">
            <div style="background:${C.vram};height:100%;width:${aimdoPct}%;" title="aimdo: ${formatBytes(data.aimdo_usage)}"></div>
            <div style="background:${C.torch};height:100%;width:${torchPct}%;" title="torch: ${formatBytes(data.torch_active)}"></div>
            <div style="background:${C.other};height:100%;width:${otherPct}%;" title="other: ${formatBytes(otherUsed)}"></div>
        </div>
        <div style="display:flex;gap:8px;font-size:10px;color:${C.textDim};margin-top:2px;">
            ${data.aimdo_active ? `<span><span style="color:${C.vram};">&#9632;</span> aimdo ${formatBytes(data.aimdo_usage)}</span>` : ""}
            <span><span style="color:${C.torch};">&#9632;</span> torch ${formatBytes(data.torch_active)}</span>
            <span><span style="color:${C.other};">&#9632;</span> other ${formatBytes(otherUsed)}</span>
        </div>
        <div style="display:flex;gap:10px;font-size:10px;color:${C.textDim};margin-top:2px;">
            <span>peak: ${formatBytes(peakVramUsed)} <span class="aimdo-reset-peak-btn" style="cursor:pointer;font-size:9px;padding:0px 2px;background:${C.btn};border-radius:2px;color:${C.btnText};" title="reset peak">rst</span></span>
            <span>cache: ${formatBytes(data.torch_reserved - data.torch_active)}</span>
            ${execState.running ? `<span style="color:${C.running};">&#9679; ${execState.node || "running"}${execState.progress ? " " + execState.progress : ""}</span>` : `<span>&#9679; idle</span>`}
        </div>
    </div>`;

    // sync canvas resolution to display size
    const displayW = r.graphCanvas.clientWidth || 300;
    if (r.graphCanvas.width !== displayW) r.graphCanvas.width = displayW;
    drawGraph(r.graphCtx, r.graphCanvas.width, r.graphCanvas.height);

    // models section — only rebuild HTML for text parts, use canvas for page grids
    let modelsHtml = "";

    if (data.models.length === 0) {
        modelsHtml += `<div style="color:${C.textDim};margin-top:6px;">No models loaded</div>`;
    }

    for (const m of data.models) {
        modelsHtml += `<div style="margin-top:6px;">
            <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:2px;">
                <span>${escHtml(m.name)}${m.dynamic ? "" : " (static)"}</span>
                <span style="display:flex;align-items:center;gap:6px;">
                    <span>${formatBytes(m.total_size)}</span>
                    ${m.dynamic && data.aimdo_active ? `<span class="aimdo-reset-wm-btn" data-index="${m.index}" style="cursor:pointer;font-size:9px;padding:0px 4px;background:${C.btn};border-radius:2px;color:${C.btnText};" title="reset watermark">wm</span>` : ""}
                    <span class="aimdo-unload-btn" data-index="${m.index}" style="cursor:pointer;font-size:9px;padding:0px 4px;background:${C.btn};border-radius:2px;color:${C.btnText};">x</span>
                </span>
            </div>`;

        if (m.dynamic) {
            const pinnedRam = m.pinned_ram || 0;
            const unloadedSize = Math.max(0, m.total_size - m.vbar_loaded - pinnedRam);
            const vramPct = m.total_size > 0 ? (m.vbar_loaded / m.total_size * 100) : 0;
            const pinnedPct = m.total_size > 0 ? (pinnedRam / m.total_size * 100) : 0;
            const unloadedPct = m.total_size > 0 ? (unloadedSize / m.total_size * 100) : 0;

            modelsHtml += `<div style="background:${C.barBg};border-radius:3px;height:10px;overflow:hidden;display:flex;">
                <div style="background:${C.vram};height:100%;width:${vramPct}%;" title="VRAM: ${formatBytes(m.vbar_loaded)}"></div>
                <div style="background:${C.pinned};height:100%;width:${pinnedPct}%;" title="pinned RAM: ${formatBytes(pinnedRam)}"></div>
                <div style="background:${C.unloaded};height:100%;width:${unloadedPct}%;" title="unloaded: ${formatBytes(unloadedSize)}"></div>
            </div>
            <div style="display:flex;gap:8px;font-size:10px;color:${C.textDim};margin-top:2px;">
                <span><span style="color:${C.vram};">&#9632;</span> VRAM ${formatBytes(m.vbar_loaded)}</span>
                ${pinnedRam > 0 ? `<span><span style="color:${C.pinned};">&#9632;</span> pinned ${formatBytes(pinnedRam)}</span>` : ""}
                <span><span style="color:${C.unloaded};">&#9632;</span> unloaded ${formatBytes(unloadedSize)}</span>
            </div>`;
        } else {
            const loadPct = m.total_size > 0 ? (m.loaded_size / m.total_size * 100).toFixed(0) : 0;
            modelsHtml += `<div style="background:${C.barBg};border-radius:3px;height:10px;overflow:hidden;">
                <div style="background:${C.vram};height:100%;width:${loadPct}%;"></div>
            </div>
            <div style="font-size:10px;color:${C.textDim};margin-top:2px;">
                <span style="color:${C.vram};">&#9632;</span> VRAM ${formatBytes(m.loaded_size)}
            </div>`;
        }

        if (m.vbars) {
            for (let vi = 0; vi < m.vbars.length; vi++) {
                const vb = m.vbars[vi];
                if (!vb.residency || vb.residency.length === 0) continue;

                const vkey = `${m.name}_${vb.device}_${vi}`;  // #8: stable key — no index shift leak
                diffResidency(vkey, vb.residency);
                let residentCount = 0, pinnedCount = 0;
                for (let i = 0; i < vb.residency.length; i++) {
                    const flag = vb.residency[i];
                    if (flag & 2) pinnedCount++;
                    else if (flag & 1) residentCount++;
                }
                const PAGE = 32 * 1024 * 1024;
                const vramPages = residentCount + pinnedCount;
                const ramPages = vb.residency.length - vramPages;

                if (m.vbars.length > 1) {
                    modelsHtml += `<div style="font-size:10px;color:${C.textDim};margin-top:3px;">${escHtml(vb.device)}</div>`;
                }
                modelsHtml += `<div data-pgrid="${escHtml(vkey)}" style="margin-top:2px;"></div>`;
                modelsHtml += `<div style="color:${C.textDim};font-size:10px;margin-top:2px;">
                    <span style="color:${C.vram};">${vramPages} VRAM (${formatBytes(vramPages * PAGE)})</span> + <span style="color:${C.unloaded};">${ramPages} unloaded (${formatBytes(ramPages * PAGE)})</span>
                </div>`;
            }
        }

        modelsHtml += `</div>`;
    }

    modelsHtml += `<div style="display:flex;flex-wrap:wrap;gap:8px;font-size:10px;color:${C.textDim};margin-top:6px;border-top:1px solid ${C.border};padding-top:4px;">
        <span><span style="color:${C.vram};">&#9632;</span> VRAM</span>
        <span><span style="color:${C.pinned};">&#9632;</span> pinned</span>
        <span><span style="color:${C.unloaded};">&#9632;</span> unloaded</span>
        <span><span style="color:${C.torch};">&#9632;</span> torch</span>
        <span><span style="color:${C.totalLine};">&#9472;</span> total used</span>
    </div>`;

    r.modelsDiv.innerHTML = modelsHtml;

    // collect active vbar keys for cleanup
    const activeKeys = new Set();
    for (const m of data.models) {
        if (!m.vbars) continue;
        for (let vi = 0; vi < m.vbars.length; vi++) {
            activeKeys.add(`${m.name}_${m.vbars[vi].device}_${vi}`);
        }
    }

    // clean up stale refs for models no longer present
    for (const key of Object.keys(r.pageCanvases)) {
        if (!activeKeys.has(key)) {
            delete r.pageCanvases[key];
            delete r.pageCtxs[key];
            delete modelState[key];
        }
    }

    // draw page grids into their placeholder divs using canvas
    for (const m of data.models) {
        if (!m.vbars) continue;
        for (let vi = 0; vi < m.vbars.length; vi++) {
            const vb = m.vbars[vi];
            if (!vb.residency || vb.residency.length === 0) continue;

            const vkey = `${m.name}_${vb.device}_${vi}`;
            const st = modelState[vkey];
            const container = r.modelsDiv.querySelector(`[data-pgrid="${vkey}"]`);  // #11: data-attr selector
            if (!container) continue;

            let canvas = r.pageCanvases[vkey];
            if (!canvas) {
                canvas = document.createElement("canvas");
                canvas.style.cssText = "width:100%;border-radius:2px;";
                r.pageCanvases[vkey] = canvas;
                r.pageCtxs[vkey] = canvas.getContext("2d");
            }
            const newW = container.clientWidth || r.modelsDiv.clientWidth || 300;
            if (canvas.width !== newW) canvas.width = newW;
            container.appendChild(canvas);
            drawPageGrid(r.pageCtxs[vkey], canvas.width, vb.residency, st ? st.changeAge : new Uint8Array(vb.residency.length));
        }
    }

    // attach button handlers via event delegation (once)
    if (!r.modelsDiv._delegated) {
        r.modelsDiv._delegated = true;
        r.modelsDiv.addEventListener("click", async (e) => {
            const peakBtn = e.target.closest(".aimdo-reset-peak-btn");
            if (peakBtn) {
                peakVramUsed = 0;  // #7: reset peak
                return;
            }
            const wmBtn = e.target.closest(".aimdo-reset-wm-btn");
            if (wmBtn) {
                const idx = parseInt(wmBtn.dataset.index);
                wmBtn.textContent = "...";
                try {
                    await api.fetchApi("/aimdo/reset_watermark", {
                        method: "POST",
                        body: JSON.stringify({ index: idx }),
                        headers: { "Content-Type": "application/json" },
                    });
                } finally {
                    wmBtn.textContent = "wm";
                }
                return;
            }
            const unloadBtn = e.target.closest(".aimdo-unload-btn");
            if (unloadBtn) {
                const idx = parseInt(unloadBtn.dataset.index);
                unloadBtn.textContent = "...";
                try {
                    await api.fetchApi("/aimdo/unload_model", {
                        method: "POST",
                        body: JSON.stringify({ index: idx }),
                        headers: { "Content-Type": "application/json" },
                    });
                } catch { /* next poll will reflect state */ }
            }
        });
    }
}

app.registerExtension({
    name: "aimdo.VRAMVisualization",
    async setup() {
        const body = createPanel();

        api.addEventListener("execution_start", () => {
            execState.running = true;
            execState.node = null;
            execState.progress = null;
        });
        api.addEventListener("executing", ({ detail }) => {
            execState.running = detail != null;
            execState.node = null;
            execState.progress = null;
        });
        api.addEventListener("progress", ({ detail }) => {
            if (detail) {
                execState.progress = `${detail.value}/${detail.max}`;
            }
        });

        async function poll() {
            try {
                const resp = await api.fetchApi("/aimdo/vram");
                const data = await resp.json();
                renderData(body, data);
            } catch (e) {
                // #10: keep last good render, show transient error overlay
                if (!refs) {
                    body.innerHTML = `<div style="color:#aa5555;">Error fetching data</div>`;
                }
            }
            setTimeout(poll, pollInterval);
        }

        poll();
    }
});

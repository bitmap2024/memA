#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""记忆抽取流水线 —— 简易本地 Dashboard。

启动（项目根目录 memA/）::

    pip install prefect>=3.0.0
    python -m src.queue.prefect_dashboard

浏览器打开 http://127.0.0.1:8765

另开终端跑 flow，dashboard 会自动刷新显示步骤进度::

    python -m src.queue.prefect_memory_flow --dry-run
    python -m src.queue.prefect_memory_flow --dry-run --mode both

可选：同时启动 Prefect 官方 UI（更完整的 Flow/Task 视图）::

    python -m src.queue.prefect_dashboard --with-prefect-ui
    # Prefect UI → http://127.0.0.1:4200
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
import webbrowser
from typing import Any, Dict, List, Optional

try:
    from starlette.applications import Starlette
    from starlette.responses import HTMLResponse, JSONResponse
    from starlette.routing import Route
    import uvicorn
except ImportError as exc:
    raise ImportError(
        "请先安装 Prefect（自带 starlette/uvicorn）: pip install 'prefect>=3.0.0'"
    ) from exc

from src.queue.run_store import get_run_store

DASHBOARD_PORT = 8765
PREFECT_UI_URL = "http://127.0.0.1:4200"
PREFECT_API_URL = "http://127.0.0.1:4200/api"

DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>memA 流水线 Dashboard</title>
  <style>
    :root {
      --bg: #0f1419;
      --card: #1a2332;
      --border: #2d3a4d;
      --text: #e7ecf3;
      --muted: #8b9cb3;
      --ok: #3dd68c;
      --run: #5b9cf5;
      --fail: #f07178;
      --warn: #ebc06d;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: "Segoe UI", system-ui, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
      line-height: 1.5;
    }
    h1 { font-size: 1.4rem; margin-bottom: 4px; }
    .sub { color: var(--muted); font-size: 0.9rem; margin-bottom: 20px; }
    .links { margin-bottom: 20px; }
    .links a {
      color: var(--run);
      margin-right: 16px;
      text-decoration: none;
    }
    .links a:hover { text-decoration: underline; }
    .stats {
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
      margin-bottom: 20px;
    }
    .stat {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px 16px;
      min-width: 120px;
    }
    .stat b { display: block; font-size: 1.5rem; }
    .stat span { color: var(--muted); font-size: 0.85rem; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: var(--card);
      border-radius: 8px;
      overflow: hidden;
      border: 1px solid var(--border);
    }
    th, td {
      text-align: left;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      font-size: 0.88rem;
    }
    th { color: var(--muted); font-weight: 600; }
    tr:last-child td { border-bottom: none; }
    tr:hover { background: rgba(255,255,255,0.03); }
    .badge {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 999px;
      font-size: 0.75rem;
      font-weight: 600;
    }
    .badge.completed { background: rgba(61,214,140,0.2); color: var(--ok); }
    .badge.running { background: rgba(91,156,245,0.2); color: var(--run); }
    .badge.failed { background: rgba(240,113,120,0.2); color: var(--fail); }
    .badge.prefect { background: rgba(91,156,245,0.15); color: #7eb8ff; }
    .badge.task_queue { background: rgba(235,192,109,0.15); color: var(--warn); }
    .steps { display: flex; gap: 6px; flex-wrap: wrap; }
    .step {
      font-size: 0.72rem;
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid var(--border);
      color: var(--muted);
    }
    .step.completed { border-color: var(--ok); color: var(--ok); }
    .step.running { border-color: var(--run); color: var(--run); }
    .step.failed { border-color: var(--fail); color: var(--fail); }
    .empty { color: var(--muted); padding: 40px; text-align: center; }
    .refresh { color: var(--muted); font-size: 0.8rem; margin-top: 12px; }
  </style>
</head>
<body>
  <h1>memA 记忆抽取 Dashboard</h1>
  <p class="sub">本地运行记录 · 每 3 秒自动刷新</p>
  <div class="links" id="links"></div>
  <div class="stats" id="stats"></div>
  <div id="table-wrap"></div>
  <p class="refresh" id="refresh-hint">加载中…</p>
  <script>
    const STEP_ORDER = ["1-validate","2-compress","3-topic-segment","4-llm-extract","5-persist"];

    function badge(cls, text) {
      return `<span class="badge ${cls}">${text}</span>`;
    }

    function renderSteps(steps) {
      const map = Object.fromEntries((steps || []).map(s => [s.name, s]));
      return `<div class="steps">${STEP_ORDER.map(name => {
        const s = map[name];
        const st = s ? s.status : "pending";
        const label = name.replace(/^\\d+-/, "");
        return `<span class="step ${st}">${label}</span>`;
      }).join("")}</div>`;
    }

    function renderReport(r) {
      if (!r || !r.report) return "-";
      const rep = r.report;
      const ext = rep.extracted || {};
      return `topics=${rep.topics ?? "-"} stored=${rep.stored ?? 0} ` +
        `(P${ext.profile||0}/E${ext.episodic||0}/S${ext.state||0})`;
    }

    async function load() {
      const res = await fetch("/api/runs");
      const data = await res.json();
      const runs = data.runs || [];
      const prefect = data.prefect_ui_reachable;

      document.getElementById("links").innerHTML =
        `<a href="/" target="_self">刷新</a>` +
        (prefect ? `<a href="${data.prefect_ui_url}" target="_blank">Prefect 官方 UI ↗</a>` : "") +
        `<a href="/api/runs" target="_blank">JSON API</a>`;

      const completed = runs.filter(r => r.status === "completed").length;
      const running = runs.filter(r => r.status === "running").length;
      const failed = runs.filter(r => r.status === "failed").length;

      document.getElementById("stats").innerHTML = `
        <div class="stat"><b>${runs.length}</b><span>总运行次数</span></div>
        <div class="stat"><b>${running}</b><span>进行中</span></div>
        <div class="stat"><b>${completed}</b><span>成功</span></div>
        <div class="stat"><b>${failed}</b><span>失败</span></div>
      `;

      if (!runs.length) {
        document.getElementById("table-wrap").innerHTML =
          `<div class="empty">暂无记录。请另开终端运行：<br><code>python -m src.queue.prefect_memory_flow --dry-run</code></div>`;
      } else {
        const rows = runs.map(r => `
          <tr>
            <td>${(r.started_at || "").replace("T"," ").slice(0,19)}</td>
            <td>${badge(r.runner, r.runner)}</td>
            <td>${r.user_id || "-"}</td>
            <td>${badge(r.status, r.status)}</td>
            <td>${renderSteps(r.steps)}</td>
            <td>${renderReport(r)}</td>
          </tr>
        `).join("");
        document.getElementById("table-wrap").innerHTML = `
          <table>
            <thead><tr>
              <th>时间</th><th>Runner</th><th>User</th><th>状态</th><th>步骤</th><th>结果</th>
            </tr></thead>
            <tbody>${rows}</tbody>
          </table>`;
      }
      document.getElementById("refresh-hint").textContent =
        `上次刷新: ${new Date().toLocaleTimeString()} · Prefect UI ${prefect ? "在线" : "未启动"}`;
    }

    load();
    setInterval(load, 3000);
  </script>
</body>
</html>
"""


async def _prefect_ui_reachable() -> bool:
    try:
        import httpx
    except ImportError:
        return False
    try:
        async with httpx.AsyncClient(timeout=1.5) as client:
            resp = await client.get(PREFECT_UI_URL)
            return resp.status_code < 500
    except Exception:
        return False


async def _fetch_prefect_flow_runs(limit: int = 10) -> List[Dict[str, Any]]:
    try:
        from prefect.client.orchestration import get_client
        from prefect.client.schemas.filters import FlowRunFilter, FlowRunFilterOrder
        from prefect.client.schemas.sorting import FlowRunSort
    except ImportError:
        return []

    try:
        async with get_client() as client:
            runs = await client.read_flow_runs(
                flow_run_filter=FlowRunFilter(),
                limit=limit,
                sort=FlowRunSort.EXPECTED_START_TIME_DESC,
            )
            return [
                {
                    "id": str(r.id),
                    "name": r.name,
                    "state": r.state.type.value if r.state else "unknown",
                    "start_time": r.start_time.isoformat() if r.start_time else "",
                    "total_run_time": r.total_run_time,
                }
                for r in runs
            ]
    except Exception:
        return []


async def homepage(_request):
    return HTMLResponse(DASHBOARD_HTML)


async def api_runs(_request):
    store = get_run_store()
    prefect_ok = await _prefect_ui_reachable()
    prefect_runs: List[Dict[str, Any]] = []
    if prefect_ok:
        prefect_runs = await _fetch_prefect_flow_runs()

    return JSONResponse(
        {
            "runs": store.list_runs(limit=50),
            "prefect_ui_reachable": prefect_ok,
            "prefect_ui_url": PREFECT_UI_URL,
            "prefect_flow_runs": prefect_runs,
        }
    )


def create_app() -> Starlette:
    return Starlette(
        routes=[
            Route("/", homepage),
            Route("/api/runs", api_runs),
        ],
    )


def _start_prefect_server() -> subprocess.Popen:
    return subprocess.Popen(
        [sys.executable, "-m", "prefect", "server", "start"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="memA 流水线 Dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DASHBOARD_PORT)
    parser.add_argument(
        "--with-prefect-ui",
        action="store_true",
        help="后台启动 Prefect Server（官方 UI :4200）",
    )
    parser.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    args = parser.parse_args(argv)

    prefect_proc = None
    if args.with_prefect_ui:
        print(f"启动 Prefect Server → {PREFECT_UI_URL}")
        prefect_proc = _start_prefect_server()

    url = f"http://{args.host}:{args.port}"
    print(f"Dashboard → {url}")
    print("跑 flow: python -m src.queue.prefect_memory_flow --dry-run")

    if not args.no_browser:
        webbrowser.open(url)

    try:
        uvicorn.run(create_app(), host=args.host, port=args.port, log_level="warning")
    finally:
        if prefect_proc is not None:
            prefect_proc.terminate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

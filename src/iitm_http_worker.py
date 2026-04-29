#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
iitm_http_worker.py — IITM 后端的 Python 侧薄适配层
由 GUI 以与 mie_worker.py 完全相同的方式启动（subprocess），
但自身不执行任何物理计算——只负责：
  1. 检测 Julia HTTP 服务是否在线，若无则启动它
  2. 将 GUI 传入的 config 转发给 Julia /simulate 接口
  3. 将 Julia 的响应透传给 GUI（stdout JSON IPC，协议不变）

这样 GUI 层零改动，Julia 层完全独立。
"""

import os
import sys
import json
import time
import signal
import argparse
import subprocess
import threading
import traceback
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")

try:
    import httpx
    _HTTP_LIB = "httpx"
except ImportError:
    import urllib.request
    import urllib.error
    _HTTP_LIB = "urllib"

# ─────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────

JULIA_PORT       = 2700
JULIA_BASE_URL   = f"http://127.0.0.1:{JULIA_PORT}"
HEALTH_ENDPOINT  = f"{JULIA_BASE_URL}/health"
SIMULATE_ENDPOINT = f"{JULIA_BASE_URL}/simulate/stream" 

# Julia 服务器启动超时（秒）；Julia 首次 JIT 编译较慢，留足时间
SERVER_START_TIMEOUT = 120
# HTTP 请求超时（仿真可能耗时较长）
REQUEST_TIMEOUT = 600

SRC_DIR     = Path(__file__).resolve().parent
PROJECT_ROOT= SRC_DIR.parent


def cleanup_stale_iitm_html(project_name: str, keep_artifacts: list[str]) -> None:
    output_dir = PROJECT_ROOT / "outputs" / "iitm" / project_name
    if not output_dir.exists():
        return
    keep = {os.path.basename(str(name)) for name in keep_artifacts if str(name).lower().endswith(".html")}
    for path in output_dir.glob("render*.html"):
        if path.name in keep:
            continue
        try:
            path.unlink()
            print(f">> [Worker] Removed stale HTML: {path.name}", flush=True)
        except Exception as exc:
            print(f">> [Worker WARNING] Failed to remove stale HTML {path.name}: {exc}", flush=True)


def default_field_metadata(config: dict) -> dict:
    requested_mode = str(config.get("field_compute_mode", "proxy_only"))
    return {
        "field_catalog": {
            "proxy": [
                {"name": "beta_back", "label": "后向代理场"},
                {"name": "beta_forward", "label": "前向代理场"},
                {"name": "depol_ratio", "label": "退偏代理场"},
                {"name": "density", "label": "密度场"},
            ],
        },
        "available_field_families": ["proxy"],
        "requested_field_compute_mode": requested_mode,
        "effective_field_compute_mode": "proxy_only",
        "field_mode_note": "",
    }


# ─────────────────────────────────────────────────────────
# HTTP 工具（兼容 httpx 和 urllib）
# ─────────────────────────────────────────────────────────

def http_get(url: str, timeout: float = 5.0) -> dict:
    if _HTTP_LIB == "httpx":
        r = httpx.get(url, timeout=timeout)
        return r.json()
    else:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())


def get_server_info(timeout: float = 3.0) -> dict | None:
    try:
        return http_get(HEALTH_ENDPOINT, timeout=timeout)
    except Exception:
        return None


def http_post_stream(url: str, data: dict, timeout: float = REQUEST_TIMEOUT) -> dict:
    """
    向 /simulate/stream 发送 POST，以 SSE 流式读取进度和结果。
    - event: log    → 实时打印到 stdout（GUI 控制台）
    - event: result → 解析为最终结果 dict 并返回
    - event: error  → 抛出 RuntimeError
    """
    body = json.dumps(data).encode()

    if _HTTP_LIB == "httpx":
        with httpx.Client(timeout=httpx.Timeout(
            connect=15.0,
            read=None,       # 无限等待：仿真完成后服务端关闭连接，客户端自然结束
            write=30.0,
            pool=15.0,
        )) as client:
            with client.stream(
                "POST", url,
                content=body,
                headers={"Content-Type": "application/json"},
            ) as resp:
                return _parse_sse_stream(resp.iter_lines())
    else:
        # urllib 回退：不支持流式，退化为普通 POST
        req = urllib.request.Request(
            url, data=body,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
            # urllib 路径下把结果直接打印一行摘要
            metrics = result.get("metrics", {})
            print(f">> [Julia] R_back={metrics.get('R_back',0):.5f}", flush=True)
            return result


def _parse_sse_stream(line_iter) -> dict:
    """
    解析 SSE 行流，实时打印 log 事件，返回 result 事件的 dict。
    SSE 格式：
      event: log\ndata: <message>\n\n
      event: result\ndata: <json>\n\n
      event: error\ndata: <message>\n\n
    """
    current_event = None
    result = None

    for line in line_iter:
        line = line.strip()
        if not line:
            current_event = None
            continue

        if line.startswith("event:"):
            current_event = line[len("event:"):].strip()
        elif line.startswith("data:"):
            data_str = line[len("data:"):].strip()
            if current_event == "log":
                print(data_str, flush=True)
            elif current_event == "result":
                result = json.loads(data_str)
            elif current_event == "error":
                raise RuntimeError(f"Julia 服务端错误: {data_str}")

    if result is None:
        raise RuntimeError("SSE 流结束但未收到 result 事件")
    return result

# ─────────────────────────────────────────────────────────
# Julia 服务管理
# ─────────────────────────────────────────────────────────

_julia_proc = None   # 全局持有，防止 GC


def is_server_alive() -> bool:
    """检查 Julia 服务是否在线"""
    return get_server_info(timeout=3.0) is not None


def start_julia_server(cpu_limit: str = "4") -> subprocess.Popen:
    """
    启动 Julia HTTP 服务进程。
    服务以后台进程形式运行，本脚本退出时随之终止。
    """
    server_script = SRC_DIR / "julia" / "iitm_server.jl"
    if not server_script.exists():
        raise FileNotFoundError(f"找不到 Julia 服务脚本: {server_script}")

    log_dir = PROJECT_ROOT / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts  = time.strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"julia_server_{ts}.log"

    env = os.environ.copy()
    env["JULIA_NUM_THREADS"] = cpu_limit

    cmd = [
        "pixi", "run", "-e", "julia",
        "julia",
        f"--threads={cpu_limit}",
        f"--project={SRC_DIR / 'julia'}",
        str(server_script),
        "--port", str(JULIA_PORT),
        "--root", str(PROJECT_ROOT),
    ]

    print(f">> [Worker] 启动 Julia 服务: {' '.join(cmd)}", flush=True)
    print(f">> [Worker] 服务日志: {log_path}", flush=True)

    log_file = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        cwd=str(PROJECT_ROOT),
        env=env,
    )
    return proc


def ensure_julia_server(cpu_limit: str = "4") -> bool:
    """
    确保 Julia 服务在线。若未启动则启动并等待就绪。
    返回 True 表示服务可用。
    """
    global _julia_proc

    info = get_server_info(timeout=3.0)
    if info:
        running_threads = info.get("threads", "?")
        if cpu_limit == "auto":
            print(f">> [Worker] Julia 服务已在线，复用现有实例 ({running_threads} 线程)", flush=True)
            print(">> [Worker] 注意：已运行服务不会因本次请求自动切换到新的线程配置", flush=True)
        else:
            try:
                desired_threads = int(cpu_limit)
            except Exception:
                desired_threads = None
            if desired_threads is not None and running_threads != desired_threads:
                print(f">> [Worker] Julia 服务已在线，复用现有实例 ({running_threads} 线程)", flush=True)
                print(f">> [Worker] 注意：本次请求期望 {desired_threads} 线程，但不会自动重启已运行服务", flush=True)
            else:
                print(f">> [Worker] Julia 服务已在线 ({running_threads} 线程)", flush=True)
        return True

    print(f">> [Worker] Julia 服务未就绪，正在启动（超时 {SERVER_START_TIMEOUT}s）...",
          flush=True)

    _julia_proc = start_julia_server(cpu_limit)

    deadline = time.time() + SERVER_START_TIMEOUT
    dot_count = 0
    while time.time() < deadline:
        if is_server_alive():
            print(f"\n>> [Worker] Julia 服务就绪（用时 {time.time()-deadline+SERVER_START_TIMEOUT:.1f}s）",
                  flush=True)
            return True
        if _julia_proc.poll() is not None:
            print(f"\n>> [Worker] Julia 进程意外退出 (code={_julia_proc.returncode})",
                  flush=True)
            return False
        time.sleep(2)
        dot_count += 1
        if dot_count % 5 == 0:
            elapsed = SERVER_START_TIMEOUT - (deadline - time.time())
            print(f">> [Worker] 等待 Julia JIT 编译... {elapsed:.0f}s", flush=True)

    print(">> [Worker] 超时：Julia 服务启动失败", flush=True)
    return False


# ─────────────────────────────────────────────────────────
# 主逻辑
# ─────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", required=True)
    parser.add_argument("--config",       required=True, help="JSON 配置字符串")
    parser.add_argument("--cpu_limit",    default="4")
    args = parser.parse_args()

    result_payload = {"status": "failed", "metrics": {}, "artifacts": []}

    try:
        user_config = json.loads(args.config)
        field_meta = default_field_metadata(user_config)

        # ── 确保 Julia 服务在线 ──
        if not ensure_julia_server(args.cpu_limit):
            raise RuntimeError("Julia HTTP 服务无法启动，请检查 Julia 环境与日志")

        print(">> [Worker] 连接 Julia SSE 流式端点...", flush=True)
        t0 = time.time()

        request_body = {
            "project_name": args.project_name,
            "config":       user_config,
        }

        # http_post_stream 会把每条 log 事件实时 print 到 stdout
        # GUI 的 readline 循环会实时接收并显示
        julia_response = http_post_stream(SIMULATE_ENDPOINT, request_body)
        dt = time.time() - t0

        # ── 透传结果 ──
        if julia_response.get("status") == "success":
            metrics = julia_response.get("metrics", {})
            metrics["duration_sec"] = dt
            field_catalog = julia_response.get("field_catalog", field_meta["field_catalog"])
            artifacts = julia_response.get("artifacts", [])
            cleanup_stale_iitm_html(args.project_name, artifacts)
            result_payload["status"]    = "success"
            result_payload["metrics"]   = metrics
            result_payload["artifacts"] = artifacts
            result_payload["field_catalog"] = field_catalog
            result_payload["available_field_families"] = julia_response.get(
                "available_field_families", field_meta["available_field_families"]
            )
            result_payload["requested_field_compute_mode"] = julia_response.get(
                "requested_field_compute_mode", field_meta["requested_field_compute_mode"]
            )
            result_payload["effective_field_compute_mode"] = julia_response.get(
                "effective_field_compute_mode", field_meta["effective_field_compute_mode"]
            )
            result_payload["field_mode_note"] = julia_response.get(
                "field_mode_note", field_meta["field_mode_note"]
            )
            result_payload["lidar_observation_available"] = bool(
                julia_response.get("lidar_observation_available", False)
            )
            print(f">> [Worker] 仿真成功，耗时 {dt:.2f}s，R_back={metrics.get('R_back',0):.6f}",
                  flush=True)
        else:
            err = julia_response.get("error", "未知错误")
            raise RuntimeError(f"Julia 仿真失败: {err}")

    except Exception as e:
        tb = traceback.format_exc()
        print(f">> [Worker FATAL]\n{tb}", flush=True)
        result_payload["status"] = "error"
        result_payload["error"]  = str(e)

    finally:
        # IPC：最后一行输出 JSON（GUI 依赖此协议）
        print(json.dumps(result_payload), flush=True)


if __name__ == "__main__":
    main()

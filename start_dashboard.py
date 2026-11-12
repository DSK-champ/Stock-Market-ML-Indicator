"""Unified Single-Command Launcher for the AI Trader Dashboard.

Runs all 3 services concurrently in a single terminal:
1. Backend Forward-Testing & Live Engine (Port 8000)
2. Dashboard API Service (Port 8001)
3. Frontend Static Server (Port 8080)
and automatically launches the dashboard in your default web browser.

Usage:
    python start_dashboard.py
    python start_dashboard.py --mode replay --speed 60
    python start_dashboard.py --mock
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def print_banner(mode: str, mock: bool):
    print("=" * 80)
    print("   BTC/USDT 5m AI TRADING INDICATOR — ALL-IN-ONE SYSTEM LAUNCHER")
    print("=" * 80)
    if mock:
        print(" Mode:            MOCK DEMO MODE (Offline Synthetic Signals)")
    else:
        print(f" Mode:            {mode.upper()}")
    print(" Backend Engine:  http://localhost:8000 (WebSocket: ws://localhost:8000/ws/live)")
    print(" Dashboard API:   http://localhost:8001")
    print(" UI Dashboard:    http://localhost:8080/blank.html")
    print("=" * 80)
    print(" Starting all 3 services in this single terminal window...")
    print(" Press Ctrl+C at any time to gracefully stop all services.")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Unified Launcher for AI Trader Dashboard")
    parser.add_argument("--mode", choices=["live", "replay"], default="live", help="Execution mode (default: live)")
    parser.add_argument("--speed", type=float, default=60.0, help="Replay speed multiplier (default: 60.0)")
    parser.add_argument("--from", dest="from_date", type=str, default=None, help="Replay start date (ISO)")
    parser.add_argument("--to", dest="to_date", type=str, default=None, help="Replay end date (ISO)")
    parser.add_argument("--allow-dirty", action="store_true", default=True, help="Allow launch with dirty git tree")
    parser.add_argument("--mock", action="store_true", help="Launch directly into Mock demo mode")
    parser.add_argument("--no-browser", action="store_true", help="Do not automatically open the browser")
    args = parser.parse_args()

    print_banner(args.mode, args.mock)

    procs = []
    
    # Auto-detect project virtual environment python (Windows & Linux/macOS)
    venv_win = BASE_DIR / ".venv" / "Scripts" / "python.exe"
    venv_nix = BASE_DIR / ".venv" / "bin" / "python"
    if venv_win.exists():
        py_bin = str(venv_win)
    elif venv_nix.exists():
        py_bin = str(venv_nix)
    else:
        py_bin = sys.executable
    
    print(f" Using Python interpreter: {py_bin}\n")

    try:
        # 1. Start Frontend Static Server (Port 8080)
        print("[1/3] Starting Frontend HTTP Server on port 8080...")
        p_front = subprocess.Popen(
            [py_bin, "-m", "http.server", "8080", "--directory", str(BASE_DIR / "frontend")],
            cwd=str(BASE_DIR),
        )
        procs.append(("Frontend Server (8080)", p_front))

        # 2. Start Dashboard API (Port 8001)
        print("[2/3] Starting Dashboard API on port 8001...")
        p_dash = subprocess.Popen(
            [py_bin, "-m", "uvicorn", "dashboard_api.app:app", "--port", "8001", "--host", "0.0.0.0", "--log-level", "warning"],
            cwd=str(BASE_DIR),
        )
        procs.append(("Dashboard API (8001)", p_dash))

        # 3. Start Backend Engine (Port 8000) - only if not mock-only
        if not args.mock:
            backend_cmd = [py_bin, "-m", "backend.live.run", "--mode", args.mode]
            if args.allow_dirty:
                backend_cmd.append("--allow-dirty")
            if args.mode == "replay":
                backend_cmd.extend(["--speed", str(args.speed)])
                if args.from_date:
                    backend_cmd.extend(["--from", args.from_date])
                if args.to_date:
                    backend_cmd.extend(["--to", args.to_date])

            print(f"[3/3] Starting Backend Forward Engine on port 8000 ({args.mode.upper()})...")
            p_back = subprocess.Popen(backend_cmd, cwd=str(BASE_DIR))
            procs.append(("Backend Engine (8000)", p_back))
        else:
            print("[3/3] Skipping backend engine (Running in Mock Demo Mode).")

        # 4. Wait for servers to bind and launch browser
        time.sleep(1.8)
        target_url = "http://localhost:8080/blank.html?mock=1" if args.mock else "http://localhost:8080/blank.html"

        if not args.no_browser:
            print(f"\n[OK] Opening dashboard in your browser: {target_url}\n")
            webbrowser.open(target_url)
        else:
            print(f"\n[OK] Dashboard ready at: {target_url}\n")

        # Keep running and monitor processes
        while True:
            time.sleep(1.0)
            for name, p in procs:
                ret = p.poll()
                if ret is not None:
                    print(f"\n[WARN] {name} exited unexpectedly with code {ret}.")
                    raise KeyboardInterrupt

    except KeyboardInterrupt:
        print("\n\n" + "=" * 80)
        print(" SHUTTING DOWN ALL SERVICES...")
        print("=" * 80)
        for name, p in procs:
            if p.poll() is None:
                print(f" Stopping {name}...")
                p.terminate()
                try:
                    p.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    p.kill()
        print(" All services stopped. Goodbye!")
        print("=" * 80)


if __name__ == "__main__":
    main()


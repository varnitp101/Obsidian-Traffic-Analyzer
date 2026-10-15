"""
Single-Command Platform Launcher for UK Traffic Accident Analytics & MLOps
===========================================================================
Launches the entire stack (PostgreSQL verification, FastAPI inference microservice,
and the unified Glassmorphic Streamlit Control Plane) with a single command:

    python run_platform.py
"""

import os
import sys
import time
import subprocess
import webbrowser
import socket


def check_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    """Check whether a local port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(1.0)
        return s.connect_ex((host, port)) == 0


def ensure_postgres_running():
    """Verify or start the PostgreSQL container."""
    print("\n[1/3] Checking PostgreSQL container status...")
    try:
        res = subprocess.run(["docker", "ps", "--filter", "name=accident_postgres", "--format", "{{.Names}}"],
                             capture_output=True, text=True, check=False)
        if "accident_postgres" in res.stdout:
            print("      [OK] PostgreSQL (accident_postgres) is running on port 5432.")
            return True
        
        # Check if container exists but is stopped
        res_all = subprocess.run(["docker", "ps", "-a", "--filter", "name=accident_postgres", "--format", "{{.Names}}"],
                                 capture_output=True, text=True, check=False)
        if "accident_postgres" in res_all.stdout:
            print("      [+] Starting existing accident_postgres container...")
            subprocess.run(["docker", "start", "accident_postgres"], check=True)
            time.sleep(3)
            print("      [OK] PostgreSQL started successfully.")
            return True
        else:
            print("      [+] Spinning up PostgreSQL via docker compose...")
            subprocess.run(["docker", "compose", "up", "-d", "postgres"], check=True)
            time.sleep(4)
            print("      [OK] PostgreSQL initialized successfully.")
            return True
    except FileNotFoundError:
        print("      [!] Docker CLI not found in PATH. Assuming external or native PostgreSQL on port 5432.")
        return check_port_in_use(5432)
    except Exception as e:
        print(f"      [!] Warning during Postgres check: {e}")
        return False


def start_fastapi_service():
    """Start the FastAPI inference service as a background subprocess."""
    print("\n[2/3] Checking FastAPI Inference Service...")
    if check_port_in_use(8000):
        print("      [OK] Port 8000 already active (FastAPI service is already listening).")
        return None
    
    print("      [+] Spawning FastAPI service on http://localhost:8000...")
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    # Wait up to 6 seconds for FastAPI to bind
    for _ in range(12):
        if check_port_in_use(8000):
            print("      [OK] FastAPI service is live on http://localhost:8000")
            return proc
        time.sleep(0.5)
    
    print("      [!] FastAPI service spawned (binding in progress).")
    return proc


def launch_unified_dashboard():
    """Launch the unified Streamlit Control Plane and keep in foreground."""
    print("\n[3/3] Launching Unified Control Center Dashboard...")
    port = 8501
    
    print("=" * 72)
    print("  Obsidian Traffic Analyzer - MLOps & Analytics Platform is Online!")
    print("=" * 72)
    print("  * Unified Control Center UI : http://localhost:8501")
    print("  * FastAPI Interactive Docs  : http://localhost:8000/docs")
    print("  * FastAPI Health Endpoint   : http://localhost:8000/health")
    print("  * PostgreSQL PostGIS Layer  : localhost:5432 (accident_db)")
    print("=" * 72)
    print("\nPress Ctrl+C at any time to shut down all platform services.\n")
    
    # Try opening browser after a brief delay
    time.sleep(1.5)
    try:
        webbrowser.open("http://localhost:8501")
    except Exception:
        pass

    # Run streamlit as foreground process
    subprocess.run([
        sys.executable, "-m", "streamlit", "run",
        "src/dashboard/app.py",
        "--server.port", str(port),
        "--server.headless", "true"
    ])


def main():
    api_proc = None
    try:
        ensure_postgres_running()
        api_proc = start_fastapi_service()
        launch_unified_dashboard()
    except KeyboardInterrupt:
        print("\n\n[!] Received shutdown signal. Terminating platform services...")
    finally:
        if api_proc and api_proc.poll() is None:
            print("[+] Stopping FastAPI inference service...")
            api_proc.terminate()
            api_proc.wait(timeout=3)
        print("[OK] Platform stopped cleanly.")


if __name__ == "__main__":
    main()

"""
Live training monitor + control panel for GigaLearnCPP.

Serves a dashboard at http://localhost:8050 with:
  - Multi-bot management (create, select, train different bots)
  - Real-time charts for all training and gameplay metrics
  - Start / Save & Stop / Kill controls
  - Quick actions: edit rewards, open visualizer, build for RLBot
  - Persistent metrics per bot (saved to metrics_log.jsonl)
  - Checkpoint browser

Usage:
    python monitor/server.py                 # Dashboard only
    python monitor/server.py --port 9000     # Custom port

Or use the one-click launcher:
    train.bat
"""

import argparse
import ctypes
import ctypes.wintypes as wintypes
import http.server
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# Paths relative to the rocket_league/ directory (one level up from this script)
ROCKET_LEAGUE_DIR = Path(__file__).parent.parent.resolve()
EXE_PATH = ROCKET_LEAGUE_DIR / "build" / "RocketLeagueStrategyBot.exe"
CHECKPOINTS_DIR = ROCKET_LEAGUE_DIR / "checkpoints"
STATIC_DIR = Path(__file__).parent.resolve() / "static"
SRC_DIR = ROCKET_LEAGUE_DIR / "src"
RLBOT_TEMPLATE_DIR = ROCKET_LEAGUE_DIR / "GigaLearnCPP-Leak" / "rlbot"

# ---------------------------------------------------------------------------
# Metric parser
# ---------------------------------------------------------------------------

METRIC_PATTERNS = {
    "avg_reward":       re.compile(r"Average Step Reward:\s*([\d.e+-]+)"),
    "entropy":          re.compile(r"Policy Entropy:\s*([\d.e+-]+)"),
    "policy_update":    re.compile(r"Policy Update Magnitude:\s*([\d.e+-]+)"),
    "critic_update":    re.compile(r"Critic Update Magnitude:\s*([\d.e+-]+)"),
    "sps":              re.compile(r"Overall Steps/Second:\s*([\d.,]+)"),
    "total_timesteps":  re.compile(r"Total Timesteps:\s*([\d.,]+)"),
    "total_iterations": re.compile(r"Total Iterations:\s*(\d+)"),
    "collection_time":  re.compile(r"Collection Time:\s*([\d.e+-]+)"),
    "consumption_time": re.compile(r"Consumption Time:\s*([\d.e+-]+)"),
    "inference_time":   re.compile(r"Inference Time:\s*([\d.e+-]+)"),
    "env_step_time":    re.compile(r"Env Step Time:\s*([\d.e+-]+)"),
    "ppo_learn_time":   re.compile(r"PPO Learn Time:\s*([\d.e+-]+)"),
    "player_speed":         re.compile(r"Player/Speed:\s*([\d.,e+-]+)"),
    "player_in_air":        re.compile(r"Player/In Air:\s*([\d.,e+-]+)"),
    "player_ball_touch":    re.compile(r"Player/Ball Touch:\s*([\d.,e+-]+)"),
    "player_speed_to_ball": re.compile(r"Player/Speed Toward Ball:\s*([\d.,e+-]+)"),
    "player_boost":         re.compile(r"Player/Boost:\s*([\d.,e+-]+)"),
    "touch_height":         re.compile(r"Player/Touch Height:\s*([\d.,e+-]+)"),
    "goal_speed":           re.compile(r"Game/Goal Speed:\s*([\d.,e+-]+)"),
}


def parse_number(s: str) -> float:
    return float(s.replace(",", ""))


class MetricStore:
    def __init__(self):
        self.history = []
        self._current = {}
        self._lock = threading.Lock()
        self._in_block = False
        self.start_time = time.time()
        self._log_file = None
        self._log_path = None

    def set_log_path(self, path: Path):
        """Set the metrics log file path (per-bot)."""
        self.close()
        self._log_path = path
        self.history.clear()
        self._current = {}
        self._in_block = False
        self.start_time = time.time()

    def load_from_disk(self):
        """Load previous metrics from the current bot's metrics_log.jsonl."""
        if not self._log_path or not self._log_path.exists():
            return
        count = 0
        with open(self._log_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    self.history.append(entry)
                    count += 1
                except json.JSONDecodeError:
                    continue
        if count > 0:
            print(f"Loaded {count} iterations from {self._log_path}")

    def _open_log(self):
        if self._log_file is None and self._log_path:
            self._log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(self._log_path, "a")

    def _flush_entry(self, entry: dict):
        self._open_log()
        if self._log_file:
            self._log_file.write(json.dumps(entry) + "\n")
            self._log_file.flush()

    def process_line(self, line: str):
        stripped = line.strip()
        if stripped.startswith("=" * 10):
            with self._lock:
                if self._in_block and self._current:
                    self._current["wall_time"] = round(time.time() - self.start_time, 1)
                    self._current["timestamp"] = time.time()
                    self.history.append(self._current)
                    self._flush_entry(self._current)
                    self._current = {}
                self._in_block = not self._in_block
            return
        if not self._in_block:
            return
        for key, pattern in METRIC_PATTERNS.items():
            m = pattern.search(stripped)
            if m:
                with self._lock:
                    self._current[key] = parse_number(m.group(1))
                return

    def get_json(self, since_idx=0) -> dict:
        with self._lock:
            return {"total": len(self.history), "data": self.history[since_idx:]}

    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None


# ---------------------------------------------------------------------------
# Bot manager
# ---------------------------------------------------------------------------

class BotManager:
    """Manages named bots, each with their own checkpoint folder and metrics."""

    def __init__(self):
        CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)
        self.current_bot = "default"

    def list_bots(self) -> list:
        """List all bot directories with summary info."""
        bots = []
        if not CHECKPOINTS_DIR.exists():
            return bots
        for d in sorted(CHECKPOINTS_DIR.iterdir()):
            if not d.is_dir():
                continue
            # Skip numbered dirs (legacy flat checkpoints)
            try:
                int(d.name)
                continue
            except ValueError:
                pass
            info = {"name": d.name, "checkpoints": 0, "latest_timesteps": 0}
            # Count checkpoint subdirectories
            for sub in d.iterdir():
                if sub.is_dir():
                    try:
                        int(sub.name)
                        info["checkpoints"] += 1
                        stats_file = sub / "RUNNING_STATS.json"
                        if stats_file.exists():
                            try:
                                stats = json.load(open(stats_file))
                                ts = stats.get("total_timesteps", 0)
                                if ts > info["latest_timesteps"]:
                                    info["latest_timesteps"] = ts
                            except (json.JSONDecodeError, OSError):
                                pass
                    except ValueError:
                        pass
            bots.append(info)
        return bots

    def create_bot(self, name: str) -> dict:
        """Create a new named bot directory."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        if not name:
            return {"ok": False, "error": "Invalid bot name"}
        bot_dir = CHECKPOINTS_DIR / name
        if bot_dir.exists():
            return {"ok": False, "error": f"Bot '{name}' already exists"}
        bot_dir.mkdir(parents=True)
        return {"ok": True, "name": name}

    def delete_bot(self, name: str) -> dict:
        """Delete a bot and all its checkpoints/metrics."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        if not name:
            return {"ok": False, "error": "Invalid bot name"}
        bot_dir = CHECKPOINTS_DIR / name
        if not bot_dir.exists():
            return {"ok": False, "error": f"Bot '{name}' not found"}
        try:
            shutil.rmtree(bot_dir)
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    def get_bot_dir(self, name: str) -> Path:
        return CHECKPOINTS_DIR / name

    def get_metrics_path(self, name: str) -> Path:
        return CHECKPOINTS_DIR / name / "metrics_log.jsonl"

    def scan_checkpoints(self, bot_name: str) -> list:
        """Scan a specific bot's checkpoint directory."""
        bot_dir = CHECKPOINTS_DIR / bot_name
        if not bot_dir.exists():
            return []
        results = []
        for d in sorted(bot_dir.iterdir()):
            if not d.is_dir():
                continue
            try:
                int(d.name)
            except ValueError:
                continue
            stats_file = d / "RUNNING_STATS.json"
            info = {"name": d.name, "timesteps": 0, "iterations": 0}
            try:
                info["timesteps"] = int(d.name)
            except ValueError:
                pass
            if stats_file.exists():
                try:
                    with open(stats_file) as f:
                        stats = json.load(f)
                    info["iterations"] = stats.get("total_iterations", 0)
                    info["timesteps"] = stats.get("total_timesteps", info["timesteps"])
                    rs = stats.get("return_stat", {})
                    info["mean_return"] = round(rs.get("mean", 0), 2)
                    info["episodes"] = rs.get("count", 0)
                except (json.JSONDecodeError, OSError):
                    pass
            info["has_model"] = (d / "POLICY.lt").exists()
            results.append(info)
        return results


# ---------------------------------------------------------------------------
# Process manager
# ---------------------------------------------------------------------------

class TrainingManager:
    PY313_DIR = Path(r"C:\Program Files\WindowsApps\PythonSoftwareFoundation.Python.3.13_3.13.3312.0_x64__qbz5n2kfra8p0")

    def __init__(self, store: MetricStore, bot_mgr: BotManager):
        self.store = store
        self.bot_mgr = bot_mgr
        self.proc = None
        self._reader_thread = None
        self._lock = threading.Lock()
        self.status = "idle"
        self.exit_code = None
        self.log_lines = []

    @staticmethod
    def _ensure_pth_isolation():
        """Create python313._pth and copy stdlib .pyd files for the embedded DLL."""
        build_dir = EXE_PATH.parent
        pth = build_dir / "python313._pth"
        py_lib = TrainingManager.PY313_DIR / "Lib"
        py_dlls = TrainingManager.PY313_DIR / "DLLs"
        # Include GigaLearnCPP dir so RenderSender can import
        # python_scripts.render_receiver (PYTHONPATH is ignored
        # by embedded Python when a _pth file exists)
        giga_py = ROCKET_LEAGUE_DIR / "GigaLearnCPP-Leak" / "GigaLearnCPP"
        expected = f".\n{py_lib}\n{py_dlls}\n{giga_py}\n"
        try:
            if not pth.exists() or pth.read_text() != expected:
                pth.write_text(expected)
                print(f"Updated {pth} for embedded Python isolation")
        except OSError as e:
            print(f"Warning: Could not update {pth}: {e}")

        # Copy stdlib .pyd files to build dir — the Windows Store
        # Python's DLLs folder is access-restricted so the embedded
        # interpreter can't load them from there directly.
        if py_dlls.is_dir() and not (build_dir / "_socket.pyd").exists():
            copied = 0
            for pyd in py_dlls.glob("*.pyd"):
                dest = build_dir / pyd.name
                if not dest.exists():
                    try:
                        shutil.copy2(pyd, dest)
                        copied += 1
                    except OSError:
                        pass
            if copied:
                print(f"Copied {copied} .pyd files to {build_dir}")

    def start(self, bot_name: str = None):
        with self._lock:
            if self.status == "running":
                return {"ok": False, "error": "Already running"}
            if not EXE_PATH.exists():
                return {"ok": False, "error": f"{EXE_PATH} not found. Run build.bat first."}

            if bot_name:
                self.bot_mgr.current_bot = bot_name

            name = self.bot_mgr.current_bot
            self.log_lines.clear()
            self.exit_code = None

            # Ensure bot directory exists
            bot_dir = self.bot_mgr.get_bot_dir(name)
            bot_dir.mkdir(parents=True, exist_ok=True)

            # Switch metrics store to this bot
            self.store.close()
            self.store.set_log_path(self.bot_mgr.get_metrics_path(name))
            self.store.load_from_disk()

            self._ensure_pth_isolation()

            env = os.environ.copy()
            env.pop("PYTHONHOME", None)
            env["PYTHONNOUSERSITE"] = "1"
            env.pop("PYTHONPATH", None)

            cmd = [str(EXE_PATH), "--bot", name]

            self.proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(ROCKET_LEAGUE_DIR),
                env=env,
                bufsize=0,
            )
            self.status = "running"
            self._reader_thread = threading.Thread(target=self._read_output, daemon=True)
            self._reader_thread.start()
            return {"ok": True, "bot": name}

    def _read_output(self):
        for raw_line in iter(self.proc.stdout.readline, b""):
            line = raw_line.decode("utf-8", errors="replace")
            sys.stdout.write(line)
            sys.stdout.flush()
            self.store.process_line(line)
            self.log_lines.append(line.rstrip())
            if len(self.log_lines) > 200:
                self.log_lines.pop(0)

        self.proc.wait()
        with self._lock:
            self.exit_code = self.proc.returncode
            self.status = "idle"
            self.proc = None

    def save_and_stop(self):
        with self._lock:
            if self.status != "running" or self.proc is None:
                return {"ok": False, "error": "Not running"}
            self.status = "stopping"
        _send_q_to_console()
        return {"ok": True, "msg": "Save queued - will stop after current iteration"}

    def kill(self):
        with self._lock:
            if self.proc is None:
                return {"ok": False, "error": "Not running"}
            self.proc.terminate()
            self.status = "idle"
        return {"ok": True, "msg": "Process terminated"}

    def get_status(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "exit_code": self.exit_code,
                "pid": self.proc.pid if self.proc else None,
                "bot": self.bot_mgr.current_bot,
            }


# ---------------------------------------------------------------------------
# Quick actions
# ---------------------------------------------------------------------------

def open_reward_file():
    """Open main.cpp in the default editor."""
    target = SRC_DIR / "main.cpp"
    if not target.exists():
        return {"ok": False, "error": "main.cpp not found"}
    try:
        os.startfile(str(target))
        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def build_for_rlbot(bot_name: str):
    """Export the latest checkpoint as an RLBot-ready package."""
    bot_dir = CHECKPOINTS_DIR / bot_name
    if not bot_dir.exists():
        return {"ok": False, "error": f"Bot '{bot_name}' not found"}

    # Find latest checkpoint
    latest = None
    latest_ts = 0
    for d in bot_dir.iterdir():
        if not d.is_dir():
            continue
        try:
            ts = int(d.name)
            if ts > latest_ts and (d / "POLICY.lt").exists():
                latest_ts = ts
                latest = d
        except ValueError:
            pass

    if latest is None:
        return {"ok": False, "error": "No checkpoint with a saved model found"}

    export_dir = ROCKET_LEAGUE_DIR / "rlbot_export" / bot_name
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy model files
    for f in latest.iterdir():
        if f.suffix == ".lt":
            shutil.copy2(f, export_dir / f.name)

    # Copy RLBot template files if available
    if RLBOT_TEMPLATE_DIR.exists():
        for f in RLBOT_TEMPLATE_DIR.iterdir():
            if f.is_file():
                shutil.copy2(f, export_dir / f.name)

    # Copy the exe
    if EXE_PATH.exists():
        shutil.copy2(EXE_PATH, export_dir / EXE_PATH.name)

    # Copy required DLLs
    for dll in EXE_PATH.parent.glob("*.dll"):
        shutil.copy2(dll, export_dir / dll.name)

    # Copy python313._pth and .pyd files so the embedded Python
    # can find its standard library (encodings, socket, etc.)
    pth_file = EXE_PATH.parent / "python313._pth"
    if pth_file.exists():
        shutil.copy2(pth_file, export_dir / pth_file.name)
    for pyd in EXE_PATH.parent.glob("*.pyd"):
        shutil.copy2(pyd, export_dir / pyd.name)

    return {
        "ok": True,
        "path": str(export_dir),
        "checkpoint": latest.name,
        "files": [f.name for f in export_dir.iterdir()],
    }


RSVIS_DIR = ROCKET_LEAGUE_DIR / "RocketSimVis"

# Track the separate render process so we can kill it later
_render_proc = None


def open_visualizer(bot_name: str):
    """Launch a separate exe in render mode + RocketSimVis."""
    global _render_proc

    if not EXE_PATH.exists():
        return {"ok": False, "error": f"{EXE_PATH} not found. Run build.bat first."}

    main_py = RSVIS_DIR / "src" / "main.py"
    if not main_py.exists():
        return {"ok": False, "error": "RocketSimVis not found at " + str(RSVIS_DIR)}

    # Kill previous render process if still running
    if _render_proc and _render_proc.poll() is None:
        _render_proc.terminate()
        _render_proc = None

    try:
        # Ensure the _pth file includes GigaLearnCPP for render_receiver
        TrainingManager._ensure_pth_isolation()

        # Launch exe in render-only mode (separate from training)
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)

        _render_proc = subprocess.Popen(
            [str(EXE_PATH), "--bot", bot_name, "--render"],
            cwd=str(ROCKET_LEAGUE_DIR),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        # Launch RocketSimVis (clean env for PyQt5/moderngl)
        viz_env = os.environ.copy()
        for key in ("PYTHONHOME", "PYTHONNOUSERSITE", "PYTHONPATH"):
            viz_env.pop(key, None)

        subprocess.Popen(
            ["py", "-3", str(main_py)],
            cwd=str(RSVIS_DIR),
            env=viz_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        return {"ok": True}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def stop_visualizer():
    """Kill the render process."""
    global _render_proc
    if _render_proc and _render_proc.poll() is None:
        _render_proc.terminate()
        _render_proc = None
        return {"ok": True, "msg": "Render process stopped"}
    return {"ok": False, "error": "No render process running"}


# ---------------------------------------------------------------------------
# Win32: inject 'Q' keypress into console input buffer
# ---------------------------------------------------------------------------

class KEY_EVENT_RECORD(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", wintypes.BOOL),
        ("wRepeatCount", wintypes.WORD),
        ("wVirtualKeyCode", wintypes.WORD),
        ("wVirtualScanCode", wintypes.WORD),
        ("uChar", ctypes.c_wchar),
        ("dwControlKeyState", wintypes.DWORD),
    ]


class _Event(ctypes.Union):
    _fields_ = [("KeyEvent", KEY_EVENT_RECORD)]


class INPUT_RECORD(ctypes.Structure):
    _fields_ = [
        ("EventType", wintypes.WORD),
        ("Event", _Event),
    ]


def _send_q_to_console():
    try:
        kernel32 = ctypes.windll.kernel32
        stdin_handle = kernel32.GetStdHandle(wintypes.DWORD(-10))
        for key_down in (True, False):
            ir = INPUT_RECORD()
            ir.EventType = 0x0001
            ir.Event.KeyEvent.bKeyDown = key_down
            ir.Event.KeyEvent.wRepeatCount = 1
            ir.Event.KeyEvent.wVirtualKeyCode = 0x51
            ir.Event.KeyEvent.wVirtualScanCode = 0x10
            ir.Event.KeyEvent.uChar = 'Q'
            ir.Event.KeyEvent.dwControlKeyState = 0
            written = wintypes.DWORD(0)
            kernel32.WriteConsoleInputW(
                stdin_handle, ctypes.byref(ir), 1, ctypes.byref(written)
            )
    except Exception as e:
        print(f"Warning: Could not send Q to console: {e}")


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

MIME_TYPES = {
    ".html": "text/html",
    ".css":  "text/css",
    ".js":   "application/javascript",
}


def make_handler(store: MetricStore, manager: TrainingManager, bot_mgr: BotManager):
    class Handler(http.server.BaseHTTPRequestHandler):
        def _parse_body(self) -> dict:
            length = int(self.headers.get("Content-Length", 0))
            if length == 0:
                return {}
            try:
                return json.loads(self.rfile.read(length))
            except (json.JSONDecodeError, ValueError):
                return {}

        def do_GET(self):
            parsed = urlparse(self.path)
            path = parsed.path
            qs = parse_qs(parsed.query)

            if path in ("/", "/index.html"):
                self._serve_file(STATIC_DIR / "index.html")
            elif path.startswith("/static/"):
                rel = path[len("/static/"):]
                file_path = STATIC_DIR / rel
                if file_path.is_file() and STATIC_DIR in file_path.resolve().parents:
                    self._serve_file(file_path)
                else:
                    self.send_response(404)
                    self.end_headers()
            elif path == "/api/metrics":
                since = int(qs.get("since", ["0"])[0])
                self._json(store.get_json(since))
            elif path == "/api/status":
                self._json(manager.get_status())
            elif path == "/api/log":
                self._json({"lines": manager.log_lines[-50:]})
            elif path == "/api/checkpoints":
                bot = qs.get("bot", [bot_mgr.current_bot])[0]
                self._json(bot_mgr.scan_checkpoints(bot))
            elif path == "/api/bots":
                self._json({"bots": bot_mgr.list_bots(), "current": bot_mgr.current_bot})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            body = self._parse_body()

            if self.path == "/api/start":
                bot = body.get("bot", bot_mgr.current_bot)
                self._json(manager.start(bot_name=bot))
            elif self.path == "/api/stop":
                self._json(manager.save_and_stop())
            elif self.path == "/api/kill":
                self._json(manager.kill())
            elif self.path == "/api/bots/create":
                name = body.get("name", "")
                self._json(bot_mgr.create_bot(name))
            elif self.path == "/api/bots/select":
                name = body.get("name", "")
                bot_mgr.current_bot = name
                # Reload metrics for new bot
                store.close()
                store.set_log_path(bot_mgr.get_metrics_path(name))
                store.load_from_disk()
                self._json({"ok": True, "bot": name})
            elif self.path == "/api/bots/delete":
                name = body.get("name", "")
                if name == bot_mgr.current_bot:
                    store.close()
                result = bot_mgr.delete_bot(name)
                if result.get("ok") and name == bot_mgr.current_bot:
                    # Switch to first available bot or default
                    bots = bot_mgr.list_bots()
                    bot_mgr.current_bot = bots[0]["name"] if bots else "default"
                    store.set_log_path(bot_mgr.get_metrics_path(bot_mgr.current_bot))
                    store.load_from_disk()
                self._json(result)
            elif self.path == "/api/open-rewards":
                self._json(open_reward_file())
            elif self.path == "/api/build-rlbot":
                bot = body.get("bot", bot_mgr.current_bot)
                self._json(build_for_rlbot(bot))
            elif self.path == "/api/open-viz":
                bot = body.get("bot", bot_mgr.current_bot)
                self._json(open_visualizer(bot))
            elif self.path == "/api/stop-viz":
                self._json(stop_visualizer())
            else:
                self.send_response(404)
                self.end_headers()

        def _serve_file(self, path: Path):
            suffix = path.suffix.lower()
            content_type = MIME_TYPES.get(suffix, mimetypes.guess_type(str(path))[0] or "application/octet-stream")
            try:
                data = path.read_bytes()
            except OSError:
                self.send_response(404)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _json(self, obj):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(json.dumps(obj).encode())

        def log_message(self, *a):
            pass

    return Handler


# ---------------------------------------------------------------------------
# Orphan cleanup
# ---------------------------------------------------------------------------

def _kill_orphan_training():
    """Kill any RocketLeagueStrategyBot.exe left over from a previous session."""
    exe_name = EXE_PATH.name
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            if exe_name.lower() in line.lower():
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    pid = parts[1].strip('"')
                    print(f"Killing orphan {exe_name} (PID {pid})")
                    subprocess.run(["taskkill", "/F", "/PID", pid],
                                   capture_output=True, timeout=5)
    except Exception as e:
        print(f"Warning: Could not check for orphan processes: {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RL Training Monitor + Controls")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()

    # Kill any orphan training processes from previous sessions
    _kill_orphan_training()

    bot_mgr = BotManager()
    store = MetricStore()

    # Load metrics for the default bot
    store.set_log_path(bot_mgr.get_metrics_path(bot_mgr.current_bot))
    store.load_from_disk()

    manager = TrainingManager(store, bot_mgr)

    server = http.server.HTTPServer(
        ("0.0.0.0", args.port),
        make_handler(store, manager, bot_mgr),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{args.port}"
    print(f"Dashboard ready: {url}")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        store.close()
        if manager.proc:
            manager.proc.terminate()
        server.shutdown()


if __name__ == "__main__":
    main()

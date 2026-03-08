"""
Live training monitor + control panel for GigaLearnCPP.

Serves a dashboard at http://localhost:8050 with:
  - Multi-bot management (create, select, train different bots)
  - Real-time charts for all training and gameplay metrics
  - Start / Save & Stop / Kill controls
  - Quick actions: edit config, rebuild, test game, build for RLBot
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
RLBOT_PYTHON = Path(r"C:\Users\noame\AppData\Local\RLBotGUIX\Python311\python.exe")

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

    def get_json(self, since_idx=0, max_points=0) -> dict:
        with self._lock:
            data = self.history[since_idx:]
            total = len(self.history)

        if max_points > 0 and len(data) > max_points:
            data = self._downsample(data, max_points)

        return {"total": total, "data": data}

    @staticmethod
    def _downsample(data, max_points):
        """Keep every Nth point, always including first and last."""
        n = len(data)
        if n <= max_points:
            return data
        tail = min(20, max_points // 4)
        head_budget = max_points - tail
        head_data = data[: n - tail]
        tail_data = data[n - tail :]
        step = max(1, len(head_data) / head_budget)
        sampled = []
        i = 0.0
        while int(i) < len(head_data) and len(sampled) < head_budget:
            sampled.append(head_data[int(i)])
            i += step
        return sampled + tail_data

    def close(self):
        if self._log_file:
            self._log_file.close()
            self._log_file = None


# ---------------------------------------------------------------------------
# Default bot config (shared between server and main.cpp)
# ---------------------------------------------------------------------------

DEFAULT_BOT_CONFIG = {
    "gamemode": "1v1",
    "training": {
        "numGames": 32, "tickSkip": 8,
        "randomSeed": 42, "tsPerSave": 5000000,
    },
    "ppo": {
        "tsPerItr": 50000, "batchSize": 50000, "miniBatchSize": 50000,
        "epochs": 2, "entropyScale": 0.035,
        "gaeGamma": 0.99, "gaeLambda": 0.95, "clipRange": 0.2,
        "policyLR": 0.00015, "criticLR": 0.00015,
    },
    "network": {
        "sharedHead": [256, 256],
        "policy": [256, 256, 256],
        "critic": [256, 256, 256],
    },
}


def _deep_merge(defaults: dict, overrides: dict) -> dict:
    """Merge overrides into a copy of defaults (one level deep)."""
    result = {}
    for key, default_val in defaults.items():
        if key in overrides:
            if isinstance(default_val, dict) and isinstance(overrides[key], dict):
                merged = dict(default_val)
                merged.update(overrides[key])
                result[key] = merged
            else:
                result[key] = overrides[key]
        else:
            result[key] = default_val if not isinstance(default_val, dict) else dict(default_val)
    return result


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
            try:
                int(d.name)
                continue
            except ValueError:
                pass
            info = {"name": d.name, "checkpoints": 0, "latest_timesteps": 0}
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

    def create_bot(self, name: str, config: dict = None) -> dict:
        """Create a new named bot directory, optionally with config."""
        name = re.sub(r"[^a-zA-Z0-9_-]", "", name)
        if not name:
            return {"ok": False, "error": "Invalid bot name"}
        bot_dir = CHECKPOINTS_DIR / name
        if bot_dir.exists():
            return {"ok": False, "error": f"Bot '{name}' already exists"}
        bot_dir.mkdir(parents=True)
        if config:
            self.save_bot_config(name, config)
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

    def get_bot_config(self, name: str) -> dict:
        """Read bot_config.json for a bot, return defaults if missing."""
        config_path = CHECKPOINTS_DIR / name / "bot_config.json"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    saved = json.load(f)
                return _deep_merge(DEFAULT_BOT_CONFIG, saved)
            except (json.JSONDecodeError, OSError):
                pass
        import copy
        return copy.deepcopy(DEFAULT_BOT_CONFIG)

    def save_bot_config(self, name: str, config: dict) -> dict:
        """Save bot_config.json for a bot."""
        bot_dir = CHECKPOINTS_DIR / name
        bot_dir.mkdir(parents=True, exist_ok=True)
        config_path = bot_dir / "bot_config.json"
        try:
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            return {"ok": True}
        except OSError as e:
            return {"ok": False, "error": str(e)}

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
        giga_py = ROCKET_LEAGUE_DIR / "GigaLearnCPP-Leak" / "GigaLearnCPP"
        expected = f".\n{py_lib}\n{py_dlls}\n{giga_py}\n"
        try:
            if not pth.exists() or pth.read_text() != expected:
                pth.write_text(expected)
                print(f"Updated {pth} for embedded Python isolation")
        except OSError as e:
            print(f"Warning: Could not update {pth}: {e}")

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

            bot_dir = self.bot_mgr.get_bot_dir(name)
            bot_dir.mkdir(parents=True, exist_ok=True)

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


def _pick_folder(title="Select Export Folder"):
    """Open a modern Windows folder picker dialog. Returns path or None."""
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        folder = filedialog.askdirectory(title=title, parent=root)
        root.destroy()
        return folder or None
    except Exception:
        return None


def build_for_rlbot(bot_name: str, export_path: str = None):
    """Export the latest checkpoint as an RLBot-ready package."""
    bot_dir = CHECKPOINTS_DIR / bot_name
    if not bot_dir.exists():
        return {"ok": False, "error": f"Bot '{bot_name}' not found"}

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

    if export_path:
        export_dir = Path(export_path)
    else:
        picked = _pick_folder("Export RLBot Package -- Choose Folder")
        if picked:
            export_dir = Path(picked)
        else:
            return {"ok": False, "error": "Export cancelled"}

    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy model files
    for f in latest.iterdir():
        if f.suffix == ".lt":
            shutil.copy2(f, export_dir / f.name)

    # Copy RLBot template files
    if RLBOT_TEMPLATE_DIR.exists():
        for f in RLBOT_TEMPLATE_DIR.iterdir():
            if f.is_file():
                shutil.copy2(f, export_dir / f.name)

    # Fix CppPythonAgent.cfg to point to actual exe name
    agent_cfg = export_dir / "CppPythonAgent.cfg"
    if agent_cfg.exists():
        cfg_text = agent_cfg.read_text()
        cfg_text = cfg_text.replace("CPPExampleBot.exe", EXE_PATH.name)
        agent_cfg.write_text(cfg_text)

    # Copy exe
    if EXE_PATH.exists():
        shutil.copy2(EXE_PATH, export_dir / EXE_PATH.name)

    # Copy DLLs
    for dll in EXE_PATH.parent.glob("*.dll"):
        shutil.copy2(dll, export_dir / dll.name)

    # Copy python runtime files
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


def rebuild_bot():
    """Run build.bat and return the output."""
    build_bat = ROCKET_LEAGUE_DIR / "build.bat"
    if not build_bat.exists():
        return {"ok": False, "error": "build.bat not found"}
    try:
        result = subprocess.run(
            ["cmd", "/c", str(build_bat)],
            capture_output=True,
            text=True,
            cwd=str(ROCKET_LEAGUE_DIR),
            timeout=300,
        )
        success = result.returncode == 0
        output = result.stdout + result.stderr
        lines = output.strip().splitlines()
        if len(lines) > 100:
            lines = lines[-100:]
        return {
            "ok": success,
            "returncode": result.returncode,
            "output": "\n".join(lines),
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "Build timed out (5 min)"}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def launch_test_game(bot_name: str, bot_mgr: BotManager):
    """Export bot to temp dir and launch RLBot match against itself."""
    config = bot_mgr.get_bot_config(bot_name)
    gamemode = config.get("gamemode", "1v1")
    num_participants = {"1v1": 2, "2v2": 4, "3v3": 6}.get(gamemode, 2)

    # Export to a temp directory
    export_dir = ROCKET_LEAGUE_DIR / "rlbot_test_export"
    if export_dir.exists():
        shutil.rmtree(export_dir)

    result = build_for_rlbot(bot_name, export_path=str(export_dir))
    if not result.get("ok"):
        return result

    # Generate rlbot.cfg with correct participant count
    lines = [
        "[RLBot Configuration]",
        "",
        "[Team Configuration]",
        "",
        "[Match Configuration]",
        f"num_participants = {num_participants}",
        "game_mode = Soccer",
        "game_map = Mannfield",
        "",
        "[Mutator Configuration]",
        "Match Length = Unlimited",
        "",
        "[Participant Configuration]",
    ]

    for i in range(num_participants):
        lines.append(f"participant_config_{i} = CppPythonAgent.cfg")
    lines.append("")
    for i in range(num_participants):
        lines.append(f"participant_team_{i} = {i % 2}")
    lines.append("")
    for i in range(num_participants):
        lines.append(f"participant_type_{i} = rlbot")
    lines.append("")
    for i in range(num_participants):
        lines.append(f"participant_bot_skill_{i} = 1.0")

    cfg_path = export_dir / "rlbot.cfg"
    cfg_path.write_text("\n".join(lines))

    # Check RLBot Python exists
    if not RLBOT_PYTHON.exists():
        return {"ok": False, "error": f"RLBot Python not found at {RLBOT_PYTHON}"}

    try:
        subprocess.Popen(
            [str(RLBOT_PYTHON), "-m", "rlbot.runner", str(cfg_path)],
            cwd=str(export_dir),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return {"ok": True, "gamemode": gamemode, "path": str(export_dir)}
    except Exception as e:
        return {"ok": False, "error": str(e)}


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
                max_pts = int(qs.get("max_points", ["0"])[0])
                self._json(store.get_json(since, max_points=max_pts))
            elif path == "/api/status":
                self._json(manager.get_status())
            elif path == "/api/log":
                self._json({"lines": manager.log_lines[-50:]})
            elif path == "/api/checkpoints":
                bot = qs.get("bot", [bot_mgr.current_bot])[0]
                self._json(bot_mgr.scan_checkpoints(bot))
            elif path == "/api/bots":
                self._json({"bots": bot_mgr.list_bots(), "current": bot_mgr.current_bot})
            elif path == "/api/bots/config":
                bot = qs.get("bot", [bot_mgr.current_bot])[0]
                self._json(bot_mgr.get_bot_config(bot))
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
                config = body.get("config", None)
                self._json(bot_mgr.create_bot(name, config=config))
            elif self.path == "/api/bots/select":
                name = body.get("name", "")
                bot_mgr.current_bot = name
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
                    bots = bot_mgr.list_bots()
                    bot_mgr.current_bot = bots[0]["name"] if bots else "default"
                    store.set_log_path(bot_mgr.get_metrics_path(bot_mgr.current_bot))
                    store.load_from_disk()
                self._json(result)
            elif self.path == "/api/bots/config":
                bot = body.get("bot", bot_mgr.current_bot)
                config = body.get("config", {})
                self._json(bot_mgr.save_bot_config(bot, config))
            elif self.path == "/api/build-rlbot":
                bot = body.get("bot", bot_mgr.current_bot)
                export_path = body.get("path", None)
                self._json(build_for_rlbot(bot, export_path=export_path))
            elif self.path == "/api/rebuild":
                self._json(rebuild_bot())
            elif self.path == "/api/test-game":
                bot = body.get("bot", bot_mgr.current_bot)
                self._json(launch_test_game(bot, bot_mgr))
            elif self.path == "/api/open-rewards":
                self._json(open_reward_file())
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

    _kill_orphan_training()

    bot_mgr = BotManager()
    store = MetricStore()

    store.set_log_path(bot_mgr.get_metrics_path(bot_mgr.current_bot))
    store.load_from_disk()

    manager = TrainingManager(store, bot_mgr)

    server = http.server.ThreadingHTTPServer(
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

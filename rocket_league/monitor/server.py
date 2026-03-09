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
import datetime
import http.server
import json
import mimetypes
import os
import re
import shutil
import socket
import stat
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
# Server activity log (shown in terminal + dashboard)
# ---------------------------------------------------------------------------

_activity_log = []
_activity_lock = threading.Lock()


def _log(action: str, message: str, ok: bool = True):
    """Log an action to both the terminal and the activity log."""
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    status = "OK" if ok else "FAIL"
    print(f"[{ts}] [{action}] {message} [{status}]")
    sys.stdout.flush()
    with _activity_lock:
        _activity_log.append({
            "time": ts,
            "action": action,
            "message": message,
            "ok": ok,
            "timestamp": time.time(),
        })
        # Keep last 100 entries
        if len(_activity_log) > 100:
            _activity_log.pop(0)

# ---------------------------------------------------------------------------
# Task runner (background tasks with streamable console output)
# ---------------------------------------------------------------------------

class TaskRunner:
    """Runs long operations (rebuild, build-rlbot, test-game) in background threads."""

    def __init__(self):
        self._lock = threading.Lock()
        self._tasks = {}

    def start_task(self, task_id: str, func, *args, **kwargs) -> dict:
        with self._lock:
            if task_id in self._tasks and self._tasks[task_id]["status"] == "running":
                return {"ok": False, "error": f"Task '{task_id}' is already running"}
            task = {
                "status": "running",
                "output_lines": [],
                "result": None,
                "started": time.time(),
            }
            self._tasks[task_id] = task

        def _run():
            try:
                result = func(task, *args, **kwargs)
                with self._lock:
                    task["result"] = result
                    task["status"] = "done"
            except Exception as e:
                with self._lock:
                    task["result"] = {"ok": False, "error": str(e)}
                    task["status"] = "failed"

        threading.Thread(target=_run, daemon=True).start()
        return {"ok": True, "task_id": task_id}

    def get_status(self, task_id: str, since_line: int = 0) -> dict:
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return {"ok": False, "error": "Unknown task"}
            lines = task["output_lines"][since_line:]
            return {
                "ok": True,
                "status": task["status"],
                "lines": lines,
                "total_lines": len(task["output_lines"]),
                "result": task["result"],
            }


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
    "ball_speed":           re.compile(r"Game/Ball Speed:\s*([\d.,e+-]+)"),
    "player_boost_usage":   re.compile(r"Player/Boost Usage:\s*([\d.,e+-]+)"),
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
            try:
                self._log_file.flush()
                self._log_file.close()
            except Exception:
                pass
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

        def _on_rm_error(func, path, exc_info):
            """Handle read-only or locked files on Windows."""
            try:
                os.chmod(path, stat.S_IWRITE)
                func(path)
            except Exception:
                pass

        # Retry up to 3 times for Windows file locking
        last_err = None
        for attempt in range(3):
            try:
                shutil.rmtree(bot_dir, onerror=_on_rm_error)
                return {"ok": True}
            except Exception as e:
                last_err = e
                time.sleep(0.2 * (attempt + 1))

        return {"ok": False, "error": str(last_err)}

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

    # --- Notes ---

    def get_notes(self, bot_name: str) -> list:
        notes_path = CHECKPOINTS_DIR / bot_name / "notes.json"
        if notes_path.exists():
            try:
                with open(notes_path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return []

    def add_note(self, bot_name: str, timestep: int, text: str) -> dict:
        notes = self.get_notes(bot_name)
        note = {
            "id": f"note_{int(time.time() * 1000)}",
            "timestep": timestep,
            "text": text,
            "created": datetime.datetime.now().isoformat(),
        }
        notes.append(note)
        notes_path = CHECKPOINTS_DIR / bot_name / "notes.json"
        notes_path.parent.mkdir(parents=True, exist_ok=True)
        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=2)
        _log("NOTE", f"Added note at {timestep:,} ts: {text[:50]}")
        return {"ok": True, "note": note}

    def delete_note(self, bot_name: str, note_id: str) -> dict:
        notes = self.get_notes(bot_name)
        notes = [n for n in notes if n["id"] != note_id]
        notes_path = CHECKPOINTS_DIR / bot_name / "notes.json"
        with open(notes_path, "w") as f:
            json.dump(notes, f, indent=2)
        _log("NOTE", f"Deleted note {note_id}")
        return {"ok": True}


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

def _check_source_modified():
    """Check if source files have been modified since the exe was last built."""
    if not EXE_PATH.exists():
        return {"modified": True, "changed_files": ["(exe not found)"],
                "reason": "Exe not found — needs initial build"}
    exe_mtime = EXE_PATH.stat().st_mtime
    changed = []
    for src in [SRC_DIR / "main.cpp", SRC_DIR / "CustomRewards.h"]:
        if src.exists() and src.stat().st_mtime > exe_mtime:
            changed.append(src.name)
    return {"modified": len(changed) > 0, "changed_files": changed}


def open_reward_file():
    """Open main.cpp and CustomRewards.h in the default editor."""
    main_cpp = SRC_DIR / "main.cpp"
    custom_h = SRC_DIR / "CustomRewards.h"
    opened = []
    errors = []

    for f in (main_cpp, custom_h):
        if f.exists():
            try:
                os.startfile(str(f))
                opened.append(f.name)
            except Exception as e:
                errors.append(f"{f.name}: {e}")
        else:
            errors.append(f"{f.name} not found")

    if not opened:
        return {"ok": False, "error": "; ".join(errors)}
    return {"ok": True, "opened": opened}


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


def build_for_rlbot(task, bot_name: str, export_path: str = None):
    """Export the latest checkpoint as an RLBot-ready package.

    Layout isolates Python 3.13 (C++ exe) from Python 3.11 (RLBot):
      export_dir/
        CppPythonAgent.py, .cfg   ← RLBot Python 3.11 agent
        appearance.cfg, port.cfg  ← RLBot config files
        rlbot.cfg                 ← match config (test-game only)
        bot_bin/                  ← C++ exe + Python 3.13 runtime
          RocketLeagueStrategyBot.exe
          *.dll  (including python313.dll)
          *.pyd, python313._pth
          POLICY.lt, CRITIC.lt …  ← model files (exe loads from its own dir)
    """
    steps = []  # Track steps for frontend display

    def _task_log(msg):
        """Append a step message to the task console output."""
        if task:
            task["output_lines"].append(msg)
    _log("BUILD-RLBOT", f"Starting RLBot export for bot '{bot_name}'...")
    _task_log(f"Starting RLBot export for bot '{bot_name}'...")

    bot_dir = CHECKPOINTS_DIR / bot_name
    if not bot_dir.exists():
        _log("BUILD-RLBOT", f"Bot '{bot_name}' not found", ok=False)
        return {"ok": False, "error": f"Bot '{bot_name}' not found"}

    # Find latest checkpoint
    _task_log("Finding latest checkpoint...")
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
        _log("BUILD-RLBOT", "No checkpoint with a saved model found", ok=False)
        return {"ok": False, "error": "No checkpoint with a saved model found", "steps": steps}

    steps.append(f"Found checkpoint: {latest.name} ({latest_ts:,} timesteps)")
    _log("BUILD-RLBOT", steps[-1])
    _task_log(steps[-1])

    export_dir = Path(export_path)
    export_dir.mkdir(parents=True, exist_ok=True)

    bot_bin = export_dir / "bot_bin"
    bot_bin.mkdir(parents=True, exist_ok=True)

    steps.append(f"Export folder: {export_dir}")
    _log("BUILD-RLBOT", steps[-1])
    _task_log(steps[-1])

    # --- Model files → bot_bin/ ---
    model_count = 0
    for f in latest.iterdir():
        if f.suffix == ".lt":
            shutil.copy2(f, bot_bin / f.name)
            model_count += 1
    steps.append(f"Copied {model_count} model file(s) → bot_bin/")
    _log("BUILD-RLBOT", steps[-1])
    _task_log(steps[-1])

    # --- RLBot template files → root ---
    template_count = 0
    if RLBOT_TEMPLATE_DIR.exists():
        for f in RLBOT_TEMPLATE_DIR.iterdir():
            if f.is_file():
                shutil.copy2(f, export_dir / f.name)
                template_count += 1
    steps.append(f"Copied {template_count} RLBot template file(s)")
    _log("BUILD-RLBOT", steps[-1])
    _task_log(steps[-1])

    # Patch CppPythonAgent.cfg → point to bot_bin\<exe>
    agent_cfg = export_dir / "CppPythonAgent.cfg"
    if agent_cfg.exists():
        cfg_text = agent_cfg.read_text()
        cfg_text = cfg_text.replace("CPPExampleBot.exe",
                                    f"bot_bin\\{EXE_PATH.name}")
        agent_cfg.write_text(cfg_text)
        steps.append("Patched CppPythonAgent.cfg → bot_bin/")
        _log("BUILD-RLBOT", steps[-1])
        _task_log(steps[-1])

    # --- C++ exe → bot_bin/ ---
    if EXE_PATH.exists():
        shutil.copy2(EXE_PATH, bot_bin / EXE_PATH.name)
        steps.append(f"Copied {EXE_PATH.name} → bot_bin/")
        _log("BUILD-RLBOT", steps[-1])
        _task_log(steps[-1])

    # --- DLLs (torch, python313, etc.) → bot_bin/ ---
    dll_count = 0
    for dll in EXE_PATH.parent.glob("*.dll"):
        shutil.copy2(dll, bot_bin / dll.name)
        dll_count += 1
        if dll_count % 5 == 0:
            _task_log(f"  Copying DLLs... ({dll_count})")
    steps.append(f"Copied {dll_count} DLL(s) → bot_bin/")
    _log("BUILD-RLBOT", steps[-1])
    _task_log(steps[-1])

    # --- Python 3.13 runtime (.pyd + .pth) → bot_bin/ ---
    rt_count = 0
    pth_file = EXE_PATH.parent / "python313._pth"
    if pth_file.exists():
        shutil.copy2(pth_file, bot_bin / pth_file.name)
        rt_count += 1
    for pyd in EXE_PATH.parent.glob("*.pyd"):
        shutil.copy2(pyd, bot_bin / pyd.name)
        rt_count += 1
    steps.append(f"Copied {rt_count} Python 3.13 runtime file(s) → bot_bin/")
    _log("BUILD-RLBOT", steps[-1])
    _task_log(steps[-1])

    total_files = sum(1 for f in export_dir.rglob("*") if f.is_file())
    _log("BUILD-RLBOT", f"Export complete! {total_files} files in {export_dir}")
    steps.append(f"Done! {total_files} files exported")
    _task_log(steps[-1])

    return {
        "ok": True,
        "path": str(export_dir),
        "checkpoint": latest.name,
        "files": [f.name for f in export_dir.iterdir()],
        "steps": steps,
    }


def rebuild_bot(task=None):  # task is passed by TaskRunner as first arg
    """Run build.bat and stream output in real-time to the server terminal."""
    _log("REBUILD", "Starting build...")
    build_bat = ROCKET_LEAGUE_DIR / "build.bat"
    if not build_bat.exists():
        _log("REBUILD", "build.bat not found", ok=False)
        return {"ok": False, "error": "build.bat not found"}
    try:
        t0 = time.time()
        proc = subprocess.Popen(
            ["cmd", "/c", str(build_bat)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(ROCKET_LEAGUE_DIR),
            bufsize=0,
        )

        output_lines = []
        while True:
            raw = proc.stdout.readline()
            if raw == b"" and proc.poll() is not None:
                break
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace").rstrip()
            output_lines.append(line)
            if task:
                task["output_lines"].append(line)
            if line:
                print(f"  [BUILD] {line}")
                sys.stdout.flush()
            # Track build stages via markers in build.bat
            if "===CONFIGURE_START===" in line:
                _log("REBUILD", "CMake configure starting...")
            elif "===CONFIGURE_RC=" in line:
                rc = line.split("=")[-1].rstrip("=")
                if rc != "0":
                    _log("REBUILD", f"CMake configure failed (rc={rc})", ok=False)
            elif "===BUILD_START===" in line:
                _log("REBUILD", "Compiling...")
            elif "===BUILD_RC=" in line:
                rc = line.split("=")[-1].rstrip("=")
                if rc == "0":
                    _log("REBUILD", "Compilation finished")

        proc.wait()
        elapsed = time.time() - t0
        success = proc.returncode == 0

        # Keep last 100 lines for the frontend
        if len(output_lines) > 100:
            output_lines = output_lines[-100:]

        if success:
            _log("REBUILD", f"Build succeeded in {elapsed:.1f}s")
        else:
            _log("REBUILD", f"Build FAILED (exit code {proc.returncode}) in {elapsed:.1f}s", ok=False)
        return {
            "ok": success,
            "returncode": proc.returncode,
            "output": "\n".join(output_lines),
            "elapsed": round(elapsed, 1),
        }
    except Exception as e:
        _log("REBUILD", f"Build error: {e}", ok=False)
        return {"ok": False, "error": str(e)}


class TestGameState:
    """Tracks running test game processes for cleanup."""

    def __init__(self):
        self._lock = threading.Lock()
        self.exe_proc = None
        self.rlbot_proc = None
        self.running = False

    def set_procs(self, exe_proc, rlbot_proc=None):
        with self._lock:
            self.exe_proc = exe_proc
            if rlbot_proc is not None:
                self.rlbot_proc = rlbot_proc
            self.running = True

    def stop(self):
        """Kill all test game processes and clean up."""
        with self._lock:
            if not self.running:
                return {"ok": False, "error": "No test game running"}

            killed = []
            for name, proc in [("RLBot", self.rlbot_proc), ("Bot exe", self.exe_proc)]:
                if proc and proc.poll() is None:
                    try:
                        # Use taskkill /T for tree kill (catches child processes)
                        subprocess.run(
                            ["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                            capture_output=True, timeout=5,
                        )
                        killed.append(name)
                    except Exception:
                        try:
                            proc.kill()
                            killed.append(f"{name} (force)")
                        except Exception:
                            pass

            self.exe_proc = None
            self.rlbot_proc = None
            self.running = False

        # Kill Rocket League itself
        try:
            result = subprocess.run(
                ["taskkill", "/F", "/IM", "RocketLeague.exe"],
                capture_output=True, timeout=5,
            )
            if result.returncode == 0:
                killed.append("Rocket League")
        except Exception:
            pass

        # Clean up export directory
        export_dir = ROCKET_LEAGUE_DIR / "rlbot_test_export"
        if export_dir.exists():
            def _force_rm(func, path, exc_info):
                try:
                    os.chmod(path, stat.S_IWRITE)
                    func(path)
                except Exception:
                    pass
            try:
                shutil.rmtree(export_dir, onerror=_force_rm)
                killed.append("Cleaned up rlbot_test_export/")
            except Exception as e:
                killed.append(f"Cleanup warning: {e}")

        # Brief pause for file handle release
        time.sleep(0.3)

        msg = ", ".join(killed) if killed else "No active processes"
        _log("TEST-GAME", f"Stopped: {msg}")
        return {"ok": True, "stopped": killed}

    def is_running(self):
        with self._lock:
            if not self.running:
                return False
            # Check if processes are still alive
            alive = False
            if self.exe_proc and self.exe_proc.poll() is None:
                alive = True
            if self.rlbot_proc and self.rlbot_proc.poll() is None:
                alive = True
            if not alive:
                self.running = False
            return self.running


def _kill_previous_test_game(training_pid: int = None):
    """Kill any leftover processes from a previous test game."""
    exe_name = EXE_PATH.name  # RocketLeagueStrategyBot.exe

    # Kill lingering C++ bot exe (holds port 42653 open)
    # Skip the active training process (if any) so we don't disrupt it.
    try:
        result = subprocess.run(
            ["tasklist", "/FI", f"IMAGENAME eq {exe_name}", "/FO", "CSV", "/NH"],
            capture_output=True, text=True, timeout=5,
        )
        for line in result.stdout.strip().splitlines():
            if exe_name.lower() in line.lower():
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    pid_str = parts[1].strip('"')
                    if training_pid and pid_str == str(training_pid):
                        continue  # don't kill the active training process
                    _log("TEST-GAME", f"Killing leftover {exe_name} (PID {pid_str})")
                    subprocess.run(["taskkill", "/F", "/PID", pid_str],
                                   capture_output=True, timeout=5)
    except Exception:
        pass

    # Kill any leftover RLBot python processes
    try:
        subprocess.run(
            ["taskkill", "/F", "/IM", "python.exe", "/FI", f"MODULES eq rlbot"],
            capture_output=True, timeout=5,
        )
    except Exception:
        pass

    # Brief pause for port/file handle release
    time.sleep(0.3)


def launch_test_game(task, bot_name: str, bot_mgr: BotManager,
                     training_mgr: "TrainingManager" = None,
                     test_game_state=None):
    """Launch a test game: run exe from build/ (no 4 GB DLL copy), RLBot from a lightweight export dir."""
    steps = []

    def _task_log(msg):
        if task:
            task["output_lines"].append(msg)

    _log("TEST-GAME", f"Launching test game for bot '{bot_name}'...")
    _task_log(f"Launching test game for bot '{bot_name}'...")

    config = bot_mgr.get_bot_config(bot_name)
    gamemode = config.get("gamemode", "1v1")
    num_participants = {"1v1": 2, "2v2": 4, "3v3": 6}.get(gamemode, 2)
    steps.append(f"Gamemode: {gamemode} ({num_participants} players)")
    _log("TEST-GAME", steps[-1])
    _task_log(steps[-1])

    # Kill any leftover processes from a previous test game
    training_pid = training_mgr.proc.pid if training_mgr and training_mgr.proc else None
    _kill_previous_test_game(training_pid=training_pid)
    steps.append("Cleaned up previous test game processes")
    _log("TEST-GAME", steps[-1])
    _task_log(steps[-1])

    # Find latest checkpoint with a model
    bot_dir = CHECKPOINTS_DIR / bot_name
    if not bot_dir.exists():
        msg = f"Bot '{bot_name}' not found"
        _log("TEST-GAME", msg, ok=False)
        return {"ok": False, "error": msg, "steps": steps}

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
        msg = "No checkpoint with a saved model found"
        _log("TEST-GAME", msg, ok=False)
        return {"ok": False, "error": msg, "steps": steps}

    steps.append(f"Checkpoint: {latest.name} ({latest_ts:,} ts)")
    _log("TEST-GAME", steps[-1])
    _task_log(steps[-1])

    # Check exe exists in build/
    if not EXE_PATH.exists():
        msg = f"Bot exe not found at {EXE_PATH}. Run build.bat first."
        _log("TEST-GAME", msg, ok=False)
        return {"ok": False, "error": msg, "steps": steps}

    # Copy model files into build/ so the exe can find them (it loads
    # from GetExecutableDirectory(), i.e. its own dir).
    build_dir = EXE_PATH.parent
    model_count = 0
    for f in latest.iterdir():
        if f.suffix == ".lt":
            shutil.copy2(f, build_dir / f.name)
            model_count += 1
    steps.append(f"Copied {model_count} model file(s) to build/")
    _log("TEST-GAME", steps[-1])
    _task_log(steps[-1])

    # Set up lightweight RLBot config directory (just .py/.cfg files, ~50 KB)
    export_dir = ROCKET_LEAGUE_DIR / "rlbot_test_export"
    export_dir.mkdir(parents=True, exist_ok=True)

    # Copy RLBot template files
    if RLBOT_TEMPLATE_DIR.exists():
        for f in RLBOT_TEMPLATE_DIR.iterdir():
            if f.is_file():
                shutil.copy2(f, export_dir / f.name)

    # Blank out cpp_executable_path — we launch the exe ourselves
    agent_cfg = export_dir / "CppPythonAgent.cfg"
    if agent_cfg.exists():
        cfg_text = agent_cfg.read_text()
        cfg_text = cfg_text.replace("CPPExampleBot.exe", "")
        agent_cfg.write_text(cfg_text)

    # Generate rlbot.cfg
    lines = [
        "[RLBot Configuration]", "",
        "[Team Configuration]", "",
        "[Match Configuration]",
        f"num_participants = {num_participants}",
        "game_mode = Soccer", "game_map = Mannfield", "",
        "[Mutator Configuration]", "Match Length = Unlimited", "",
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

    (export_dir / "rlbot.cfg").write_text("\n".join(lines))
    steps.append("Generated RLBot config")
    _log("TEST-GAME", steps[-1])
    _task_log(steps[-1])

    # Check RLBot Python exists
    if not RLBOT_PYTHON.exists():
        msg = f"RLBot Python not found at {RLBOT_PYTHON}"
        _log("TEST-GAME", msg, ok=False)
        return {"ok": False, "error": msg, "steps": steps}

    # ---------------------------------------------------------------
    # Launch exe from build/ + RLBot from export_dir in background.
    # The exe already has all DLLs in build/ — no need to copy 4+ GB.
    # ---------------------------------------------------------------

    dll_dir = RLBOT_PYTHON.parent / "Lib" / "site-packages" / "rlbot" / "dll"

    def _launch_background():
        """Background thread: launch exe, wait for TCP, launch RLBot."""
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env["PYTHONNOUSERSITE"] = "1"
        env.pop("PYTHONPATH", None)

        try:
            exe_proc = subprocess.Popen(
                [str(EXE_PATH), "--bot", bot_name, "-dll-path", str(dll_dir)],
                cwd=str(ROCKET_LEAGUE_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                bufsize=0,
            )
        except Exception as e:
            _log("TEST-GAME", f"Failed to launch bot exe: {e}", ok=False)
            _task_log(f"FAILED: {e}")
            return

        _log("TEST-GAME", f"Bot exe launched (PID {exe_proc.pid})")
        _task_log(f"Bot exe launched (PID {exe_proc.pid})")
        if test_game_state:
            test_game_state.set_procs(exe_proc)

        def _stream_exe(p):
            for raw in iter(p.stdout.readline, b""):
                line = raw.decode("utf-8", errors="replace").rstrip()
                if line:
                    print(f"  [BOT-EXE] {line}")
                    sys.stdout.flush()
                    _task_log(f"[BOT-EXE] {line}")
            rc = p.wait()
            _log("TEST-GAME", f"Bot exe exited (code {rc})", ok=(rc == 0))
            _task_log(f"Bot exe exited (code {rc})")

        threading.Thread(target=_stream_exe, args=(exe_proc,), daemon=True).start()

        # Wait for TCP server on port 42653 (up to 15s)
        _log("TEST-GAME", "Waiting for bot exe TCP server on port 42653...")
        _task_log("Waiting for bot exe TCP server on port 42653...")
        connected = False
        for attempt in range(15):
            if exe_proc.poll() is not None:
                _log("TEST-GAME", f"Bot exe crashed (exit code {exe_proc.returncode})", ok=False)
                _task_log(f"Bot exe crashed (exit code {exe_proc.returncode})")
                return
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(0.5)
                s.connect(("127.0.0.1", 42653))
                s.close()
                connected = True
                break
            except (ConnectionRefusedError, socket.timeout, OSError):
                time.sleep(1)

        if not connected:
            _log("TEST-GAME", "Bot exe did not start TCP server within 15s", ok=False)
            _task_log("Bot exe did not start TCP server within 15s")
            exe_proc.terminate()
            return

        _log("TEST-GAME", "Bot exe TCP server ready on port 42653")
        _task_log("Bot exe TCP server ready on port 42653")

        # Start RLBot (connects to our already-running exe)
        try:
            rlbot_proc = subprocess.Popen(
                [str(RLBOT_PYTHON), "-m", "rlbot.runner"],
                cwd=str(export_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
            )

            if test_game_state:
                test_game_state.set_procs(exe_proc, rlbot_proc)

            def _stream_rlbot(p):
                for raw in iter(p.stdout.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        print(f"  [RLBOT] {line}")
                        sys.stdout.flush()
                        _task_log(f"[RLBOT] {line}")
                rc = p.wait()
                _log("TEST-GAME", f"RLBot process exited (code {rc})", ok=(rc == 0))
                _task_log(f"RLBot process exited (code {rc})")
                if test_game_state and test_game_state.running:
                    test_game_state.stop()

            threading.Thread(target=_stream_rlbot, args=(rlbot_proc,), daemon=True).start()
            _log("TEST-GAME", f"RLBot match launched ({gamemode}) — PID {rlbot_proc.pid}")
            _task_log(f"RLBot match launched ({gamemode}) — PID {rlbot_proc.pid}")

        except Exception as e:
            _log("TEST-GAME", f"Failed to launch RLBot: {e}", ok=False)
            _task_log(f"Failed to launch RLBot: {e}")
            exe_proc.terminate()

    # Run launch sequence directly (already in TaskRunner's background thread).
    # The exe + RLBot stdout streaming threads keep running after we return.
    _launch_background()
    steps.append("Launch sequence complete")
    _log("TEST-GAME", steps[-1])
    _task_log(steps[-1])

    return {"ok": True, "gamemode": gamemode, "path": str(export_dir), "steps": steps}


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


def make_handler(store: MetricStore, manager: TrainingManager, bot_mgr: BotManager,
                  task_runner: TaskRunner = None, test_game_state: TestGameState = None):
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
                status = manager.get_status()
                status["test_game_running"] = test_game_state.is_running() if test_game_state else False
                self._json(status)
            elif path == "/api/task-status":
                task_id = qs.get("id", [""])[0]
                since = int(qs.get("since", ["0"])[0])
                self._json(task_runner.get_status(task_id, since_line=since) if task_runner else {"ok": False, "error": "No task runner"})
            elif path == "/api/log":
                self._json({"lines": manager.log_lines[-50:]})
            elif path == "/api/checkpoints":
                bot = qs.get("bot", [bot_mgr.current_bot])[0]
                self._json(bot_mgr.scan_checkpoints(bot))
            elif path == "/api/activity":
                since = float(qs.get("since", ["0"])[0])
                with _activity_lock:
                    entries = [e for e in _activity_log if e["timestamp"] > since]
                self._json({"entries": entries})
            elif path == "/api/bots":
                self._json({"bots": bot_mgr.list_bots(), "current": bot_mgr.current_bot})
            elif path == "/api/bots/config":
                bot = qs.get("bot", [bot_mgr.current_bot])[0]
                self._json(bot_mgr.get_bot_config(bot))
            elif path == "/api/source-status":
                self._json(_check_source_modified())
            elif path == "/api/notes":
                bot = qs.get("bot", [bot_mgr.current_bot])[0]
                self._json({"notes": bot_mgr.get_notes(bot)})
            else:
                self.send_response(404)
                self.end_headers()

        def do_POST(self):
            body = self._parse_body()

            if self.path == "/api/start":
                bot = body.get("bot", bot_mgr.current_bot)
                result = manager.start(bot_name=bot)
                _log("TRAINING", f"Start training '{bot}'" + (" - OK" if result.get("ok") else f" - {result.get('error', '?')}"), ok=result.get("ok", False))
                self._json(result)
            elif self.path == "/api/stop":
                result = manager.save_and_stop()
                _log("TRAINING", "Save & stop requested", ok=result.get("ok", False))
                self._json(result)
            elif self.path == "/api/kill":
                result = manager.kill()
                _log("TRAINING", "Process killed", ok=result.get("ok", False))
                self._json(result)
            elif self.path == "/api/bots/create":
                name = body.get("name", "")
                config = body.get("config", None)
                result = bot_mgr.create_bot(name, config=config)
                _log("BOT", f"Create bot '{name}'" + (" - OK" if result.get("ok") else f" - {result.get('error', '?')}"), ok=result.get("ok", False))
                self._json(result)
            elif self.path == "/api/bots/select":
                name = body.get("name", "")
                bot_mgr.current_bot = name
                store.close()
                store.set_log_path(bot_mgr.get_metrics_path(name))
                store.load_from_disk()
                _log("BOT", f"Switched to bot '{name}'")
                self._json({"ok": True, "bot": name})
            elif self.path == "/api/bots/delete":
                name = body.get("name", "")
                # Don't allow deleting while training is running for this bot
                if manager.status == "running" and bot_mgr.current_bot == name:
                    _log("BOT", f"Cannot delete '{name}' while training is running", ok=False)
                    self._json({"ok": False, "error": "Cannot delete bot while training is running. Stop training first."})
                    return
                was_current = (name == bot_mgr.current_bot)
                if was_current:
                    store.close()
                    store._log_path = None  # prevent reopening during delete
                    time.sleep(0.1)  # let Windows release file handles
                result = bot_mgr.delete_bot(name)
                _log("BOT", f"Delete bot '{name}'" + (" - OK" if result.get("ok") else f" - {result.get('error', '?')}"), ok=result.get("ok", False))
                if result.get("ok") and was_current:
                    bots = bot_mgr.list_bots()
                    bot_mgr.current_bot = bots[0]["name"] if bots else "default"
                    store.set_log_path(bot_mgr.get_metrics_path(bot_mgr.current_bot))
                    store.load_from_disk()
                self._json(result)
            elif self.path == "/api/bots/config":
                bot = body.get("bot", bot_mgr.current_bot)
                config = body.get("config", {})
                result = bot_mgr.save_bot_config(bot, config)
                _log("CONFIG", f"Saved config for '{bot}'", ok=result.get("ok", False))
                self._json(result)
            elif self.path == "/api/build-rlbot":
                bot = body.get("bot", bot_mgr.current_bot)
                export_path = body.get("path", None)
                # Folder picker must run before background task (blocking UI dialog)
                if not export_path:
                    picked = _pick_folder("Export RLBot Package -- Choose Folder")
                    if not picked:
                        _log("BUILD-RLBOT", "Export cancelled by user", ok=False)
                        self._json({"ok": False, "error": "Export cancelled"})
                        return
                    export_path = picked
                self._json(task_runner.start_task(
                    "build-rlbot", build_for_rlbot, bot, export_path))
            elif self.path == "/api/rebuild":
                self._json(task_runner.start_task("rebuild", rebuild_bot))
            elif self.path == "/api/test-game":
                bot = body.get("bot", bot_mgr.current_bot)
                self._json(task_runner.start_task(
                    "test-game", launch_test_game, bot, bot_mgr,
                    training_mgr=manager, test_game_state=test_game_state))
            elif self.path == "/api/stop-test-game":
                if test_game_state:
                    self._json(test_game_state.stop())
                else:
                    self._json({"ok": False, "error": "No test game state"})
            elif self.path == "/api/notes/add":
                bot = body.get("bot", bot_mgr.current_bot)
                timestep = body.get("timestep", 0)
                text = body.get("text", "").strip()
                if not text:
                    self._json({"ok": False, "error": "Note text is required"})
                else:
                    self._json(bot_mgr.add_note(bot, timestep, text))
            elif self.path == "/api/notes/delete":
                bot = body.get("bot", bot_mgr.current_bot)
                note_id = body.get("id", "")
                self._json(bot_mgr.delete_note(bot, note_id))
            elif self.path == "/api/open-rewards":
                _log("ACTION", "Opening main.cpp in editor")
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
    task_runner = TaskRunner()
    tg_state = TestGameState()

    server = http.server.ThreadingHTTPServer(
        ("0.0.0.0", args.port),
        make_handler(store, manager, bot_mgr, task_runner, tg_state),
    )
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    url = f"http://localhost:{args.port}"
    print()
    print("=" * 52)
    print("  GigaLearnCPP Training Monitor")
    print(f"  Dashboard: {url}")
    print(f"  Bots dir:  {CHECKPOINTS_DIR}")
    print("  Press Ctrl+C to shut down")
    print("=" * 52)
    print()
    _log("SERVER", f"Dashboard ready at {url}")

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

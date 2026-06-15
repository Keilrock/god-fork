"""InterCode (NL2Bash) GRPO rollout — tool-call variant for the PR1201 eval.

Adapted from god-env-one's intercode_env.py. We keep the LocalBashEnv + princeton
reward formula VERBATIM (so training reward == eval reward) but replace its ReAct
"Action N:" text protocol with the tool-call protocol the PR1201 eval uses:
`execute_bash` is a tool (like game_action); the model emits <tool_call> blocks we
parse back with the shared parser. GRPO trains the FIRST tool turn only (turn-0),
exactly like the PvP rollout — and turn-0's prompt (system + query + tools) is
format-identical to the evaluator's turn 0, so the trained tokens match eval.

/workspace conflict: intercode fs_3 manages /workspace (+/backup), which is where
our axolotl runtime lives. Rather than relocate the runtime, we DROP fs_3 from the
trainable set and remove /workspace,/backup from the wiped managed paths — so
reset() never touches the live runtime. We train fs_1 (/testbed), fs_2 (/system),
fs_4 (none); the model generalises to fs_3 at eval (same NL2Bash task family).
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

from trl.experimental.openenv import generate_rollout_completions

from _pvp_rollout.recording_chat_fn import extract_tool_calls, strip_think_tags
from _pvp_rollout.multi_env_rollout import make_reward_func  # generic env_rewards passthrough

_PVP_DBG = bool(os.environ.get("PVP_DBG"))

# --- evaluator-equivalent knobs (same env vars/defaults as eval_intercode) ---
DEFAULT_INTERCODE_DATA_ROOT = Path("/intercode_data")
DEFAULT_INTERCODE_FS_ROOT = Path("/intercode_fs")
DEFAULT_ACTION_TIMEOUT_SECONDS = 30
DEFAULT_OBS_TRUNCATE_CHARS = 350
DEFAULT_MAX_TURNS = int(os.environ.get("INTERCODE_MAX_TURNS", "10"))

DEFAULT_SCORING_MODE = "continuous"
VALID_SCORING_MODES = {"continuous", "binary"}
SCORING_MODE = os.getenv("INTERCODE_SCORING_MODE", DEFAULT_SCORING_MODE).strip().lower()
assert SCORING_MODE in VALID_SCORING_MODES, f"invalid INTERCODE_SCORING_MODE={SCORING_MODE!r}"

# MODIFIED vs upstream: /workspace + /backup (fs_3) excluded so reset() never wipes
# the axolotl runtime that lives under /workspace. fs_3 is dropped from USABLE_FS.
ALL_MANAGED_PATHS = ("/testbed", "/system")
PATHS_PER_FS: dict[int, tuple[str, ...]] = {
    1: ("/testbed",),
    2: ("/system",),
    4: (),  # filesystem-agnostic
}
USABLE_FS = (1, 2, 4)

INTERCODE_EXECUTE_TOOL_NAME = "execute_bash"
_EXECUTE_BASH_TOOL = {
    "type": "function",
    "function": {
        "name": INTERCODE_EXECUTE_TOOL_NAME,
        "description": (
            "Run a single Bash command on the system and see its output. Inspect the "
            "filesystem and build your answer across turns. When the task is complete, "
            "call this tool with command=\"submit\" to end and be scored."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The bash command to execute, or 'submit' to finish.",
                }
            },
            "required": ["command"],
        },
    },
}

_SYSTEM_PROMPT = (
    "You are interacting with a Bourne Shell (Bash) system to answer a question. "
    "Each turn, call the execute_bash tool with exactly one bash command; you will "
    "see its output and may issue more commands. Reason briefly if helpful, but you "
    "MUST call execute_bash every turn. When you have produced the answer / made the "
    "required filesystem changes, call execute_bash with command=\"submit\" to finish."
)


# --- managed-path safety guardrail (verbatim; now only checks /testbed,/system) ---
def _protected_runtime_dirs() -> set[str]:
    import sys
    candidates = [
        sys.prefix, sys.base_prefix, os.path.dirname(sys.executable or ""),
        os.getcwd(), os.path.dirname(os.path.abspath(__file__)),
    ]
    out: set[str] = set()
    for c in candidates:
        if not c:
            continue
        try:
            out.add(os.path.realpath(c))
        except OSError:
            pass
    return out


def _conflicting_protected_dirs(path: str) -> list[str]:
    path_real = os.path.realpath(path)
    prefix = path_real.rstrip("/") + os.sep
    return [p for p in _protected_runtime_dirs() if p == path_real or p.startswith(prefix)]


def _assert_path_safe_to_wipe(path: str) -> None:
    conflicts = _conflicting_protected_dirs(path)
    if conflicts:
        raise RuntimeError(
            f"refusing to rmtree managed path {path!r}: contains protected runtime {conflicts}"
        )


def _assert_no_managed_path_conflicts() -> None:
    conflicts = [(mp, p) for mp in ALL_MANAGED_PATHS for p in _conflicting_protected_dirs(mp)]
    if conflicts:
        details = "; ".join(f"{mp} contains {p}" for mp, p in conflicts)
        raise RuntimeError(f"InterCode managed paths overlap live runtime: {details}")


# --- NL2Bash assets (load only the usable fs variants) ---
def _load_data(data_root: Path) -> dict[int, list[dict]]:
    out: dict[int, list[dict]] = {}
    for fs in USABLE_FS:
        path = data_root / f"nl2bash_fs_{fs}.json"
        out[fs] = __import__("json").loads(path.read_text())
    return out


@dataclass(frozen=True)
class InterCodeAssets:
    data: dict[int, list[dict]]
    snapshot_root: Path

    @property
    def total_tasks(self) -> int:
        return sum(len(v) for v in self.data.values())


def load_intercode_assets(data_root=None, snapshot_root=None) -> InterCodeAssets:
    data_path = Path(data_root) if data_root is not None else Path(
        os.getenv("INTERCODE_DATA_ROOT", str(DEFAULT_INTERCODE_DATA_ROOT)))
    snapshot_path = Path(snapshot_root) if snapshot_root is not None else Path(
        os.getenv("INTERCODE_FS_ROOT", str(DEFAULT_INTERCODE_FS_ROOT)))
    if not data_path.exists():
        raise RuntimeError(f"NL2Bash data not found at {data_path}; image may be misbuilt")
    if not snapshot_path.exists():
        raise RuntimeError(f"InterCode fs snapshots not found at {snapshot_path}; image may be misbuilt")
    data = _load_data(data_path)
    total = sum(len(v) for v in data.values())
    if total <= 0:
        raise RuntimeError(f"intercode assets empty (data_path={data_path})")
    return InterCodeAssets(data=data, snapshot_root=snapshot_path)


# --- LocalBashEnv: VERBATIM port of eval_intercode.LocalBashEnv (reward 1:1) ---
class LocalBashEnv:
    def __init__(self, fs_version: int, entries: list[dict], snapshot_root: Path):
        self.fs_version = fs_version
        self.entries = entries
        self.managed_paths = PATHS_PER_FS[fs_version]
        self.snapshot_tar = snapshot_root / f"fs{fs_version}.tar"
        self.workdir = "/"
        self.observation = ""
        self.observation_eval = ""
        self.action_executed = False
        self.query = None
        self.gold = None
        self._snapshot_state = None
        self._agent_state = None
        self._eval_state = None

    def reset(self, index: int) -> str:
        record = self.entries[index]
        self.query = record["query"]
        self.gold = record.get("gold", "") or ""
        self.workdir = "/"
        self.observation = ""
        self.observation_eval = ""
        self._restore_fs()
        self._snapshot_state = self._capture_state()
        return self.query

    def _restore_fs(self) -> None:
        for p in ALL_MANAGED_PATHS:
            if os.path.exists(p):
                _assert_path_safe_to_wipe(p)
                shutil.rmtree(p, ignore_errors=True)
        if not self.managed_paths or not self.snapshot_tar.exists():
            return
        try:
            subprocess.run(["tar", "-xpf", str(self.snapshot_tar), "-C", "/"],
                           check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"failed to restore fs_{self.fs_version} snapshot: "
                f"{exc.stderr.decode('utf-8', errors='replace')}")

    def _capture_state(self) -> dict[str, tuple]:
        state: dict[str, tuple] = {}
        for root_path in self.managed_paths:
            if not os.path.exists(root_path):
                continue
            for cur, dirs, files in os.walk(root_path):
                for name in dirs:
                    full = os.path.join(cur, name)
                    try:
                        st = os.lstat(full)
                        state[full] = ("<DIR>", st.st_mode)
                    except OSError:
                        state[full] = ("<ERR>", 0)
                for name in files:
                    full = os.path.join(cur, name)
                    try:
                        if os.path.islink(full):
                            state[full] = ("<LINK>", os.readlink(full))
                        else:
                            h = hashlib.md5()
                            with open(full, "rb") as fh:
                                for chunk in iter(lambda: fh.read(65536), b""):
                                    h.update(chunk)
                            st = os.lstat(full)
                            state[full] = (h.hexdigest(), st.st_size)
                    except OSError:
                        state[full] = ("<ERR>", 0)
        return state

    @staticmethod
    def _simplify_path(current: str, changed: str) -> str:
        if not changed:
            return current
        if changed[0] == "/":
            current = ""
        path: list[str] = []
        for seg in (current + "/" + changed).split("/"):
            if seg == "..":
                if path:
                    path.pop()
            elif seg and seg != ".":
                path.append(seg)
        return "/" + "/".join(path)

    def _exec_action(self, action: str) -> None:
        is_cd = action.startswith("cd")
        new_path = None
        if is_cd and "cd " in action:
            cd_arg = action[action.index("cd ") + 3:].strip()
            new_path = self._simplify_path(self.workdir, cd_arg)
            action = f"cd {new_path}"
        try:
            res = subprocess.run(["/bin/bash", "-c", action],
                                 cwd="/" if is_cd else (self.workdir or "/"),
                                 capture_output=True, timeout=DEFAULT_ACTION_TIMEOUT_SECONDS)
            stdout = res.stdout.decode("utf-8", errors="replace")
            stderr = res.stderr.decode("utf-8", errors="replace")
            self.observation = stdout + (stderr if not stdout else "")
            self.action_executed = res.returncode == 0
            if is_cd and self.action_executed and new_path is not None:
                self.workdir = new_path
        except subprocess.TimeoutExpired:
            self.observation = "Command timed out"
            self.action_executed = False
        except Exception:
            self.observation = "Malformed command"
            self.action_executed = False

    def step(self, action: str) -> tuple[str, float, bool, dict]:
        if action.startswith("submit"):
            reward, info = self._get_reward()
            info["action_executed"] = True
            return self.observation, reward, True, info
        self._exec_action(action)
        return self.observation, 0.0, False, {"action_executed": self.action_executed}

    def _get_reward(self) -> tuple[float, dict]:
        self._agent_state = self._capture_state()
        self._restore_fs()
        gold_obs = ""
        corrupt_gold = False
        if self.gold:
            try:
                res = subprocess.run(["/bin/bash", "-c", self.gold], cwd="/",
                                     capture_output=True, timeout=DEFAULT_ACTION_TIMEOUT_SECONDS)
                gold_obs = (res.stdout.decode("utf-8", errors="replace")
                            + res.stderr.decode("utf-8", errors="replace"))
            except Exception:
                corrupt_gold = True
        self.observation_eval = gold_obs
        self._eval_state = self._capture_state()

        snapshot = self._snapshot_state or {}
        agent_changed = self._changed_paths(snapshot, self._agent_state or {})
        eval_changed = self._changed_paths(snapshot, self._eval_state or {})
        diff_miss = eval_changed - agent_changed
        diff_extra = agent_changed - eval_changed
        diff_same = agent_changed & eval_changed
        common_changes_total = len(diff_same)
        common_changes_correct = sum(
            1 for path in diff_same
            if (self._agent_state or {}).get(path) == (self._eval_state or {}).get(path))
        agent_obs = self.observation or ""
        gold_obs = self.observation_eval or ""

        p1 = round(0.33 * (1 - math.erf(len(diff_miss) + len(diff_extra))), 2)
        p2 = round(0.33 * (common_changes_correct / common_changes_total), 2) if common_changes_total else 0.33
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            vect = TfidfVectorizer()
            tfidf = vect.fit_transform([agent_obs, gold_obs])
            sim = float((tfidf * tfidf.T).toarray()[0][1])
        except Exception:
            sim = 1.0 if agent_obs == gold_obs else 0.0
        p3 = round(0.33 * sim, 2)
        continuous_reward = 0.01 + p1 + p2 + p3

        fs_pass = (len(diff_miss) == 0) and (len(diff_extra) == 0)
        content_pass = (common_changes_total == 0) or (common_changes_correct == common_changes_total)
        answer_pass = " ".join(agent_obs.split()) == " ".join(gold_obs.split())
        binary_reward = 1.0 if (fs_pass and content_pass and answer_pass) else 0.0
        reward = binary_reward if SCORING_MODE == "binary" else continuous_reward
        return reward, {"continuous_reward": continuous_reward, "binary_reward": binary_reward,
                        "corrupt_gold": corrupt_gold}

    @staticmethod
    def _changed_paths(before: dict, after: dict) -> set:
        keys = set(before) | set(after)
        return {k for k in keys if before.get(k) != after.get(k)}


# --- tool-call episode (replaces ReAct); trains turn-0 only ---
_RENDER_TOK = None
_CHECKED_PATHS = False


def _render_tokenizer(trainer):
    global _RENDER_TOK
    if _RENDER_TOK is not None:
        return _RENDER_TOK
    tok = trainer.processing_class
    _RENDER_TOK = tok
    try:
        from transformers import AutoTokenizer
        mp = getattr(tok, "name_or_path", None)
        if mp:
            cand = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
            if cand.chat_template and "tool_call" in cand.chat_template:
                _RENDER_TOK = cand
    except Exception as e:  # noqa: BLE001
        if _PVP_DBG:
            print(f"[PVP_DBG] intercode render tok fallback ({e!r})", flush=True)
    return _RENDER_TOK


def _first_command(text: str):
    for c in extract_tool_calls(text) or []:
        if c.name == INTERCODE_EXECUTE_TOOL_NAME:
            cmd = c.arguments.get("command")
            if isinstance(cmd, str) and cmd.strip():
                return cmd.strip()
    return None


def _run_episode(env, index: int, trainer, render_tok, decode_tok, max_turns: int):
    """Play one NL2Bash episode via tool-calls; capture turn-0 tokens + terminal reward."""
    query = env.reset(index)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    turn0 = None
    n_tool_turns = 0
    done = False
    reward = 0.0
    for turn in range(max_turns):
        prompt_text = render_tok.apply_chat_template(
            messages, tools=[_EXECUTE_BASH_TOOL], add_generation_prompt=True, tokenize=False)
        out = generate_rollout_completions(trainer, prompts=[prompt_text])[0]
        prompt_ids = out.get("prompt_ids", []) or []
        completion_ids = out.get("completion_ids", []) or []
        logprobs = out.get("logprobs", []) or []
        text = decode_tok.decode(completion_ids, skip_special_tokens=True).strip() if completion_ids else ""

        if turn0 is None and completion_ids:
            turn0 = (list(prompt_ids), list(completion_ids), [float(x) for x in logprobs],
                     len(completion_ids) == len(logprobs))

        cmd = _first_command(text)
        messages.append({"role": "assistant", "content": text})
        if cmd is None:
            break  # no valid execute_bash -> end episode, score below
        n_tool_turns += 1
        obs, reward, done, _info = env.step(cmd)
        if done:
            break
        messages.append({"role": "user", "content": f"Observation: {obs[:DEFAULT_OBS_TRUNCATE_CHARS]}"})

    if not done:  # episode ran out / no submit -> force terminal scoring
        _, reward, done, _info = env.step("submit")
    return turn0, float(reward), n_tool_turns


def make_intercode_rollout():
    """Build the intercode rollout_func (single 'env' in the dispatcher's rotation)."""
    def rollout_first_prompt_and_completion(prompts: list, trainer, max_turns: int = DEFAULT_MAX_TURNS) -> dict[str, list]:
        global _CHECKED_PATHS
        if not _CHECKED_PATHS:
            _assert_no_managed_path_conflicts()
            _CHECKED_PATHS = True

        assets = load_intercode_assets()
        render_tok = _render_tokenizer(trainer)
        decode_tok = trainer.processing_class

        all_prompt_ids, all_completion_ids, all_logprobs, all_rewards = [], [], [], []
        dbg_ok = dbg_fb = dbg_align_bad = 0
        dbg_tool_turns: list[int] = []
        dbg_rewards: list[float] = []

        for prompt in prompts:
            try:
                fs = random.choice(USABLE_FS)
                entries = assets.data[fs]
                idx = random.randrange(len(entries))
                env = LocalBashEnv(fs, entries, assets.snapshot_root)
                turn0, reward, n_tool = _run_episode(env, idx, trainer, render_tok, decode_tok, max_turns)
                if turn0 and turn0[1]:
                    all_prompt_ids.append(turn0[0])
                    all_completion_ids.append(turn0[1])
                    all_logprobs.append(turn0[2])
                    all_rewards.append(reward)
                    dbg_ok += 1
                    dbg_align_bad += int(not turn0[3])
                    dbg_tool_turns.append(n_tool)
                    dbg_rewards.append(round(reward, 3))
                    continue
                raise RuntimeError("no turn-0 trace captured")
            except Exception as e:  # noqa: BLE001 — one entry per prompt
                dbg_fb += 1
                print(f"[pvp_rollout:intercode] episode failed ({e!r}); padding plain completion", flush=True)
                fb = generate_rollout_completions(trainer, prompts=[prompt])[0]
                all_prompt_ids.append(fb.get("prompt_ids", []))
                all_completion_ids.append(fb.get("completion_ids", []))
                all_logprobs.append(fb.get("logprobs", []))
                all_rewards.append(0.0)

        if _PVP_DBG:
            print(f"[PVP_DBG] rollout intercode: prompts={len(prompts)} ok={dbg_ok} fallback={dbg_fb} "
                  f"align_bad={dbg_align_bad} tool_turns={dbg_tool_turns} "
                  f"rewards={dbg_rewards} distinct={sorted(set(dbg_rewards))} mode={SCORING_MODE}",
                  flush=True)

        return {"prompt_ids": all_prompt_ids, "completion_ids": all_completion_ids,
                "logprobs": all_logprobs, "env_rewards": all_rewards}

    return rollout_first_prompt_and_completion

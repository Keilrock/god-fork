"""Multi-env PvP rollout dispatcher.

Tournament env tasks are multi-env (R1=2, R2=4, R3=6 envs per task) and the model
is evaluated on EVERY assigned env, so we must train on all of them — not just the
first. text_trainer resolves rollout_func as
'multi_env.rollout_first_prompt_and_completion' for env tasks and passes the full
env list via the PVP_ENV_NAMES env var (comma-separated EnvironmentName values).

One env is picked per training step (round-robin) and applied to every generation
in that step's GRPO group, so reward scales never mix within a group (othello is
shaped ~[0,0.3], leduc binary [0,1], etc.) — advantage normalisation stays valid.
Different steps cover different envs. Falls back to leduc_poker if the env var is
unset (e.g. a stray single-env invocation).
"""
import os

from _pvp_rollout.multi_env_rollout import make_multi_env_rollout, make_reward_func

_RAW = os.environ.get("PVP_ENV_NAMES", "").strip()
_ENV_VALUES = [e.strip() for e in _RAW.split(",") if e.strip()] or ["leduc_poker"]

rollout_first_prompt_and_completion = make_multi_env_rollout(_ENV_VALUES)
rollout_reward_func = make_reward_func()

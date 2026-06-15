"""GRPO rollout for PvP environment tasks — tool-calling + memory, binary reward.

This wires our training ChatFn into the *dev* PvP harness (core/pvp) so the
model trains in EXACTLY the format it is evaluated in: the tool-calling,
memory-managing LLMBot plays pyspiel games against an in-process MCTS opponent,
working memory resets per game, long-term memory carries across the games of a
matchup (the opponent read). We reuse the dev harness wholesale — agents,
memory, scoring, fairness, forfeit handling — and only swap two things:

  1. generation: the LLMBot's ChatFn routes through TRL's
     generate_rollout_completions (colocate vLLM, policy-synced) instead of an
     SGLang HTTP client, and records the first turn's tokens for GRPO;
  2. reward: the matchup's binary outcome (WIN=1.0 / DRAW=0.5 / LOSS=0.0),
     averaged over the games, instead of a raw env reward.

GRPO trains on the FIRST turn only (later turns have per-step prefixes that
break importance sampling — huggingface/trl#4543; the reference rollout does the
same). Memory still matters: it steers the later turns that decide the game, and
the turn-0 gradient is the binary outcome those later turns produced.

The trainer resolves `rollout_func` per env as `<env>.rollout_first_prompt_and_completion`
(text_trainer.py). Each env's thin wrapper calls make_rollout(<env>) from here.
"""
from __future__ import annotations

import os
import random

# Verbose rollout instrumentation: off by default, enable with PVP_DBG=1.
_PVP_DBG = bool(os.environ.get("PVP_DBG"))

from core.constants import EnvironmentName, ENVIRONMENT_CONFIGS
from core.pvp import constants as cst
from core.pvp.baseline import _make_mcts_bot, _mcts_simulations_for
from core.pvp.game_eval import (
    _AGENT_REGISTRY,
    _evaluate_game_with_timeout,
    config_id_for_seed,
)
from core.pvp.memory import SlotMemory
from core.pvp.scoring import determine_outcome
from core.pvp.tokenizer_counter import load_token_counter
from core.models.pvp_models import (
    ChatCompletionConfig,
    GameScoringContext,
    GameOutcome,
    MemoryArea,
)

from _pvp_rollout.recording_chat_fn import TurnRecord, make_recording_chat_fn

import pyspiel


# WIN/DRAW/LOSS -> scalar reward. Mirrors the eval harness's binary scoring.
_OUTCOME_REWARD = {
    GameOutcome.WIN: 1.0,
    GameOutcome.DRAW: 0.5,
    GameOutcome.LOSS: 0.0,
}

# Games per matchup: long-term memory accrues an opponent read across these, so
# >1 is what makes memory matter. Kept small to bound rollout wall-clock.
_GAMES_PER_MATCHUP = int(os.environ.get("PVP_GAMES_PER_MATCHUP", "2"))


# --- Per-env terminal reward ---------------------------------------------------
# Default: the binary WIN/DRAW/LOSS reward (mirrors the eval harness). Envs that
# need a denser signal register a custom fn in _TERMINAL_REWARD_FNS. KEY: this
# only changes the envs listed there — leduc/gin_rummy/liars_dice keep the proven
# binary reward untouched.

def _binary_reward(outcome: GameOutcome, state, model_seat: int) -> float:
    return _OUTCOME_REWARD[outcome]


def _disc_counts(state, seat: int) -> tuple[float, float]:
    """(own, opp) disc counts from othello's terminal observation tensor.

    observation_tensor(seat) is [3, 8, 8] flattened and PERSPECTIVE-RELATIVE
    (verified): plane 0 = empty, plane 1 = own discs, plane 2 = opponent discs.
    Plane size is derived (len // 3) rather than hard-coded to 64.
    """
    obs = state.observation_tensor(seat)
    plane = len(obs) // 3
    own = float(sum(obs[plane:2 * plane]))
    opp = float(sum(obs[2 * plane:3 * plane]))
    return own, opp


# Level C reward shaping for othello. Its returns are binary +-1, so a 3B model
# that loses to MCTS almost every game produces all-LOSS rollouts -> zero GRPO
# advantage -> no gradient. Blend the binary outcome with the normalized disc
# share so a narrow loss outscores a blowout and gradient survives even among
# losses.  reward = 0.7 * outcome_binary + 0.3 * disc_share
_OTHELLO_OUTCOME_WEIGHT = 0.7
_OTHELLO_DISC_WEIGHT = 0.3


def _othello_reward(outcome: GameOutcome, state, model_seat: int) -> float:
    base = _OUTCOME_REWARD[outcome]
    # disc_share is only meaningful at a terminal board. On a forfeit the game
    # ended early (state not terminal) -> fall back to pure binary so we never
    # reward bailing out early over playing a close game to the end.
    if not state.is_terminal():
        return base
    own, opp = _disc_counts(state, model_seat)
    total = own + opp
    disc_share = 0.5 if total <= 0 else own / total  # neutral if board empty (guard div-by-zero)
    reward = _OTHELLO_OUTCOME_WEIGHT * base + _OTHELLO_DISC_WEIGHT * disc_share
    if _PVP_DBG:
        print(
            f"[PVP_DBG] othello game: outcome={outcome.name} discs={own:.0f}/{opp:.0f} "
            f"share={disc_share:.3f} base={base} reward={reward:.3f}",
            flush=True,
        )
    return reward


# Level C reward shaping for gin_rummy. Unlike othello (binary +-1 returns),
# gin_rummy's returns ARE the deadwood-based score margin (zero-sum, +-knock/gin
# points), so we don't parse hands or reimplement a meld optimizer — pyspiel
# hands us the margin. Blend the binary outcome with the normalized score margin
# so a narrow loss (deadwood close to the opponent's = nearly knocked) outscores
# a blowout (messy hand). margin_share parallels othello's disc_share: it is the
# SAME normalization determine_outcome uses, so >0.5 win / <0.5 loss with the
# magnitude carrying the margin.  reward = 0.7 * outcome_binary + 0.3 * margin_share
_GIN_OUTCOME_WEIGHT = 0.7
_GIN_MARGIN_WEIGHT = 0.3


def _gin_rummy_reward(outcome: GameOutcome, state, model_seat: int) -> float:
    base = _OUTCOME_REWARD[outcome]
    # margin is only meaningful at a terminal hand. On a forfeit the game ended
    # early (state not terminal) -> pure binary, same anti-forfeit rule as othello.
    if not state.is_terminal():
        return base
    game = state.get_game()
    lo, hi = game.min_utility(), game.max_utility()
    player_return = state.returns()[model_seat]
    margin_share = 0.5 if hi <= lo else (player_return - lo) / (hi - lo)
    reward = _GIN_OUTCOME_WEIGHT * base + _GIN_MARGIN_WEIGHT * margin_share
    if _PVP_DBG:
        print(
            f"[PVP_DBG] gin_rummy game: outcome={outcome.name} score={player_return:.0f} "
            f"margin_share={margin_share:.3f} base={base} reward={reward:.3f}",
            flush=True,
        )
    return reward


_TERMINAL_REWARD_FNS = {
    EnvironmentName.OTHELLO: _othello_reward,
    EnvironmentName.GIN_RUMMY: _gin_rummy_reward,
}


def _chat_config_from_trainer(trainer) -> ChatCompletionConfig:
    """Build the ChatCompletionConfig the LLMBot needs. base_url/api_key are
    unused (generation goes through generate_rollout_completions, not HTTP), but
    the model carries the config for tokenizer-budgeting and temperature."""
    model_name = getattr(trainer, "model_name", None) or os.environ.get(
        "BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct"
    )
    args = getattr(trainer, "args", None)
    temperature = getattr(args, "temperature", 1.0)
    return ChatCompletionConfig(
        inference_model=model_name,
        tokenizer_repo=model_name,
        base_url="http://unused.local/v1",
        temperature=temperature,
        max_tokens=cst.PVP_TURN_MAX_TOKENS,
    )


def _play_matchup(env_name: EnvironmentName, trainer, base_seed: int) -> tuple[TurnRecord, float, bool]:
    """Play one matchup (model vs MCTS, _GAMES_PER_MATCHUP games) and return the
    first turn's token trace, the mean binary reward, and a forfeit flag.

    Structure mirrors core.pvp.baseline.run_mcts_baseline, but the model's
    ChatFn records turn-0 tokens, and we return the binary reward for GRPO.
    """
    agent = _AGENT_REGISTRY[env_name]()
    env_config = ENVIRONMENT_CONFIGS[env_name]
    simulations = _mcts_simulations_for(env_name)
    config = _chat_config_from_trainer(trainer)
    counter = load_token_counter(config.tokenizer_repo or config.inference_model)

    # long-term memory persists across the games of THIS matchup (opponent read)
    long_term = SlotMemory(cst.PVP_LONGTERM_MEM_SLOTS, cst.PVP_LONGTERM_SLOT_TOKENS, counter)

    sink = TurnRecord()
    chat_fn = make_recording_chat_fn(trainer, sink)

    seed_rng = random.Random(base_seed)
    rewards: list[float] = []
    any_forfeit = False

    from core.pvp.bot import LLMBot  # local import: heavy (pyspiel/openai) deps

    for i in range(_GAMES_PER_MATCHUP):
        seed = seed_rng.randint(1, cst.PVP_SEED_RANGE_MAX)
        config_id = config_id_for_seed(seed, env_config)
        game = pyspiel.load_game(agent.game_name, agent.generate_params(config_id))
        game_type = game.get_type()

        model_seat = i % 2  # alternate seats for fairness
        mcts_seat = 1 - model_seat
        working = SlotMemory(cst.PVP_WORKING_MEM_SLOTS, cst.PVP_WORKING_SLOT_TOKENS, counter)
        model_bot = LLMBot(
            game=game,
            player_id=model_seat,
            chat_fn=chat_fn,
            config=config,
            agent=agent,
            memories={MemoryArea.WORKING: working, MemoryArea.LONG_TERM: long_term},
        )
        bots: list = [None, None]
        bots[model_seat] = model_bot
        bots[mcts_seat] = _make_mcts_bot(game, simulations, seed + mcts_seat)

        state = game.new_initial_state()
        agent.setup_initial_state(state, seed)
        evaluation = _evaluate_game_with_timeout(state, bots, seed)

        outcome = determine_outcome(
            GameScoringContext(
                returns=evaluation.returns,
                player_id=model_seat,
                is_zero_sum=game_type.utility == pyspiel.GameType.Utility.ZERO_SUM,
                min_utility=game.min_utility(),
                max_utility=game.max_utility(),
            )
        )
        reward_fn = _TERMINAL_REWARD_FNS.get(env_name, _binary_reward)
        rewards.append(reward_fn(outcome, state, model_seat))
        if getattr(evaluation, "forfeit", False):
            any_forfeit = True

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    return sink, mean_reward, any_forfeit


def make_rollout(env_value: str):
    """Build the rollout_func the trainer calls for env `env_value`."""
    env_name = EnvironmentName(env_value)

    def rollout_first_prompt_and_completion(prompts: list[str], trainer, max_turns: int = 30) -> dict[str, list]:
        from trl.experimental.openenv import generate_rollout_completions

        all_prompt_ids: list[list[int]] = []
        all_completion_ids: list[list[int]] = []
        all_logprobs: list[list[float]] = []
        all_rewards: list[float] = []

        base = random.randint(1, 2_000_000_000)

        # --- debug instrumentation (smoke verification) ---
        dbg_ok = 0
        dbg_fallback = 0
        dbg_forfeits = 0
        dbg_turns: list[int] = []
        dbg_tool_turns: list[int] = []
        dbg_align_bad = 0

        for n, prompt in enumerate(prompts):
            try:
                sink, reward, _forfeit = _play_matchup(env_name, trainer, base_seed=base + n)
                if sink.captured and sink.completion_ids:
                    all_prompt_ids.append(sink.prompt_ids)
                    all_completion_ids.append(sink.completion_ids)
                    all_logprobs.append(sink.logprobs)
                    all_rewards.append(reward)
                    dbg_ok += 1
                    dbg_turns.append(sink.n_turns)
                    dbg_tool_turns.append(sink.n_tool_call_turns)
                    dbg_forfeits += int(bool(_forfeit))
                    dbg_align_bad += int(not sink.align_ok)
                    continue
                # matchup produced no usable turn-0 trace: fall through to pad
                raise RuntimeError("no turn-0 trace captured")
            except Exception as e:  # noqa: BLE001 — trainer needs one entry per prompt
                dbg_fallback += 1
                print(f"[pvp_rollout:{env_value}] matchup failed ({e!r}); padding with plain completion", flush=True)
                fb = generate_rollout_completions(trainer, prompts=[prompt])[0]
                all_prompt_ids.append(fb.get("prompt_ids", []))
                all_completion_ids.append(fb.get("completion_ids", []))
                all_logprobs.append(fb.get("logprobs", []))
                all_rewards.append(0.0)

        if _PVP_DBG:
            print(
                f"[PVP_DBG] rollout {env_value}: prompts={len(prompts)} matchup_ok={dbg_ok} "
                f"fallback={dbg_fallback} forfeits={dbg_forfeits} align_bad={dbg_align_bad} "
                f"turns_per_matchup={dbg_turns} tool_call_turns={dbg_tool_turns} "
                f"rewards={all_rewards} distinct_rewards={sorted(set(all_rewards))}",
                flush=True,
            )

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_rewards": all_rewards,
        }

    return rollout_first_prompt_and_completion


def make_reward_func():
    """Build the reward_func: surface the binary env_rewards GRPO advantages use."""
    def rollout_reward_func(completions, **kwargs):
        rewards = kwargs.get("env_rewards") if kwargs else None
        return [float(r) for r in rewards] if rewards is not None else [0.0] * len(completions)

    return rollout_reward_func

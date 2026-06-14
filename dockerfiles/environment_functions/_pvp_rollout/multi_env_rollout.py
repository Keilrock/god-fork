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
        rewards.append(_OUTCOME_REWARD[outcome])
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

        for n, prompt in enumerate(prompts):
            try:
                sink, reward, _forfeit = _play_matchup(env_name, trainer, base_seed=base + n)
                if sink.captured and sink.completion_ids:
                    all_prompt_ids.append(sink.prompt_ids)
                    all_completion_ids.append(sink.completion_ids)
                    all_logprobs.append(sink.logprobs)
                    all_rewards.append(reward)
                    continue
                # matchup produced no usable turn-0 trace: fall through to pad
                raise RuntimeError("no turn-0 trace captured")
            except Exception as e:  # noqa: BLE001 — trainer needs one entry per prompt
                print(f"[pvp_rollout:{env_value}] matchup failed ({e}); padding with plain completion")
                fb = generate_rollout_completions(trainer, prompts=[prompt])[0]
                all_prompt_ids.append(fb.get("prompt_ids", []))
                all_completion_ids.append(fb.get("completion_ids", []))
                all_logprobs.append(fb.get("logprobs", []))
                all_rewards.append(0.0)

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

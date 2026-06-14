"""Thin wrapper: liars_dice PvP rollout delegates to the shared multi-env module.

The trainer resolves rollout_func as 'liars_dice.rollout_first_prompt_and_completion'
and reward as 'liars_dice.rollout_reward_func' (text_trainer.py). Both come from
_pvp_rollout, which plays the dev tool-calling/memory LLMBot vs MCTS and scores
the matchup binary (WIN/DRAW/LOSS). Edge over the reference ReAct rollout:
training format == eval format (tool calls + memory).
"""
from _pvp_rollout.multi_env_rollout import make_rollout, make_reward_func

rollout_first_prompt_and_completion = make_rollout("liars_dice")
rollout_reward_func = make_reward_func()

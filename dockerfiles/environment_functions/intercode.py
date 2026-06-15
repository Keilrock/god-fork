"""Thin wrapper: intercode (NL2Bash) rollout for single-env intercode tasks.

text_trainer resolves 'intercode.rollout_first_prompt_and_completion' +
'intercode.rollout_reward_func'. The rollout runs the LocalBashEnv (princeton
reward, verbatim) driven by execute_bash tool-calls — training format == PR1201
eval format. Multi-env tasks reach the same rollout via the multi_env dispatcher.
"""
from _pvp_rollout.intercode_rollout import make_intercode_rollout
from _pvp_rollout.multi_env_rollout import make_reward_func

rollout_first_prompt_and_completion = make_intercode_rollout()
rollout_reward_func = make_reward_func()

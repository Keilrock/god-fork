"""Othello-vs-MCTS eval for the weighting experiment.

Plays N othello games as the tool-calling LLMBot against pyspiel MCTS (same
strength as the real eval, mcts_max_simulations from env config), alternating
seats. Records for each game: outcome (W/D/L) AND terminal disc-share
own/(own+opp) — the continuous signal the reward shaping optimises. Win rate
from a ~3% base is noisy at small N, so disc-share is the sensitive metric for
"did training make the model play better othello".

Usage (inside the trainer image, with a vLLM OpenAI server already up):
    python othello_eval.py --model <served-name> --base-url http://127.0.0.1:8000/v1 \
        --tokenizer <hf-repo-or-path> --num-games 20 --time-budget 1200
"""
import argparse
import logging
import random
import time

import pyspiel

from core.constants import ENVIRONMENT_CONFIGS, EnvironmentName
from core.models.pvp_models import (
    ChatCompletionConfig, GameOutcome, GameScoringContext, MemoryArea,
)
from core.pvp import constants as cst
from core.pvp.bot import LLMBot
from core.pvp.chat import chat_completion, create_client
from core.pvp.game_eval import (
    _AGENT_REGISTRY, _evaluate_game_with_timeout, config_id_for_seed,
)
from core.pvp.memory import SlotMemory
from core.pvp.scoring import determine_outcome
from core.pvp.tokenizer_counter import load_token_counter

logging.basicConfig(level=logging.WARNING)


def _disc_counts(state, seat):
    obs = state.observation_tensor(seat)
    plane = len(obs) // 3
    own = float(sum(obs[plane:2 * plane]))
    opp = float(sum(obs[2 * plane:3 * plane]))
    return own, opp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--num-games", type=int, default=20)
    ap.add_argument("--mcts-sims", type=int, default=None, help="override MCTS strength")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--time-budget", type=float, default=None)
    args = ap.parse_args()

    env_name = EnvironmentName.OTHELLO
    agent = _AGENT_REGISTRY[env_name]()
    env_config = ENVIRONMENT_CONFIGS[env_name]
    extra = env_config.eval_payload_extra or {}
    sims = args.mcts_sims if args.mcts_sims is not None else int(extra.get("mcts_max_simulations", 50))

    config = ChatCompletionConfig(
        inference_model=args.model,
        tokenizer_repo=args.tokenizer,
        base_url=args.base_url,
        temperature=0.0,
        seed=0,
        max_tokens=cst.PVP_TURN_MAX_TOKENS,
        max_retries=3,
        read_timeout=30.0,
    )
    client = create_client(config)

    def chat_fn(cfg, messages, tools):
        return chat_completion(client, cfg, messages, tools)

    counter = load_token_counter(config.tokenizer_repo or config.inference_model)
    long_term = SlotMemory(cst.PVP_LONGTERM_MEM_SLOTS, cst.PVP_LONGTERM_SLOT_TOKENS, counter)

    import numpy as np
    from open_spiel.python.algorithms import mcts

    def make_mcts(game, seed):
        ev = mcts.RandomRolloutEvaluator(n_rollouts=int(extra.get("mcts_num_rollouts", 1)),
                                         random_state=np.random.RandomState(seed))
        return mcts.MCTSBot(game, uct_c=2.0, max_simulations=sims, evaluator=ev,
                            random_state=np.random.RandomState(seed))

    seed_rng = random.Random(args.base_seed)
    wins = draws = losses = forfeits = 0
    shares = []
    started = time.monotonic()
    print(f"[EVAL] othello vs MCTS-{sims}, model={args.model}, N={args.num_games}", flush=True)

    for i in range(args.num_games):
        if args.time_budget and time.monotonic() - started >= args.time_budget:
            print(f"[EVAL] time budget hit after {i} games", flush=True)
            break
        seed = seed_rng.randint(1, cst.PVP_SEED_RANGE_MAX)
        config_id = config_id_for_seed(seed, env_config)
        game = pyspiel.load_game(agent.game_name, agent.generate_params(config_id))
        gtype = game.get_type()
        model_seat = i % 2
        mcts_seat = 1 - model_seat
        working = SlotMemory(cst.PVP_WORKING_MEM_SLOTS, cst.PVP_WORKING_SLOT_TOKENS, counter)
        model_bot = LLMBot(game=game, player_id=model_seat, chat_fn=chat_fn, config=config,
                           agent=agent, memories={MemoryArea.WORKING: working, MemoryArea.LONG_TERM: long_term})
        bots = [None, None]
        bots[model_seat] = model_bot
        bots[mcts_seat] = make_mcts(game, seed + mcts_seat)

        state = game.new_initial_state()
        agent.setup_initial_state(state, seed)
        ev = _evaluate_game_with_timeout(state, bots, seed)
        outcome = determine_outcome(GameScoringContext(
            returns=ev.returns, player_id=model_seat,
            is_zero_sum=gtype.utility == pyspiel.GameType.Utility.ZERO_SUM,
            min_utility=game.min_utility(), max_utility=game.max_utility()))

        share = None
        if state.is_terminal():
            own, opp = _disc_counts(state, model_seat)
            tot = own + opp
            share = 0.5 if tot <= 0 else own / tot
            shares.append(share)
        else:
            forfeits += 1

        if outcome == GameOutcome.WIN:
            wins += 1
        elif outcome == GameOutcome.LOSS:
            losses += 1
        else:
            draws += 1
        sstr = f"{share:.3f}" if share is not None else "FORFEIT"
        print(f"[EVAL] game {i+1}/{args.num_games} seat={model_seat} outcome={outcome.name} disc_share={sstr}", flush=True)

        if ev.forfeiting_player_id != model_seat:
            model_bot.reflect(state, outcome)

    n = wins + draws + losses
    win_rate = wins / n if n else 0.0
    mean_score = (wins + 0.5 * draws) / n if n else 0.0
    mean_share = sum(shares) / len(shares) if shares else float("nan")
    print("=" * 60, flush=True)
    print(f"[RESULT] model={args.model} MCTS-{sims} games={n}", flush=True)
    print(f"[RESULT] W-D-L = {wins}-{draws}-{losses}  forfeits={forfeits}", flush=True)
    print(f"[RESULT] win_rate={win_rate:.3f}  mean_score={mean_score:.3f}", flush=True)
    print(f"[RESULT] mean_disc_share={mean_share:.4f} (n_terminal={len(shares)})", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()

"""Game-specific agents for PvP evaluation.

Each agent provides state formatting and parameter generation for its game.
Rules text is loaded from core/config/pvp_game_prompts.yml.
"""

import functools
import os
import random
import re
from abc import ABC, abstractmethod
from pathlib import Path

import pyspiel
import yaml

_PROMPTS_PATH = Path(__file__).resolve().parents[2] / "core" / "config" / "pvp_game_prompts.yml"


@functools.cache
def load_prompts() -> dict[str, str]:
    with open(_PROMPTS_PATH) as f:
        return yaml.safe_load(f)


class BaseGameAgent(ABC):
    """Abstract base for game-specific LLM prompt generation."""

    @property
    @abstractmethod
    def game_name(self) -> str:
        ...

    @property
    @abstractmethod
    def rules_key(self) -> str:
        """Key in pvp_game_prompts.yml for this game's rules."""
        ...

    @abstractmethod
    def generate_params(self, config_id: int) -> dict[str, int]:
        """Generate pyspiel game parameters from a config variant ID."""
        ...

    def setup_initial_state(self, state: pyspiel.State, seed: int) -> None:
        """Advance the fresh state before the models take over. Default: no-op.

        Games with chance nodes (dice, card deals) get their per-game variety for
        free from the seed passed to evaluate_bots. Deterministic games with no
        chance nodes (e.g. othello) override this to inject seeded variety so the
        same seed reproduces the same start while different seeds diverge.
        """
        return None

    def get_rules(self) -> str:
        return load_prompts()[self.rules_key]

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Format game state as text. Override for game-specific formatting."""
        try:
            return state.observation_string(player_id)
        except (RuntimeError, AttributeError):
            pass
        try:
            return state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            raise ValueError(
                f"Game {self.game_name} supports neither observation_string nor "
                f"information_state_string — override format_state() for this game"
            )

    def describe_action(self, state: pyspiel.State, player_id: int, action: int) -> str:
        """Human-readable label for a legal action id. Default: pyspiel's string.

        Override per game when the native action string is ambiguous out of
        context (e.g. liars_dice bids render as '1-1', which a model misreads as
        a bridge/trick bid)."""
        return state.action_to_string(player_id, action)

    def select_actions(self, state: pyspiel.State, player_id: int, legal_actions: list[int]) -> list[int]:
        """Which legal action ids to PRESENT to the model (prompt list + tool enum).
        Default: all of them. Override per game when the full legal set is large
        enough to overwhelm the model into summarising instead of acting (e.g.
        liars_dice opens with 60 bids). The bot still validates the chosen move
        against the FULL legal set, so narrowing only steers, never illegalises."""
        return legal_actions

    def generate_system_prompt(self) -> str:
        prompts = load_prompts()
        return prompts["system_prompt_template"].format(
            game_name=self.game_name, rules=self.get_rules()
        )


# --- Concrete agents ---


class LiarsDiceAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "liars_dice"

    @property
    def rules_key(self) -> str:
        return "liars_dice_rules"

    def describe_action(self, state: pyspiel.State, player_id: int, action: int) -> str:
        # pyspiel renders bids as "<quantity>-<face>" (e.g. "1-1") and the
        # challenge as "Liar". Bare "1-1" reads like a bridge/trick bid; spell it
        # out so the model treats it as a dice bid and calls game_action.
        raw = state.action_to_string(player_id, action)
        quantity, sep, face = raw.partition("-")
        if sep and quantity.isdigit() and face.isdigit():
            return f"bid: quantity={quantity} face={face} (claim at least {quantity} dice show face {face})"
        if raw.strip().lower() == "liar":
            return "call Liar (challenge the previous bid as false)"
        return raw

    # The opening turn exposes ~60 legal bids; presenting all of them pushes the
    # model into "summarise the list" mode and it never emits a tool call. Show a
    # small, strategically-sane subset instead: the lowest few bids (smallest
    # quantity-then-face — the natural opening moves) plus "call Liar" whenever
    # it's legal. The bot validates against the FULL legal set, so this only
    # narrows what's shown, never what's allowed.
    _LIARS_MENU_K = int(os.environ.get("LIARS_MENU_K", "12"))

    def select_actions(self, state: pyspiel.State, player_id: int, legal_actions: list[int]) -> list[int]:
        if len(legal_actions) <= self._LIARS_MENU_K:
            return legal_actions
        liar, bids = [], []
        for a in legal_actions:
            (liar if state.action_to_string(player_id, a).strip().lower() == "liar" else bids).append(a)
        # action ids are ordered by (quantity, face); the lowest are the
        # canonical opening raises. Keep the K lowest bids + always allow a call.
        return sorted(bids)[: self._LIARS_MENU_K] + liar

    def generate_params(self, config_id: int) -> dict[str, int]:
        return {"players": 2, "numdice": 5}

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        try:
            info_str = state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            return str(state)

        if not info_str:
            return str(state)

        parts = info_str.split()
        dice_part = parts[0]
        bid_parts = [p for p in parts[1:] if "-" in p]

        dice = [int(d) for d in dice_part if d.isdigit()]
        num_dice = len(dice)
        total_dice = num_dice * state.num_players()

        lines = [
            f"Your dice: {dice} (showing: {', '.join(map(str, dice))})",
            f"Dice per player: {num_dice}",
            f"Total dice in game: {total_dice}",
            f"Players: {state.num_players()}",
            f"Current player: Player {state.current_player()}",
        ]

        if bid_parts:
            last_bid = bid_parts[-1]
            quantity, face = last_bid.split("-")
            lines.append(
                f'\nCurrent bid: "{quantity}-{face}" '
                f"(at least {quantity} dice showing {face} across all players)"
            )
            lines.append("You can: (1) Make a higher bid, or (2) Call 'Liar'")
        else:
            lines.append("No bid yet - you must make the first bid")

        return "\n".join(lines)


class LeducPokerAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "leduc_poker"

    @property
    def rules_key(self) -> str:
        return "leduc_poker_rules"

    def generate_params(self, config_id: int) -> dict[str, int]:
        return {"players": 2}

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        try:
            info_str = state.information_state_string(player_id)
        except (RuntimeError, AttributeError):
            return str(state)

        private_card = self._extract(info_str, r"\[Private: (-?\d+)\]")
        round_num = self._extract(info_str, r"\[Round (\d+)\]")
        pot = self._extract(info_str, r"\[Pot: (\d+)\]")
        money = self._extract(info_str, r"\[Money: ([\d ]+)\]")
        public_card = self._extract(info_str, r"\[Public: (-?\d+)\]")
        round1_seq = self._extract(info_str, r"\[Round1: ([^\]]*)\]")
        round2_seq = self._extract(info_str, r"\[Round2: ([^\]]*)\]")

        lines: list[str] = []

        if private_card and private_card != "-10000":
            lines.append(f"Your card: {self._card_name(int(private_card))}")
        else:
            lines.append("Your card: (not dealt yet)")

        if public_card and public_card != "-10000":
            lines.append(f"Public card: {self._card_name(int(public_card))}")
            if private_card and private_card != "-10000":
                if int(private_card) // 2 == int(public_card) // 2:
                    lines.append("Hand: PAIR")

        lines.append(f"Round: {round_num}/2")
        lines.append(f"Pot: {pot} chips")

        if money:
            chips = money.split()
            if len(chips) >= 2:
                lines.append(f"Your chips: {chips[player_id]}")
                lines.append(f"Opponent chips: {chips[1 - player_id]}")

        if round1_seq:
            lines.append(f"Round 1 actions: {self._parse_betting(round1_seq)}")
        if round2_seq:
            lines.append(f"Round 2 actions: {self._parse_betting(round2_seq)}")

        return "\n".join(lines)

    @staticmethod
    def _extract(info_str: str, pattern: str) -> str:
        match = re.search(pattern, info_str)
        return match.group(1) if match else ""

    @staticmethod
    def _card_name(card_id: int) -> str:
        ranks = ["J", "Q", "K", "A"]  # A used only in 3+ player variants
        suits = ["\u2660", "\u2665"]
        rank_idx = card_id // 2
        suit_idx = card_id % 2
        if rank_idx < len(ranks):
            return f"{ranks[rank_idx]}{suits[suit_idx]}"
        return f"Card_{card_id}"

    @staticmethod
    def _parse_betting(seq: str) -> str:
        if not seq or not seq.strip():
            return "(none)"
        actions_map = {0: "Fold", 1: "Call", 2: "Raise"}
        numbers = [int(x) for x in seq.split() if x.isdigit()]
        if not numbers:
            return "(none)"
        return ", ".join(actions_map.get(a, f"Action{a}") for a in numbers)


class GinRummyAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "gin_rummy"

    @property
    def rules_key(self) -> str:
        return "gin_rummy_rules"

    def generate_params(self, config_id: int) -> dict[str, int]:
        hand_var = (config_id // 3) % 3
        knock_var = config_id % 3
        return {
            "hand_size": 7 + hand_var,
            "knock_card": 10 - knock_var,
        }

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        return state.observation_string(player_id)


# Number of seeded random opening plies applied to an othello game, sampled from
# this inclusive range. Enough to diverge the opening tree for variety, few
# enough that positions stay balanced and game-like.
_OTHELLO_OPENING_PLIES = (2, 6)


class OthelloAgent(BaseGameAgent):

    @property
    def game_name(self) -> str:
        return "othello"

    @property
    def rules_key(self) -> str:
        return "othello_rules"

    def generate_params(self, config_id: int) -> dict[str, int]:
        return {}

    def format_state(self, state: pyspiel.State, player_id: int) -> str:
        """Prefix the board with the player's colour.

        The observation only says whose turn it is ("Black (x) to play"), so
        without this line the model must infer its own colour — small models
        get it wrong and play for the opponent.
        """
        colour = "x (Black)" if player_id == 0 else "o (White)"
        return f"You play {colour}.\n{state.observation_string(player_id)}"

    def setup_initial_state(self, state: pyspiel.State, seed: int) -> None:
        """Apply a seeded number of uniformly-random legal opening moves.

        Othello is deterministic with no chance nodes, so every game would start
        from the identical board. Deriving the opening plies from the instance
        seed keeps games reproducible (same seed -> same start) while giving each
        seed a distinct mid-game position to play from.
        """
        rng = random.Random(seed)
        num_plies = rng.randint(*_OTHELLO_OPENING_PLIES)
        for _ in range(num_plies):
            if state.is_terminal():
                break
            legal_actions = state.legal_actions()
            if not legal_actions:
                break
            state.apply_action(rng.choice(legal_actions))

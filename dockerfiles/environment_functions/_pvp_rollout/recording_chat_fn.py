"""A training-time ChatFn for the PvP LLMBot.

The eval harness drives LLMBot with an OpenAI/SGLang client (chat.py): it sends
`tools` natively and SGLang's qwen25 parser returns structured tool_calls. In
TRAINING we can't use that path — generation goes through TRL's
`generate_rollout_completions` (colocate vLLM, weight-synced with the policy),
which returns raw token ids + logprobs, not parsed tool calls. So this ChatFn:

  1. takes the same (config, messages, tools) the LLMBot hands any ChatFn,
  2. generates the completion via generate_rollout_completions(as_chat=True),
  3. parses Qwen `<tool_call>` blocks out of the completion TEXT ourselves
     (brace-balanced, tolerant of a missing close tag — identical behaviour to
     what SGLang's parser yields, so training format == eval format),
  4. records the FIRST turn's prompt_ids / completion_ids / logprobs into a
     caller-supplied sink, because GRPO trains on the first turn only (later
     turns have per-step prefixes that break importance sampling — see
     huggingface/trl#4543, and the reference rollout trains turn-0 only too).

The tools are rendered into the prompt as TEXT (via the tokenizer chat template
applying `tools`), NOT through TRL's native `tools=` feature — that feature
needs transformers>=5.0, which conflicts with axolotl's 4.53 pin. Qwen's chat
template renders tool schemas into the prompt as text on its own, and the model
emits `<tool_call>` text we parse back. No transformers 5.0 needed.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from core.models.pvp_models import (
    ChatCompletionConfig,
    ChatMessage,
    ChatResult,
    ToolCall,
    ToolSchema,
)

_THINK_COMPLETE = re.compile(r"<think(?:ing)?>.*?</think(?:ing)?>", re.DOTALL | re.IGNORECASE)
_THINK_UNCLOSED = re.compile(r"<think(?:ing)?>.*", re.DOTALL | re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    cleaned = _THINK_COMPLETE.sub("", text)
    for tag in ("</think>", "</thinking>"):
        if tag in cleaned:
            cleaned = cleaned.split(tag)[-1]
    cleaned = _THINK_UNCLOSED.sub("", cleaned)
    return cleaned.strip()


def _decode_arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    if not isinstance(raw, str):
        return {}
    try:
        out = json.loads(raw)
        return out if isinstance(out, dict) else {}
    except json.JSONDecodeError:
        return {}


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Pull tool calls from Qwen-style <tool_call> blocks (brace-balanced).

    Qwen emits `<tool_call>\\n{json}` and does NOT reliably close with
    </tool_call> (and may stack several). Locate each <tool_call>, extract the
    following JSON object by brace-balancing (handles nested 'arguments'),
    independent of any closing tag. Malformed blocks are skipped, never raised.
    """
    calls: list[ToolCall] = []
    i = 0
    idx = 0
    tag = "<tool_call>"
    while True:
        start = text.find(tag, idx)
        if start == -1:
            break
        brace = text.find("{", start + len(tag))
        if brace == -1:
            break
        depth = 0
        end = -1
        in_str = False
        esc = False
        for j in range(brace, len(text)):
            ch = text[j]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = j
                    break
        if end == -1:
            break
        blob = text[brace:end + 1]
        idx = end + 1
        try:
            obj = json.loads(blob)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict) or "name" not in obj:
            continue
        args = obj.get("arguments", {})
        if isinstance(args, str):
            args = _decode_arguments(args)
        elif not isinstance(args, dict):
            args = {}
        else:
            args = {k: strip_think_tags(v) if isinstance(v, str) else v for k, v in args.items()}
        calls.append(ToolCall(id=f"call_{i}", name=str(obj["name"]), arguments=args))
        i += 1
    return calls


@dataclass
class TurnRecord:
    """The token trace of the matchup's first model turn — what GRPO trains on."""
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    captured: bool = False


def make_recording_chat_fn(trainer, sink: TurnRecord):
    """Build a ChatFn(config, messages, tools) -> ChatResult for training.

    Generation runs through TRL's generate_rollout_completions (colocate vLLM,
    policy-synced logprobs). The FIRST call's token trace is written into `sink`
    (GRPO trains turn-0 only). `tools` is accepted for interface parity with the
    eval ChatFn but not sent natively — Qwen's chat template renders the schemas
    into the prompt as text, and we parse the `<tool_call>` text back out.
    """
    from trl.experimental.openenv import generate_rollout_completions

    tokenizer = trainer.processing_class

    def chat_fn(
        config: ChatCompletionConfig,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
    ) -> ChatResult:
        wire_messages = [m.to_openai() for m in messages]

        gen_kwargs = {"prompts": [wire_messages], "as_chat": True}
        if tools:
            # Pass tool schemas so the chat template can render them into the
            # prompt as text (Qwen does this); the model then emits <tool_call>.
            gen_kwargs["tools"] = [t.to_openai() for t in tools]

        try:
            out = generate_rollout_completions(trainer, **gen_kwargs)[0]
        except TypeError:
            # Older generate_rollout_completions without `tools=`: fall back to
            # plain chat. The tool schemas must then already be baked into the
            # message text by the caller (multi_env_rollout does this).
            out = generate_rollout_completions(trainer, prompts=[wire_messages], as_chat=True)[0]

        prompt_ids = out.get("prompt_ids", []) or []
        completion_ids = out.get("completion_ids", []) or []
        logprobs = out.get("logprobs", []) or []

        if not sink.captured:
            sink.prompt_ids = list(prompt_ids)
            sink.completion_ids = list(completion_ids)
            sink.logprobs = [float(x) for x in logprobs]
            sink.captured = True

        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip() if completion_ids else ""
        cleaned = strip_think_tags(text)
        tool_calls = extract_tool_calls(text) or None
        content = cleaned or None

        return ChatResult(content=content, tool_calls=tool_calls, usage=None)

    return chat_fn

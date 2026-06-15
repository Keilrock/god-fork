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
import os
import re
from dataclasses import dataclass, field

# Verbose rollout instrumentation: off by default, enable with PVP_DBG=1.
_PVP_DBG = bool(os.environ.get("PVP_DBG"))

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


def _scalarize(v):
    """Coerce a tool-arg value to a JsonScalar (str/int/float/bool/None).

    ToolCall.arguments is dict[str, JsonScalar]. Models sometimes emit nested
    values (echoing the tool schema, common with richer action spaces like
    othello). Non-scalars are JSON-encoded so ToolCall construction never raises
    and any valid scalar keys (e.g. action_id) survive alongside the junk.
    """
    if v is None or isinstance(v, (str, int, float, bool)):
        return strip_think_tags(v) if isinstance(v, str) else v
    return json.dumps(v, ensure_ascii=False)


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


def _balanced_json_end(text: str, brace: int) -> int:
    """Index of the closing `}` matching the `{` at `brace` (string-aware), or -1."""
    depth = 0
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
                return j
    return -1


def _obj_to_toolcall(obj, idx: int) -> "ToolCall | None":
    """Build a ToolCall from a parsed JSON dict across the formats the whitelisted
    models actually emit. Returns None for non-calls/malformed.

    Shapes handled:
      Qwen:           {"name": ..., "arguments": {...}}
      Llama bare:     {"name": ..., "parameters": {...}}
      Llama wrapped:  {"type":"function","function":{"name":...,"parameters":{...}}}
      Llama variant:  {"function":"name", "parameters":{...}}
    Args may be under 'arguments' or 'parameters', on the name-holder or top level.
    """
    if not isinstance(obj, dict):
        return None
    fn = obj.get("function")
    if isinstance(fn, dict):              # wrapped: {"function": {"name":..., ...}}
        name = fn.get("name")
        args = fn.get("arguments")
        if args is None:
            args = fn.get("parameters")
    elif isinstance(fn, str):            # {"function": "name", "parameters": {...}}
        name = fn
        args = None
    else:                                # bare: name at top level
        name = obj.get("name")
        args = None
    if args is None:                     # fall back to top-level args
        args = obj.get("arguments")
    if args is None:
        args = obj.get("parameters")
    if not name:
        return None
    if isinstance(args, str):
        args = _decode_arguments(args)
    elif not isinstance(args, dict):
        args = {}
    args = {str(k): _scalarize(v) for k, v in args.items()}
    try:
        return ToolCall(id=f"call_{idx}", name=str(name), arguments=args)
    except Exception:  # noqa: BLE001 — never let a malformed call crash the rollout
        return None


def extract_tool_calls(text: str) -> list[ToolCall]:
    """Parse tool calls from a completion, handling every whitelisted model family:

    - Qwen2.5/Qwen3 + Hermes: `<tool_call>\\n{json}` blocks (json has 'arguments'),
      tolerant of a missing </tool_call>, possibly stacked.
    - Llama-3.1/3.2: bare `{"name":..., "parameters":...}` (no tags; the
      <|python_tag|> prefix is already stripped by skip_special_tokens at decode).

    Brace-balanced + string-aware; malformed blobs are skipped, never raised.
    """
    calls: list[ToolCall] = []
    i = 0

    # Path A: <tool_call>-delimited blocks (Qwen / Hermes).
    idx = 0
    tag = "<tool_call>"
    while True:
        start = text.find(tag, idx)
        if start == -1:
            break
        brace = text.find("{", start + len(tag))
        if brace == -1:
            break
        end = _balanced_json_end(text, brace)
        if end == -1:
            break
        idx = end + 1
        try:
            obj = json.loads(text[brace:end + 1])
        except json.JSONDecodeError:
            continue
        tc = _obj_to_toolcall(obj, i)
        if tc is not None:
            calls.append(tc)
            i += 1
    if calls:
        return calls

    # Path B: no <tool_call> tags -> scan for bare JSON objects with a "name"
    # key (Llama-3.x format). Objects without "name" are skipped, so stray JSON
    # in the text doesn't produce false calls.
    scan = 0
    while True:
        brace = text.find("{", scan)
        if brace == -1:
            break
        end = _balanced_json_end(text, brace)
        if end == -1:
            break
        scan = end + 1
        try:
            obj = json.loads(text[brace:end + 1])
        except json.JSONDecodeError:
            continue
        tc = _obj_to_toolcall(obj, i)
        if tc is not None:
            calls.append(tc)
            i += 1
    return calls


@dataclass
class TurnRecord:
    """The token trace of the matchup's first model turn — what GRPO trains on."""
    prompt_ids: list[int] = field(default_factory=list)
    completion_ids: list[int] = field(default_factory=list)
    logprobs: list[float] = field(default_factory=list)
    captured: bool = False
    # --- debug instrumentation (smoke verification) ---
    n_turns: int = 0            # total chat_fn calls this matchup
    n_tool_call_turns: int = 0  # turns where the model emitted >=1 parsed tool_call
    tools_in_prompt: bool = False   # tool schemas were rendered into the prompt text
    render_tpl_native: bool = False  # render tokenizer used the model's native (family) tool template
    align_ok: bool = True       # captured turn: len(completion_ids)==len(logprobs)


def make_recording_chat_fn(trainer, sink: TurnRecord):
    """Build a ChatFn(config, messages, tools) -> ChatResult for training.

    Generation runs through TRL's generate_rollout_completions (colocate vLLM,
    policy-synced logprobs). The FIRST call's token trace is written into `sink`
    (GRPO trains turn-0 only). `tools` is accepted for interface parity with the
    eval ChatFn but not sent natively — Qwen's chat template renders the schemas
    into the prompt as text, and we parse the `<tool_call>` text back out.
    """
    from trl.experimental.openenv import generate_rollout_completions
    from transformers import AutoTokenizer

    tokenizer = trainer.processing_class  # used to DECODE completion ids

    # Render prompts with the model's NATIVE chat template. axolotl overrides
    # trainer.processing_class.chat_template per config (chat_template: llama3),
    # which won't emit the model's real tool-calling format. A fresh tokenizer
    # from the model dir keeps the native template (same vocab → token ids stay
    # consistent). Family-agnostic: any whitelisted model (Qwen <tool_call>,
    # Llama bare-JSON, Hermes) renders its own format here; extract_tool_calls
    # parses all of them. generate_rollout_completions has no `tools=` arg, so we
    # bake the tool schemas into the prompt text via apply_chat_template(tools=).
    render_tokenizer = tokenizer
    render_tpl_native = False
    try:
        model_path = getattr(tokenizer, "name_or_path", None)
        if model_path:
            cand = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if cand.chat_template:  # native template (renders this family's tool format)
                render_tokenizer = cand
                render_tpl_native = True
            elif _PVP_DBG:
                print(f"[PVP_DBG] render tokenizer from {model_path} has no chat_template", flush=True)
    except Exception as e:  # noqa: BLE001
        if _PVP_DBG:
            print(f"[PVP_DBG] render tokenizer load failed ({e!r}); using trainer tokenizer", flush=True)

    def chat_fn(
        config: ChatCompletionConfig,
        messages: list[ChatMessage],
        tools: list[ToolSchema] | None = None,
    ) -> ChatResult:
        wire_messages = [m.to_openai() for m in messages]
        sink.n_turns += 1
        sink.render_tpl_native = render_tpl_native

        if tools:
            # Render the tool schemas INTO the prompt text (Qwen template), then
            # send the rendered string. as_chat auto-detects False for a string,
            # so vLLM .generate tokenizes our text as-is -> model emits <tool_call>.
            tool_schemas = [t.to_openai() for t in tools]
            prompt_text = render_tokenizer.apply_chat_template(
                wire_messages, tools=tool_schemas, add_generation_prompt=True, tokenize=False
            )
            out = generate_rollout_completions(trainer, prompts=[prompt_text])[0]
            sink.tools_in_prompt = True
        else:
            out = generate_rollout_completions(trainer, prompts=[wire_messages], as_chat=True)[0]

        prompt_ids = out.get("prompt_ids", []) or []
        completion_ids = out.get("completion_ids", []) or []
        logprobs = out.get("logprobs", []) or []

        if not sink.captured:
            sink.prompt_ids = list(prompt_ids)
            sink.completion_ids = list(completion_ids)
            sink.logprobs = [float(x) for x in logprobs]
            sink.captured = True
            sink.align_ok = len(sink.completion_ids) == len(sink.logprobs)
            if _PVP_DBG:
                print(
                    f"[PVP_DBG] capture turn0: prompt_ids={len(sink.prompt_ids)} "
                    f"completion_ids={len(sink.completion_ids)} logprobs={len(sink.logprobs)} "
                    f"align={sink.align_ok} tools_in_prompt={sink.tools_in_prompt} "
                    f"render_tpl_native={sink.render_tpl_native}",
                    flush=True,
                )

        text = tokenizer.decode(completion_ids, skip_special_tokens=True).strip() if completion_ids else ""
        cleaned = strip_think_tags(text)
        tool_calls = extract_tool_calls(text) or None
        content = cleaned or None
        if tool_calls:
            sink.n_tool_call_turns += 1

        if _PVP_DBG and sink.n_turns <= 1:  # first turn of each matchup: what did the model emit?
            names = [tc.name for tc in (tool_calls or [])]
            print(
                f"[PVP_DBG] turn0 raw: n_tool_calls={len(tool_calls or [])} names={names} "
                f"text[:280]={text[:280]!r}",
                flush=True,
            )

        return ChatResult(content=content, tool_calls=tool_calls, usage=None)

    return chat_fn

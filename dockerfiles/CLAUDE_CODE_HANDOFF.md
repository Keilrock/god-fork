# Handoff Brief — G.O.D SN56 Env Tournament Training Repo

You're taking over execution on a GPU VM. A planning assistant (separate chat)
designed the rollout and verified logic; your job is to get the **smoke test
passing** by resolving a base-image dependency mess, then iterate. Work in the
VM directly — rebuild, read errors, fix, repeat. The human (Wen) is tired of
relaying long logs, so own the debug loop yourself and only surface decisions.

Communicate with Wen in casual Jakarta Indonesian ("gw/lo"), concise.

## GOAL
Get a GRPO env-tournament **training** repo that the SN56 validator runs to
train a model (the model is later pitted in PvP via SGLang — that's eval, not
your concern). Concretely: make the smoke command below run training end-to-end
(loss appears, no crash) with our custom rollout.

## WHAT WE'RE BUILDING (the edge)
The dev repo ships only a ReAct rollout (`environment_functions/*.py`, ~130
lines each, plain "Thought/Action" text). But the PR1201 eval harness scores
models on a **tool-calling + memory** protocol. That gap (ReAct training vs
tool-call eval) is the opening. Our rollout (branch `level2-clean` on
`github.com/Keilrock/god-fork`) trains the model in the SAME format it's
evaluated in: tool-calls + working/long-term memory + binary WIN/DRAW/LOSS
reward. Admin confirmed "othello + memory is what matters now" — so this is the
right direction.

## ARCHITECTURE (already decided, verified — don't re-litigate)
- **Reuse the dev harness wholesale**: `core/pvp/` has LLMBot (tool-calling +
  memory), SlotMemory, scoring (determine_outcome → WIN/DRAW/LOSS),
  run_mcts_baseline (model vs in-process pyspiel MCTS). We do NOT rewrite these.
- **Generation = colocate vLLM** via `trl.experimental.openenv.generate_rollout_completions`
  (NOT server mode / `trl vllm-serve` — that path was a multi-session dead end:
  vllm_ascend/mergekit/tokenizer hell. Colocate is what the dev reference AND the
  tournament winner use. Confirmed in dev `base_environment.yml`: vllm_mode: colocate).
- **Train turn-0 only** (later turns have per-step prefixes that break GRPO
  importance sampling — huggingface/trl#4543; the reference rollout does the same).
  Memory still matters: it steers the later turns that decide the game outcome,
  which is the turn-0 reward.
- **Tool schemas rendered as TEXT** in the prompt (Qwen chat template), parsed
  back with a brace-balanced `<tool_call>` parser. NOT TRL's native `tools=`
  (that needs transformers>=5.0, conflicts with axolotl's 4.53 pin).

## OUR FILES (branch `level2-clean`, in `dockerfiles/environment_functions/`)
- `_pvp_rollout/recording_chat_fn.py` — training ChatFn: wraps
  generate_rollout_completions, parses `<tool_call>` text, records turn-0
  prompt_ids/completion_ids/logprobs into a TurnRecord sink (capture-once).
- `_pvp_rollout/multi_env_rollout.py` — make_rollout(env)/make_reward_func();
  plays a matchup (LLMBot vs MCTS, 2 games, long-term memory carries) via the
  recording ChatFn, returns turn-0 token trace + mean binary reward. Mirrors
  run_mcts_baseline structure.
- `othello.py`, `leduc_poker.py`, `gin_rummy.py`, `liars_dice.py`,
  `goofspiel.py` — thin wrappers: `rollout_first_prompt_and_completion =
  make_rollout("<env>")` + `rollout_reward_func = make_reward_func()`.
  (text_trainer.py:118-120 resolves `<env>.rollout_first_prompt_and_completion`
  + `<env>.rollout_reward_func` for PVP env tasks.)
Logic already unit-tested (parser on real model output, reward passthrough,
capture-once turn-0). Import paths verified against the dev repo. NOT yet run on
GPU — the integration (generate_rollout_completions + LLMBot + recording) is
unproven; that's the next real test AFTER the build works.

## THE BLOCKER YOU'RE INHERITING (base image is mutable and has drifted)
`dockerfiles/standalone-text-trainer.dockerfile` does
`FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1`. The `main` tag is MUTABLE and
the maintainers have updated it since the dev repo was written. The image now
ships **transformers 5.12.0 + torch 2.8.0+cu128 + a flash_attn built for a
different torch ABI**, which breaks axolotl/trl/peft that expect older
transformers. Three errors seen IN SEQUENCE, each a different symptom of the
same drift:
1. `ImportError: cannot import name 'AutoModelForVision2Seq' from 'transformers'`
   (axolotl loader; renamed in tf 5.x)  → we pinned transformers==4.53.1
2. `flash_attn_2_cuda...so: undefined symbol: _ZN3c105...` (flash_attn built for
   a different torch) → we set flash_attention:false + tried uninstalling flash_attn
3. `ImportError: cannot import name 'is_trackio_available' from 'transformers'`
   (TRL @07b4a84e expects a NEWER transformers than 4.53.1 provides)

So it's a THREE-WAY version conflict: axolotl wants tf~4.53 (Vision2Seq,
_flash_supports_window_size), TRL @07b4a84e wants tf with is_trackio_available
(newer), and flash_attn must match torch's ABI. Patching transformers alone
won't converge.

### RECOMMENDED FIX DIRECTION (validate, don't assume)
The clean fix is to stop fighting the drifted `main` tag and pin the base image
to a **digest/tag consistent with when the dev repo last tested** (around dev
commit `a7ee3686`, "Feature/model prep container #1096", and #985 "env tasks").
Find an `axolotlai/axolotl` tag/digest whose transformers/torch/trl/flash_attn
are mutually consistent and satisfy axolotl 0.11 + TRL @07b4a84e together. Then
ONE `FROM` change + one rebuild should clear all three errors at once. Verify by:
`python -c "import axolotl, trl, transformers, peft, torch; from transformers import AutoModelForVision2Seq; from trl.experimental.openenv import generate_rollout_completions; print('ok', transformers.__version__, torch.__version__)"`
Alternatives if no clean tag exists: (a) pin a transformers version that has BOTH
the axolotl symbols AND is_trackio_available (check the 4.55–4.57 range — newer
than we assumed, since colocate may not need the old tokenizer attr that blocked
us in server mode), reinstalling a torch-matched flash_attn or disabling it;
(b) build from a pinned axolotl source commit. Prefer the digest pin.

DO NOT reintroduce `trl vllm-serve` / server mode / vllm_ascend / mergekit
patches — those were a confirmed dead end. Keep vllm_mode: colocate.

## SMOKE TEST COMMAND (the target)
Setup once: docker data-root on /ephemeral, HF_HOME=/ephemeral/hf_cache, model
at /cache/models/Qwen--Qwen2.5-3B-Instruct (snapshot_download), dummy dataset at
/ephemeral/hf_cache/datasets/smoke-001_train_data.json = `[{"dummy":"x"}]`.
Config: `core/config/base_environment.yml` (dev default: Qwen2.5-3B, vllm_mode
colocate, gpu_mem 0.25). Set `max_steps: 4` for smoke (restore 100000 after).
```
docker run --rm --gpus all -e HF_HOME=/cache -v /ephemeral/hf_cache:/cache \
  god-trainer \
  --task-id smoke-001 --model Qwen/Qwen2.5-3B-Instruct \
  --dataset env_task_dummy_dataset \
  --dataset-type '{"environment_names":["leduc_poker"]}' \
  --task-type EnvTask --file-format json --hours-to-complete 0.1 \
  --expected-repo-name smoke 2>&1 | tee /tmp/smoke.log
```
Build: `docker build -f dockerfiles/standalone-text-trainer.dockerfile -t god-trainer .`
(The dockerfile COPYs dockerfiles/environment_functions/ into the image, so our
rollout files ride along — no dockerfile edit needed for our code, only for the
base-image/deps fix.)

## SUCCESS CRITERIA
1. Build succeeds; `import axolotl, trl, transformers, peft, generate_rollout_completions` all OK.
2. Smoke runs: model loads, training starts, **our rollout actually plays**
   (LLMBot tool-calls + memory vs MCTS), loss appears and is finite (not NaN),
   completes 4 steps. Watch for: forfeit-storm (model not emitting valid
   `<tool_call>` → parser/prompt issue), token misalignment (completion_ids vs
   logprobs length), or generate_rollout_completions not accepting `tools=`
   (there's a try/except fallback in recording_chat_fn — see if it triggers).
3. Report back the loss trajectory + whether the rollout's tool-calls/memory
   actually fired (add debug prints in the rollout if needed).

## AFTER SMOKE PASSES (hand back to planning chat)
- Full run (restore max_steps), eval per env via scripts/local_environment_eval.py,
  submission gate vs baseline.
- Then "Level C": light per-env reward shaping (the winner used deep
  domain-knowledge reward shaping — gin_rummy had a 1682-line opponent-modeling
  rollout with deadwood DP etc.). That's a separate branch `level2-reward-shaping`.
- Multi-hotkey submission as a deploy-time hedge.

## ASSETS
- Code: github.com/Keilrock/god-fork branch `level2-clean` (orphan, 8 files).
- Dev repo: github.com/gradients-ai/G.O.D (clone fresh; our files overlay via the
  branch or via deploy_level2A.py).
- Old vLLM image on Docker Hub (ignore — wrong/server-mode approach).
- HuggingFace keilrockstars, GitHub Keilrock, Discord "Wen." talking to admin
  "WanderingWeights".

Own the rebuild/debug loop. Surface only decisions and the final smoke result.

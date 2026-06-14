"""Build-time patch: add missing GRPO fields to axolotl's TRLConfig schema.

axolotl 0.11's `TRLConfig` pydantic schema predates several TRL GRPO options we
rely on, so they are silently DROPPED during config validation:

  - `rollout_func`          — FQN of a custom rollout (our PvP env rollout). Without
                              it the trainer never receives our rollout, falls back
                              to default generation, and every reward is 0.0.
  - `vllm_mode`             — "colocate" vs "server". Dropped -> GRPOConfig defaults
                              to server mode and times out waiting for trl vllm-serve.
  - `vllm_enable_sleep_mode`— colocate sleep toggle.

TRL's GRPOTrainer/GRPOConfig *do* accept all three; only axolotl's schema lacks
them. This injects the fields so the values in base_environment.yml survive
validation and reach the trainer. Idempotent.
"""
import sys

SCHEMA = "/workspace/axolotl/src/axolotl/utils/schemas/trl.py"
ANCHOR = "    reward_funcs: list[str] | None = Field("
FIELDS = '''    rollout_func: str | None = Field(
        default=None,
        json_schema_extra={"description": "FQN of a custom TRL rollout function."},
    )
    vllm_mode: str | None = Field(
        default=None,
        json_schema_extra={"description": "vLLM integration mode: 'server' or 'colocate'."},
    )
    vllm_enable_sleep_mode: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Enable vLLM sleep mode (colocate)."},
    )
'''

src = open(SCHEMA).read()

if "rollout_func" in src and "vllm_mode" in src:
    print("TRLConfig schema already patched; skipping.")
    sys.exit(0)

if ANCHOR not in src:
    sys.exit(f"ERROR: anchor not found in {SCHEMA}; schema layout changed.")

src = src.replace(ANCHOR, FIELDS + ANCHOR, 1)
open(SCHEMA, "w").write(src)
print("Patched TRLConfig schema: added rollout_func, vllm_mode, vllm_enable_sleep_mode.")

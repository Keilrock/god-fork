# ── Stage: build InterCode NL2Bash fs snapshots (princeton-nlp/intercode).
# Mirrors validator-intercode.dockerfile so the training fs == the eval fs.
# Produces /intercode_fs/fs{1..4}.tar; data (queries/gold) at /opt/intercode/data/nl2bash.
FROM ubuntu:22.04 AS intercode_fs
ARG INTERCODE_COMMIT=c3e46d827cfc9d4c704ec078f7abf9f41e3191d8
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash python3 psmisc bsdmainutils cron imagemagick dnsutils git tree \
    net-tools iputils-ping coreutils curl cpio jq ca-certificates \
    findutils gawk grep sed acl attr && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/princeton-nlp/intercode /opt/intercode && \
    cd /opt/intercode && git checkout ${INTERCODE_COMMIT}
COPY dockerfiles/intercode_build_fs.sh /opt/intercode-build/build_fs.sh
RUN chmod +x /opt/intercode-build/build_fs.sh && /opt/intercode-build/build_fs.sh


FROM axolotlai/axolotl:main-py3.11-cu124-2.5.1
COPY --from=ghcr.io/astral-sh/uv:0.9.14 /uv /uvx /bin/

# InterCode assets (NL2Bash fs snapshots + dataset). The rollout restores any
# fs_{1,2,4} tar on demand; fs_3 (/workspace) is intentionally skipped in the
# rollout so reset() never wipes the axolotl runtime under /workspace.
COPY --from=intercode_fs /intercode_fs /intercode_fs
COPY --from=intercode_fs /opt/intercode/data/nl2bash /intercode_data
ENV INTERCODE_FS_ROOT=/intercode_fs \
    INTERCODE_DATA_ROOT=/intercode_data

ENV UV_SYSTEM_PYTHON=1 \
    AXOLOTL_DO_NOT_TRACK=1 \
    PYTHONPATH=/workspace:/workspace/axolotl/src

# Core deps
RUN uv pip install packaging setuptools wheel awscli pydantic \
      mlflow huggingface_hub aiohttp requests toml fastapi \
      uvicorn httpx loguru python-dotenv scipy numpy datasets \
      tenacity minio pandas tiktoken sentencepiece peft Pillow \
      PyYAML textstat langcheck detoxify \
      git+https://github.com/rayonlabs/fiber@2.4.0 \
      git+https://github.com/huggingface/trl@07b4a84e0a3c8f37a2508fe177615af019782946

RUN uv pip install --no-build-isolation vllm==0.10.2
# open_spiel (provides pyspiel) — core/pvp drives pyspiel games for the PvP rollout.
RUN uv pip install open_spiel==1.6.15
# axolotl 0.11 pins transformers==4.53.1, but TRL @07b4a84e (needed for
# trl.experimental.openenv.generate_rollout_completions) requires >=4.56.1.
# 4.56.1 satisfies both: keeps AutoModelForVision2Seq, adds is_trackio_available,
# and axolotl 0.11 doesn't reference the removed _flash_supports_window_size.
RUN uv pip install "transformers==4.56.1"
RUN uv pip uninstall flash_attn || pip uninstall -y flash_attn || true

# transformers>=4.55 removed the private `_flash_supports_window_size` flag that
# axolotl's ring-attention monkeypatch imports at module load. That code path
# (sequence-parallel flash-attn) is never used here (single GPU, flash_attention:
# false, flash_attn uninstalled), but the unconditional top-level import aborts
# `axolotl.cli.train`. Make it fall back to False — disables sliding-window in a
# path we don't execute.
RUN sed -i \
  's#^from transformers.modeling_flash_attention_utils import _flash_supports_window_size#try:\n    from transformers.modeling_flash_attention_utils import _flash_supports_window_size\nexcept ImportError:\n    _flash_supports_window_size = False#' \
  /workspace/axolotl/src/axolotl/monkeypatch/ring_attn/patch.py

WORKDIR /workspace/axolotl
RUN mkdir -p /workspace/axolotl/configs \
    /workspace/axolotl/outputs \
    /workspace/axolotl/data \
    /workspace/input_data 

COPY dockerfiles/patches/axolotl_grpo_rollout_fix.py /workspace/axolotl/src/axolotl/core/trainers/grpo/__init__.py
# axolotl 0.11's TRLConfig schema lacks rollout_func/vllm_mode/vllm_enable_sleep_mode,
# so those config keys are dropped on validation (rollout never reaches the trainer,
# vllm defaults to server mode). Inject the fields so they survive.
COPY dockerfiles/patches/add_trl_schema_fields.py /tmp/add_trl_schema_fields.py
RUN python3 /tmp/add_trl_schema_fields.py
COPY dockerfiles/environment_functions/ /workspace/axolotl/src
COPY core /workspace/core
COPY miner /workspace/miner
COPY trainer /workspace/trainer
COPY scripts /workspace/scripts
COPY core/config/base.yml /workspace/axolotl/base.yml
COPY core/config/base_grpo.yml /workspace/axolotl/base_grpo.yml
COPY core/config/base_environment.yml /workspace/axolotl/base_environment.yml

RUN chmod +x /workspace/scripts/run_text_trainer.sh /workspace/scripts/text_trainer.py

ENTRYPOINT ["/workspace/scripts/run_text_trainer.sh"]
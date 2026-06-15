#!/bin/bash
# Start vLLM OpenAI server (Qwen2.5 hermes tool parser), wait healthy, run othello eval.
# Args: MODEL_PATH SERVED_NAME TOKENIZER NUM_GAMES MCTS_SIMS LORA_PATH(optional)
set -e
MODEL_PATH="$1"; SERVED_NAME="$2"; TOKENIZER="$3"; NUM_GAMES="$4"; MCTS_SIMS="$5"; LORA_PATH="$6"; TEMP="${7:-0.0}"
LORA_ARGS=""
if [ -n "$LORA_PATH" ]; then
  LORA_ARGS="--enable-lora --lora-modules ${SERVED_NAME}=${LORA_PATH} --max-lora-rank 64"
  INFER_NAME="$SERVED_NAME"
else
  INFER_NAME="basemodel"
fi
echo "[SERVE] model=$MODEL_PATH lora=$LORA_PATH served=$INFER_NAME"
python3 -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_PATH" --served-model-name basemodel \
  --port 8000 --gpu-memory-utilization 0.85 --max-model-len 24000 \
  --enable-auto-tool-choice --tool-call-parser hermes $LORA_ARGS > /tmp/vllm.log 2>&1 &
VLLM_PID=$!
echo "[SERVE] waiting for health (pid $VLLM_PID)..."
for i in $(seq 1 90); do
  if curl -sf http://127.0.0.1:8000/health >/dev/null 2>&1; then echo "[SERVE] healthy after ${i}0s"; break; fi
  if ! kill -0 $VLLM_PID 2>/dev/null; then echo "[SERVE] vllm died:"; tail -30 /tmp/vllm.log; exit 1; fi
  sleep 10
done
python3 /workspace/othello_eval.py --model "$INFER_NAME" --tokenizer "$TOKENIZER" \
  --num-games "$NUM_GAMES" --mcts-sims "$MCTS_SIMS" --base-seed 0 --temperature "$TEMP" --time-budget 3000
kill $VLLM_PID 2>/dev/null || true

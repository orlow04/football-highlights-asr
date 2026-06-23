#!/usr/bin/env bash
# Orquestra E1 (WER) + E2 (ablação/sweep) ponta a ponta. Rode na 4090, em tmux.
# Resultados: out/e1/ e out/e2/ (CSV/JSON) + wandb se você tiver feito `wandb login`.
#
# Uso:
#   bash scripts/run_experiments.sh [AUDIO] [EVENTS] [GT_TRANSCRICAO]
#   WANDB_MODE=disabled bash scripts/run_experiments.sh   # sem wandb
set -euo pipefail

AUDIO=${1:-data/pt/jogo_corte.wav}
EVENTS=${2:-data/pt/gt_eventos.json}
GT_TRANS=${3:-data/pt/gt_transcricao.json}

# Mitiga fragmentação de VRAM (ver troubleshooting de OOM, §13).
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "== Checando a GPU antes de começar =="
command -v nvidia-smi >/dev/null && \
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv || true

echo "== E2: ablação (asr/ser/rms) + sweep α/β·k_sigma =="
python -m src.e2_sweep --audio "$AUDIO" --events "$EVENTS" --out out/e2/

echo "== E1: WER/CER + recall do léxico (base vs fine-tune) =="
python -m src.e1_asr --audio "$AUDIO" --gt "$GT_TRANS" --out out/e1/

echo "Pronto. Veja out/e1/ e out/e2/ (e o painel do wandb, se logado)."

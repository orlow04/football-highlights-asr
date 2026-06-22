# Detecção de Highlights em Narrações de Futebol (ASR + SER)

Especificação completa e protocolo experimental: [`documentacao_projeto.md`](documentacao_projeto.md).

## Setup (RTX 4090 — Ada Lovelace, CUDA 12.x)

Espaço de usuário, sem sudo/conda (runbook completo em §13 da documentação):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
uv venv --python 3.11 .venv && source .venv/bin/activate
uv pip install -r requirements.txt
# par torch+torchaudio casado POR ÚLTIMO (troque cuXXX por <= CUDA do driver: nvidia-smi)
uv pip install --force-reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu121
python -c "import torch, torchaudio; print(torch.__version__, torchaudio.__version__)"
```

As wheels `cuXXX` trazem o runtime CUDA — basta o driver NVIDIA, sem CUDA toolkit do
sistema. **Ordem importa**: deps primeiro, torch/torchaudio por último (senão o nemo
descasa o torchaudio → `libtorchaudio.so` não carrega; ver §13.1.1). `ffmpeg` no PATH
(build estático em `~/bin` se não houver sudo).

## 1. Smoke-test (rode primeiro, dentro de tmux)

Valida GPU, carga do checkpoint `.nemo` e presença de timestamps de palavra:

```bash
python smoke_test.py            # tom sintético, só exercita o setup
python smoke_test.py --audio amostra.wav
```

## 2. Pipeline ponta a ponta

```bash
# Pré-processar (se partir de vídeo): ffmpeg → 16 kHz mono
python -m src.preprocess --input jogo_bruto.mp4 --output jogo.wav

# Rodar tudo (depure no SoccerNet antes do vídeo PT)
python -m src.pipeline --audio jogo.wav --out out/ --events data/soccernet/eventos.json
```

`--mode {fusion,asr,ser}` controla a ablação do Experimento 2 (§9.3).
Estágios individuais: `python -m src.{preprocess,asr,ser,evaluate}`.

## Hiperparâmetros

Tudo em [`configs/params.yaml`](configs/params.yaml) (léxico, pesos do arousal,
α/β, k_sigma, janelas). Versione qualquer alteração — é variável de ablação.

## Estrutura

```
src/preprocess.py  asr.py  ser.py  fusion.py  detect.py  evaluate.py  pipeline.py
configs/params.yaml   smoke_test.py   data/{pt,soccernet}/
```

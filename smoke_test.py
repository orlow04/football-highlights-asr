"""Smoke-test (checklist item 1): valida ambiente da 4090 antes do GT.

Verifica, em ordem:
  1. PyTorch enxerga a GPU (4090, CUDA 12.x)
  2. NeMo importa e carrega o checkpoint .nemo
  3. transcribe(..., timestamps=True) devolve timestamps de PALAVRA

Uso:
    python smoke_test.py [--audio amostra.wav]

Se não passar --audio, sintetiza 3 s de tom para exercitar só a carga/inferência
(o texto será lixo, mas o objetivo é verificar timestamps e setup).

Rode dentro de tmux na sessão SSH; o download do checkpoint (~2.5 GB) demora.
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import wave
from pathlib import Path

MODEL = "alexandreacff/parakeet-tdt-0.6b-v3-ptBR-plus"


def check_gpu() -> bool:
    try:
        import torch
    except ImportError:
        print("✗ torch não instalado. Veja requirements.txt (CUDA 12.x).")
        return False
    ok = torch.cuda.is_available()
    name = torch.cuda.get_device_name(0) if ok else "—"
    print(f"{'✓' if ok else '✗'} CUDA disponível: {ok} | GPU: {name}")
    print(f"  torch={torch.__version__}")
    if not ok:
        print("  ⚠ sem GPU: confirme PyTorch cu121 (Ada Lovelace não roda em cu11x).")
    return ok


def _tom_3s(path: str, sr: int = 16000) -> None:
    import math

    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        frames = bytearray()
        for i in range(sr * 3):
            v = int(12000 * math.sin(2 * math.pi * 220 * i / sr))
            frames += int.to_bytes(v & 0xFFFF, 2, "little")
        w.writeframes(bytes(frames))


def check_model_and_timestamps(audio: str | None) -> bool:
    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        print("✗ NeMo não instalado: pip install -U \"nemo_toolkit[asr]\"")
        return False
    print(f"… carregando {MODEL} (download ~2.5 GB na 1ª vez)")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=MODEL)
    print("✓ checkpoint carregado")

    tmp = None
    if not audio:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        _tom_3s(tmp.name)
        audio = tmp.name
        print("  (sem --audio: usando tom sintético de 3 s)")

    hyp = model.transcribe([audio], timestamps=True)[0]
    ts = getattr(hyp, "timestamp", None) or {}
    has_word = "word" in ts and bool(ts.get("word"))
    has_seg = "segment" in ts and bool(ts.get("segment"))
    print(f"  timestamp keys: {list(ts.keys())}")
    print(f"{'✓' if has_word else '✗'} timestamps de PALAVRA presentes: {has_word}")
    print(f"{'✓' if has_seg else '—'} timestamps de SEGMENTO presentes: {has_seg}")
    if has_word:
        print(f"  exemplo: {ts['word'][:3]}")
    else:
        print("  ⚠ sem 'word' → cortes ficam na granularidade de segmento (plano B da §4.1).")
    if tmp:
        Path(tmp.name).unlink(missing_ok=True)
    return has_word or has_seg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", default=None, help="WAV 16 kHz mono curto (opcional)")
    args = ap.parse_args()
    print("== Smoke-test: ambiente Parakeet/NeMo na 4090 ==")
    gpu_ok = check_gpu()
    asr_ok = check_model_and_timestamps(args.audio)
    print("\n== Resultado ==")
    print(f"GPU: {'OK' if gpu_ok else 'FALHA/CPU'} | ASR+timestamps: {'OK' if asr_ok else 'FALHA'}")
    sys.exit(0 if (gpu_ok and asr_ok) else 1)


if __name__ == "__main__":
    main()

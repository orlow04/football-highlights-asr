"""Etapa 1 — ASR (Parakeet/NeMo) + score lexical de domínio.

Uso:
    python -m src.asr --audio jogo.wav --out out/asr.json

Saída JSON: { text, word_ts:[{word,start,end}], segment_ts, duration_s, s_kw }
onde s_kw é o vetor de score lexical na grade de 1 s.

Requer NeMo + PyTorch (CUDA 12.x na 4090). Ver requirements.txt.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import load_config, n_windows, normaliza  # noqa: E402


def carregar_modelo(model_name: str, model_file: str | None = None,
                    local_attn: bool = False,
                    att_context_size=(128, 128)):
    """Carrega o modelo NeMo.

    Com `model_file`, baixa o `.nemo` do repo HF e usa `restore_from` — caminho
    robusto. O `from_pretrained` falha neste repo porque ele empacota o snapshot
    inteiro (hparams.yaml, logs, o .nemo) e a restauração não acha `model_config.yaml`.

    Com `local_attn`, troca a self-attention global (rel_pos, O(T²) em memória) por
    atenção local de contexto limitado (`rel_pos_local_attn`), tornando a memória
    LINEAR no tempo. Essencial para áudio longo: a atenção global estoura os 24 GB
    da 4090 já em poucos minutos de áudio (matrix_ac+matrix_bd é (B,H,T,T)).
    """
    import nemo.collections.asr as nemo_asr

    if model_file:
        from huggingface_hub import hf_hub_download

        caminho = hf_hub_download(model_name, model_file)
        model = nemo_asr.models.ASRModel.restore_from(caminho)
    else:
        model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)

    if local_attn:
        model.change_attention_model(
            self_attention_model="rel_pos_local_attn",
            att_context_size=list(att_context_size),
        )
    return model


def _ts_field(item: dict, *names: str):
    for n in names:
        if n in item and item[n] is not None:
            return item[n]
    return None


def transcrever(model, audio_path: str) -> dict:
    """Transcreve com timestamps. Retorna text, word_ts, segment_ts.

    Robusto a variações de chave do NeMo ('start'/'start_offset').
    """
    hyp = model.transcribe([audio_path], timestamps=True)[0]
    ts = getattr(hyp, "timestamp", None) or {}
    if "word" not in ts:
        print("AVISO: timestamps de palavra ausentes; usando só 'segment' "
              "(cortes com granularidade de segmento).", file=sys.stderr)

    def norm_list(items):
        out = []
        for it in items or []:
            out.append({
                "word": it.get("word", it.get("segment", "")),
                "start": _ts_field(it, "start", "start_offset"),
                "end": _ts_field(it, "end", "end_offset"),
            })
        return out

    return {
        "text": getattr(hyp, "text", ""),
        "word_ts": norm_list(ts.get("word")),
        "segment_ts": norm_list(ts.get("segment")),
    }


def score_lexical(word_ts, duracao_s, cfg) -> np.ndarray:
    """Score lexical por janela, com penalização por negador próximo (§4.2)."""
    janela = cfg["audio"]["window_s"]
    lexico = {normaliza(k): v for k, v in cfg["lexicon"].items()}
    negadores = [normaliza(x) for x in cfg["negators"]]
    decai = cfg["negator_decay"]
    ctx_n = cfg["negator_context_words"]

    # chaves de uma palavra casam por substring; multi-palavra, por tokens consecutivos.
    single = {k: v for k, v in lexico.items() if " " not in k}
    multi = {tuple(k.split()): v for k, v in lexico.items() if " " in k}

    n = n_windows(duracao_s, janela)
    s = np.zeros(n)
    palavras = [(normaliza(w["word"]), w["start"]) for w in word_ts
                if w.get("start") is not None]
    for i, (w, t) in enumerate(palavras):
        peso = 0.0
        for chave, p in single.items():
            if chave in w:
                peso = max(peso, p)
        for parts, p in multi.items():
            win = tuple(pw for pw, _ in palavras[i:i + len(parts)])
            if win == parts:
                peso = max(peso, p)
        if peso == 0:
            continue
        ctx = " ".join(pw for pw, _ in palavras[max(0, i - ctx_n):i + ctx_n + 1])
        if any(ng in ctx for ng in negadores):
            peso *= decai
        idx = min(int(t // janela), n - 1)
        s[idx] += peso
    return s


def duracao_audio(audio_path: str) -> float:
    import soundfile as sf

    info = sf.info(audio_path)
    return info.frames / info.samplerate


def run(audio_path: str, config=None, model=None) -> dict:
    cfg = load_config(config)
    if model is None:
        acfg = cfg["asr"]
        model = carregar_modelo(
            acfg["model_name"], acfg.get("model_file"),
            local_attn=acfg.get("local_attn", False),
            att_context_size=acfg.get("att_context_size", (128, 128)),
        )
    trans = transcrever(model, audio_path)
    dur = duracao_audio(audio_path)
    s_kw = score_lexical(trans["word_ts"], dur, cfg)
    return {**trans, "duration_s": dur, "s_kw": s_kw.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    res = run(args.audio, config=args.config)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    print(f"OK: {args.out} | {len(res['word_ts'])} palavras | dur={res['duration_s']:.1f}s")


if __name__ == "__main__":
    main()

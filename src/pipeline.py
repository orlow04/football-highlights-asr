"""Pipeline ponta a ponta: ASR + SER → fusão → detecção → highlights.

Uso típico (depurar no SoccerNet primeiro, depois o vídeo PT):
    python -m src.pipeline --audio jogo.wav --out out/ \
        [--video jogo_bruto.mp4] [--events data/eventos.json] \
        [--mode fusion|asr|ser] [--config configs/params.yaml]

--mode controla a ablação (§9.3):
    fusion → alpha,beta do config | asr → só s_kw | ser → só s_ser
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr as asr_mod  # noqa: E402
import ser as ser_mod  # noqa: E402
from common import load_config, set_seed  # noqa: E402
from detect import detecta_highlights  # noqa: E402
from evaluate import curva_f1, _load_times  # noqa: E402
from fusion import fusao  # noqa: E402
from preprocess import extrai_audio  # noqa: E402


def _pesos(mode: str, cfg) -> tuple[float, float]:
    if mode == "asr":
        return 1.0, 0.0
    if mode == "ser":
        return 0.0, 1.0
    return cfg["fusion"]["alpha"], cfg["fusion"]["beta"]


def run(audio, out_dir, video=None, events=None, mode="fusion", config=None):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if video:
        audio = extrai_audio(video, str(out / "audio_16k.wav"),
                             sr=cfg["audio"]["sample_rate"])

    print("[1/4] ASR + léxico…")
    asr_res = asr_mod.run(audio, config=config)
    (out / "asr.json").write_text(
        json.dumps(asr_res, ensure_ascii=False, indent=2), encoding="utf-8")

    print("[2/4] SER (arousal)…")
    s_ser = np.asarray(ser_mod.run(audio, config=config)["s_ser"])
    s_kw = np.asarray(asr_res["s_kw"])

    print(f"[3/4] Fusão (mode={mode})…")
    alpha, beta = _pesos(mode, cfg)
    score = fusao(s_kw, s_ser, alpha=alpha, beta=beta)

    print("[4/4] Detecção…")
    det = detecta_highlights(score, cfg)
    resultado = {"mode": mode, "alpha": alpha, "beta": beta,
                 "score": score.tolist(), **det}

    if events:
        eventos = _load_times(events)
        resultado["deteccao"] = curva_f1(det["peaks_s"], eventos,
                                         cfg["evaluate"]["tolerances_s"])

    (out / "highlights.json").write_text(
        json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"OK: {out/'highlights.json'} | {len(det['segments'])} highlights")
    if "deteccao" in resultado:
        for m in resultado["deteccao"]:
            print(f"  τ={m['tol']:>4}s  P={m['precision']:.2f} "
                  f"R={m['recall']:.2f} F1={m['f1']:.2f}")
    return resultado


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", help="WAV 16 kHz mono (ou use --video)")
    ap.add_argument("--video", help="vídeo bruto (pré-processa antes)")
    ap.add_argument("--out", default="out")
    ap.add_argument("--events", help="JSON de eventos GT (opcional → métricas)")
    ap.add_argument("--mode", choices=["fusion", "asr", "ser"], default="fusion")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if not args.audio and not args.video:
        ap.error("forneça --audio ou --video")
    run(args.audio, args.out, video=args.video, events=args.events,
        mode=args.mode, config=args.config)


if __name__ == "__main__":
    main()

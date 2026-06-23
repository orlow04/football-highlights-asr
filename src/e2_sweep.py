"""E2 — ablação de fusão + sweep de hiperparâmetros (§9.3 da doc).

Compara, na MESMA grade de 1 s e contra o MESMO GT de eventos:
  - ASR-only (α=1, β=0)  — só o score lexical
  - SER-only (α=0, β=1)  — só o arousal acústico
  - RMS-only             — baseline ingênuo (energia pura, §9.3)
  - Fusão  α·z(s_kw)+β·z(s_ser)  varrendo α/β e k_sigma

EFICIÊNCIA: o ASR (GPU) e o SER (librosa) rodam UMA vez; as curvas s_kw/s_ser/
s_rms são cacheadas em disco e TODAS as condições/combos do sweep reusam elas —
nenhuma re-transcreve. Re-rodar o script aproveita o cache automaticamente
(use --recompute para forçar).

Saída: out/e2_results.{csv,json} (sempre) + um run por condição no wandb (opcional).

Uso:
    python -m src.e2_sweep --audio data/pt/jogo_corte.wav \
        --events data/pt/gt_eventos.json --out out/e2/
    python -m src.e2_sweep ... --recompute      # ignora o cache
    WANDB_MODE=disabled python -m src.e2_sweep ...   # sem wandb
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr as asr_mod  # noqa: E402
import ser as ser_mod  # noqa: E402
import wandb_log  # noqa: E402
from common import load_config, set_seed  # noqa: E402
from detect import detecta_highlights  # noqa: E402
from evaluate import _load_times, curva_f1  # noqa: E402
from fusion import fusao  # noqa: E402


def _curvas(audio, config, cache_dir, recompute):
    """Computa (ou recarrega do cache) as três curvas-base na grade de 1 s."""
    f = Path(cache_dir) / "curvas.json"
    if f.exists() and not recompute:
        print(f"… reusando curvas em cache: {f}")
        d = json.loads(f.read_text(encoding="utf-8"))
        return (np.array(d["s_kw"]), np.array(d["s_ser"]),
                np.array(d["s_rms"]), d["dur"])

    print("… ASR (s_kw) — etapa de GPU, roda uma vez só")
    asr_res = asr_mod.run(audio, config=config)
    s_kw, dur = np.asarray(asr_res["s_kw"]), asr_res["duration_s"]

    print("… SER (s_ser) + baseline RMS (s_rms)")
    cfg = load_config(config)
    s_ser = np.asarray(ser_mod.run(audio, config=config)["s_ser"])
    s_rms = ser_mod.curva_rms(audio, cfg)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"s_kw": s_kw.tolist(), "s_ser": s_ser.tolist(),
                             "s_rms": s_rms.tolist(), "dur": dur}),
                 encoding="utf-8")
    print(f"… curvas cacheadas em {f}")
    return s_kw, s_ser, s_rms, dur


def _cfg_k(cfg, k):
    c = copy.deepcopy(cfg)
    c["detect"]["k_sigma"] = float(k)
    return c


def _avalia(curva, cfg, k, eventos, tols) -> dict:
    det = detecta_highlights(curva, _cfg_k(cfg, k))
    linha = {"n_picos": len(det["peaks_s"]), "n_segs": len(det["segments"])}
    for m in curva_f1(det["peaks_s"], eventos, tols):
        t = int(m["tol"])
        linha[f"P@{t}"] = round(m["precision"], 4)
        linha[f"R@{t}"] = round(m["recall"], 4)
        linha[f"F1@{t}"] = round(m["f1"], 4)
    return linha


def run(audio, out_dir, events, config=None, cache_dir=None, recompute=False):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or str(out / "cache")
    eventos = _load_times(events)
    tols = cfg["evaluate"]["tolerances_s"]

    s_kw, s_ser, s_rms, _ = _curvas(audio, config, cache_dir, recompute)

    exp = cfg.get("experiments", {})
    e2 = exp.get("e2", {})
    pares = e2.get("alpha_beta", [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.3, 0.7]])
    ks = e2.get("k_sigma", [1.0, 1.5, 2.0, 2.5])
    project = exp.get("wandb_project", "football-highlights")
    group = "e2-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    # (nome, alpha, beta, curva)
    condicoes = [
        ("asr_only", 1.0, 0.0, fusao(s_kw, s_ser, 1.0, 0.0)),
        ("ser_only", 0.0, 1.0, fusao(s_kw, s_ser, 0.0, 1.0)),
        ("rms_only", None, None, s_rms),
    ]
    for a, b in pares:
        condicoes.append(("fusion", a, b, fusao(s_kw, s_ser, a, b)))

    resultados = []
    for nome, a, b, curva in condicoes:
        for k in ks:
            reg = {"exp": nome, "alpha": a, "beta": b, "k_sigma": k,
                   **_avalia(curva, cfg, k, eventos, tols)}
            resultados.append(reg)
            w = wandb_log.init(
                project, name=f"{nome}_a{a}_b{b}_k{k}", group=group,
                job_type="e2",
                config={"exp": nome, "alpha": a, "beta": b, "k_sigma": k,
                        "audio": str(audio)})
            wandb_log.summary(w, {k2: v for k2, v in reg.items() if k2 != "exp"})
            wandb_log.finish(w)
            print(f"  {nome:9} α={a} β={b} k={k} | "
                  f"F1@5={reg['F1@5']} F1@10={reg['F1@10']} F1@15={reg['F1@15']}")

    (out / "e2_results.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "e2_results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        wr.writeheader()
        wr.writerows(resultados)

    melhor = max(resultados, key=lambda r: r.get("F1@10") or 0)
    print(f"\nMelhor F1@10: {melhor['exp']} α={melhor['alpha']} "
          f"β={melhor['beta']} k={melhor['k_sigma']} → F1@10={melhor['F1@10']}")
    print(f"OK: {out/'e2_results.csv'}  ({len(resultados)} condições)")
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", default="out/e2")
    ap.add_argument("--cache", default=None, help="dir do cache de curvas")
    ap.add_argument("--recompute", action="store_true", help="ignora o cache")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.audio, args.out, args.events, config=args.config,
        cache_dir=args.cache, recompute=args.recompute)


if __name__ == "__main__":
    main()

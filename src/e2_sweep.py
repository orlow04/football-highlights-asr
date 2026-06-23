"""E2 — ablação de fusão + sweep de hiperparâmetros (Exp. 2a, jogos PT).

Compara, na MESMA grade de 1 s e contra o MESMO GT de eventos:
  - ASR-only (α=1, β=0)  — só o score lexical
  - SER-only (α=0, β=1)  — só o arousal acústico
  - RMS-only             — baseline ingênuo (CONTROLE obrigatório, §7 da estratégia)
  - Fusão  α·z(s_kw)+β·z(s_ser)  varrendo α/β e k_sigma

Hipótese (estratégia §7): F1(FUSÃO) > max(F1(ASR), F1(SER)) e > F1(RMS), em cada
jogo PT. Por isso o relatório é POR JOGO ("consistente entre jogos"), não só
agregado: cada linha tem a coluna `escopo` (nome do vídeo ou AGREGADO). O
agregado é MICRO (soma TP/FP/FN de todos os vídeos antes de P/R/F1).

EFICIÊNCIA: ASR (GPU)+SER rodam 1× por vídeo, curvas cacheadas por arquivo;
o sweep reusa o cache. `--recompute` força.

Saída: out/e2_results.{csv,json} (sempre) + runs no wandb (opcional).

Uso:
    python -m src.e2_sweep --audio data/pt/jogo_corte.wav data/pt/jogo_corte-msn.wav \
        --events data/pt/gt_eventos.json data/pt/gt_eventos-msn.json --out out/e2/
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import asr as asr_mod  # noqa: E402
import ser as ser_mod  # noqa: E402
import wandb_log  # noqa: E402
from common import load_config, set_seed  # noqa: E402
from detect import detecta_highlights  # noqa: E402
from evaluate import _load_times, avalia_deteccao  # noqa: E402
from fusion import fusao  # noqa: E402


def _curvas(audio, config, cache_dir, recompute):
    """Computa (ou recarrega do cache, por arquivo) as 3 curvas-base de 1 s."""
    f = Path(cache_dir) / f"{Path(audio).stem}.json"
    if f.exists() and not recompute:
        print(f"… reusando curvas em cache: {f}")
        d = json.loads(f.read_text(encoding="utf-8"))
        return np.array(d["s_kw"]), np.array(d["s_ser"]), np.array(d["s_rms"])

    print(f"… ASR (s_kw) de {Path(audio).name} — etapa de GPU")
    s_kw = np.asarray(asr_mod.run(audio, config=config)["s_kw"])
    print("… SER (s_ser) + baseline RMS (s_rms)")
    cfg = load_config(config)
    s_ser = np.asarray(ser_mod.run(audio, config=config)["s_ser"])
    s_rms = ser_mod.curva_rms(audio, cfg)

    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps({"s_kw": s_kw.tolist(), "s_ser": s_ser.tolist(),
                             "s_rms": s_rms.tolist()}), encoding="utf-8")
    return s_kw, s_ser, s_rms


def _cfg_k(cfg, k):
    c = copy.deepcopy(cfg)
    c["detect"]["k_sigma"] = float(k)
    return c


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _row(nome, a, b, k, escopo, n_pk, n_sg, counts, tols):
    reg = {"exp": nome, "alpha": a, "beta": b, "k_sigma": k, "escopo": escopo,
           "n_picos": n_pk, "n_segs": n_sg}
    for t in tols:
        p, r, f = _prf(*counts[t])
        reg[f"P@{t}"] = round(p, 4)
        reg[f"R@{t}"] = round(r, 4)
        reg[f"F1@{t}"] = round(f, 4)
    return reg


def run(audios, out_dir, events_list, config=None, cache_dir=None, recompute=False):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_dir = cache_dir or str(out / "cache")
    tols = cfg["evaluate"]["tolerances_s"]

    videos = []
    for audio, events in zip(audios, events_list):
        s_kw, s_ser, s_rms = _curvas(audio, config, cache_dir, recompute)
        videos.append({"stem": Path(audio).stem, "s_kw": s_kw, "s_ser": s_ser,
                       "s_rms": s_rms, "eventos": _load_times(events)})
    print(f"… {len(videos)} vídeo(s) carregado(s)")
    multi = len(videos) > 1

    exp = cfg.get("experiments", {})
    e2 = exp.get("e2", {})
    pares = e2.get("alpha_beta", [[0.7, 0.3], [0.6, 0.4], [0.5, 0.5], [0.3, 0.7]])
    ks = e2.get("k_sigma", [1.0, 1.5, 2.0, 2.5])
    project = exp.get("wandb_project", "football-highlights")
    group = "e2-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    cond = [
        ("asr_only", 1.0, 0.0, lambda v: fusao(v["s_kw"], v["s_ser"], 1.0, 0.0)),
        ("ser_only", 0.0, 1.0, lambda v: fusao(v["s_kw"], v["s_ser"], 0.0, 1.0)),
        ("rms_only", None, None, lambda v: v["s_rms"]),
    ]
    for a, b in pares:
        cond.append(("fusion", a, b,
                     lambda v, a=a, b=b: fusao(v["s_kw"], v["s_ser"], a, b)))

    resultados = []
    for nome, a, b, curva_fn in cond:
        for k in ks:
            agg = {t: [0, 0, 0] for t in tols}
            agg_pk = agg_sg = 0
            linhas = []
            for v in videos:
                det = detecta_highlights(curva_fn(v), _cfg_k(cfg, k))
                cnt = {}
                for t in tols:
                    m = avalia_deteccao(det["peaks_s"], v["eventos"], tol=t)
                    cnt[t] = (m["tp"], m["fp"], m["fn"])
                    agg[t][0] += m["tp"]
                    agg[t][1] += m["fp"]
                    agg[t][2] += m["fn"]
                agg_pk += len(det["peaks_s"])
                agg_sg += len(det["segments"])
                linhas.append(_row(nome, a, b, k, v["stem"],
                                   len(det["peaks_s"]), len(det["segments"]),
                                   cnt, tols))
            if multi:
                linhas.append(_row(nome, a, b, k, "AGREGADO", agg_pk, agg_sg,
                                   {t: tuple(agg[t]) for t in tols}, tols))
            resultados.extend(linhas)
            for reg in linhas:
                w = wandb_log.init(
                    project, name=f"{nome}_a{a}_b{b}_k{k}_{reg['escopo']}",
                    group=group, job_type="e2",
                    config={"exp": nome, "alpha": a, "beta": b, "k_sigma": k,
                            "escopo": reg["escopo"]})
                wandb_log.summary(w, {kk: vv for kk, vv in reg.items()
                                      if kk != "exp"})
                wandb_log.finish(w)

    (out / "e2_results.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "e2_results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        wr.writeheader()
        wr.writerows(resultados)

    _resumo(resultados, tols)
    print(f"\nOK: {out/'e2_results.csv'}  ({len(resultados)} linhas)")
    return resultados


def _resumo(resultados, tols) -> None:
    """Imprime, por escopo, o melhor F1@10 de cada condição e se a fusão venceu
    (hipótese 2a: fusão > asr, ser e rms)."""
    tprin = 10 if 10 in tols else tols[len(tols) // 2]
    col = f"F1@{tprin}"
    melhor = defaultdict(lambda: defaultdict(float))  # escopo -> exp -> melhor F1
    for r in resultados:
        melhor[r["escopo"]][r["exp"]] = max(melhor[r["escopo"]][r["exp"]], r[col])
    print(f"\n=== Resumo: melhor {col} por escopo × condição ===")
    for escopo, m in melhor.items():
        base = max(m.get("asr_only", 0), m.get("ser_only", 0), m.get("rms_only", 0))
        venceu = "SIM" if m.get("fusion", 0) > base else "NÃO"
        print(f"  {escopo:24} asr={m.get('asr_only',0):.3f} "
              f"ser={m.get('ser_only',0):.3f} rms={m.get('rms_only',0):.3f} "
              f"fusion={m.get('fusion',0):.3f}  → fusão vence: {venceu}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", nargs="+", required=True)
    ap.add_argument("--events", nargs="+", required=True)
    ap.add_argument("--out", default="out/e2")
    ap.add_argument("--cache", default=None)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    if len(args.audio) != len(args.events):
        ap.error("--audio e --events precisam ter o mesmo número de itens.")
    run(args.audio, args.out, args.events, config=args.config,
        cache_dir=args.cache, recompute=args.recompute)


if __name__ == "__main__":
    main()

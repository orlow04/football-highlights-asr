"""Exp. 2b — generalização do mecanismo de detecção sobre SoccerNet-Echoes.

ESCOPO (ver docs/soccernet_2b.md e docs/estrategia_validacao.md): o SoccerNet-
Echoes distribui APENAS transcrições (Whisper), sem áudio. Logo:
  - NÃO há SER nem fusão real — a detecção aqui é SÓ LÉXICA (s_kw → picos);
  - NÃO valida o ASR (transcrição é do Whisper, não humana) nem o PT (é EN/ES…);
  - valida o MECANISMO de detecção (léxico ponderado → pico → P/R/F1) com n
    grande e ground-truth de evento HUMANO (Labels-v2.json do SoccerNet).

Lê a transcrição de cada jogo, pontua um léxico de domínio em inglês por janela
de 1 s, detecta picos e avalia contra os labels de ação. Reporta por jogo +
AGREGADO (micro: soma TP/FP/FN), varrendo k_sigma.

Formato Echoes: whisper_v*/liga/temporada/jogo/{half}_asr.json,
  {"segments": {idx: [start, end, text], ...}}  (tempos em segundos).
Labels: mesma árvore liga/temporada/jogo/Labels-v2.json (gameTime "h - mm:ss",
  position em ms desde o início da metade).
PREMISSA de alinhamento: os tempos de `{half}_asr.json` e o `position/1000` dos
labels da mesma metade compartilham a origem (início da metade).

Uso:
    python -m src.soccernet_echoes \
        --echoes-root data/soccernet/whisper_v3 \
        --labels-root data/soccernet --half 1 --out out/e2b/
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import wandb_log  # noqa: E402
from common import load_config, n_windows, normaliza, set_seed  # noqa: E402
from detect import detecta_highlights  # noqa: E402
from evaluate import avalia_deteccao  # noqa: E402


def carrega_transcricao(path: str):
    """Lê {half}_asr.json → lista ordenada de (start, end, text)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    out = []
    for v in d.get("segments", {}).values():
        if isinstance(v, (list, tuple)) and len(v) >= 3:
            out.append((float(v[0]), float(v[1]), str(v[2])))
        elif isinstance(v, dict):
            st = v.get("start_time", v.get("start"))
            en = v.get("end_time", v.get("end"))
            if st is not None:
                out.append((float(st), float(en if en is not None else st),
                            str(v.get("text", ""))))
    out.sort()
    return out


def score_lexical_en(segmentos, dur, cfg) -> np.ndarray:
    """Score lexical por janela de 1 s sobre os segmentos (texto contínuo).

    Casamento por limite de palavra (\\b) evita falsos como 'goal' em 'goalkeeper';
    chaves multi-palavra ('red card') casam por substring normalizada. Negador no
    mesmo segmento atenua o peso (decai)."""
    janela = cfg["audio"]["window_s"]
    sn = cfg["soccernet"]
    lex = {normaliza(k): v for k, v in sn["lexicon_en"].items()}
    negs = [normaliza(x) for x in sn.get("negators_en", [])]
    decai = cfg["negator_decay"]
    pats = {k: re.compile(r"\b" + re.escape(k) + r"\b") for k in lex}

    n = n_windows(dur, janela)
    s = np.zeros(n)
    for start, _end, text in segmentos:
        t = normaliza(text)
        if not t:
            continue
        neg = any(ng in t for ng in negs)
        peso = 0.0
        for k, p in lex.items():
            h = len(pats[k].findall(t))
            if h:
                peso += p * h * (decai if neg else 1.0)
        if peso > 0:
            s[min(int(start // janela), n - 1)] += peso
    return s


def carrega_labels(labels_path: str, half: int, relevantes):
    data = json.loads(Path(labels_path).read_text(encoding="utf-8"))
    keep = set(relevantes) if relevantes else None
    ev = []
    for a in data.get("annotations", []):
        gt = a.get("gameTime", "")
        try:
            h = int(gt.split("-")[0].strip())
        except (ValueError, IndexError):
            continue
        if h != half or (keep is not None and a.get("label") not in keep):
            continue
        try:
            ev.append(float(a["position"]) / 1000.0)
        except (KeyError, ValueError):
            continue
    return sorted(ev)


def descobre_jogos(echoes_root: str, labels_root: str, half: int):
    """Empareja {half}_asr.json (Echoes) com o Labels-v2.json do mesmo jogo."""
    pares = []
    er = Path(echoes_root)
    for asr in sorted(er.rglob(f"{half}_asr.json")):
        rel = asr.parent.relative_to(er)
        labels = Path(labels_root) / rel / "Labels-v2.json"
        if labels.exists():
            pares.append((str(asr), str(labels), str(rel)))
    return pares


def _cfg_k(cfg, k):
    c = copy.deepcopy(cfg)
    c["detect"]["k_sigma"] = float(k)
    return c


def _prf(tp, fp, fn):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def _row(escopo, k, n_ev, n_pk, counts, tols):
    reg = {"escopo": escopo, "k_sigma": k, "n_eventos": n_ev, "n_picos": n_pk}
    for t in tols:
        p, r, f = _prf(*counts[t])
        reg[f"P@{t}"] = round(p, 4)
        reg[f"R@{t}"] = round(r, 4)
        reg[f"F1@{t}"] = round(f, 4)
    return reg


def run(echoes_root, labels_root, out_dir, half=1, config=None):
    cfg = load_config(config)
    set_seed(cfg["seed"])
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tols = cfg["evaluate"]["tolerances_s"]
    sn = cfg["soccernet"]
    ks = cfg.get("experiments", {}).get("e2", {}).get("k_sigma", [1.0, 1.5, 2.0, 2.5])
    project = cfg.get("experiments", {}).get("wandb_project", "football-highlights")

    jogos = descobre_jogos(echoes_root, labels_root, half)
    if not jogos:
        sys.exit(f"Nenhum {half}_asr.json com Labels-v2.json pareado sob "
                 f"{echoes_root} / {labels_root}.")
    print(f"… {len(jogos)} jogo(s) pareado(s) (metade {half})")

    # s_kw não depende de k_sigma → computa uma vez por jogo
    dados = []
    for asr_path, labels_path, nome in jogos:
        segs = carrega_transcricao(asr_path)
        ev = carrega_labels(labels_path, half, sn["relevant_labels"])
        if not ev:
            print(f"  (pulando {nome}: sem eventos relevantes)")
            continue
        dur = max([e for e in ev] + [s[1] for s in segs] + [1.0]) + 5
        dados.append((nome, score_lexical_en(segs, dur, cfg), ev))

    resultados = []
    for k in ks:
        agg = {t: [0, 0, 0] for t in tols}
        agg_ev = agg_pk = 0
        linhas = []
        for nome, s_kw, ev in dados:
            det = detecta_highlights(s_kw, _cfg_k(cfg, k))
            cnt = {}
            for t in tols:
                m = avalia_deteccao(det["peaks_s"], ev, tol=t)
                cnt[t] = (m["tp"], m["fp"], m["fn"])
                for i in range(3):
                    agg[t][i] += cnt[t][i]
            agg_ev += len(ev)
            agg_pk += len(det["peaks_s"])
            linhas.append(_row(nome, k, len(ev), len(det["peaks_s"]), cnt, tols))
        linhas.append(_row("AGREGADO", k, agg_ev, agg_pk,
                           {t: tuple(agg[t]) for t in tols}, tols))
        resultados.extend(linhas)
        for reg in linhas:
            w = wandb_log.init(project, name=f"e2b_{reg['escopo']}_k{k}",
                               group="e2b", job_type="e2b",
                               config={"escopo": reg["escopo"], "k_sigma": k,
                                       "half": half})
            wandb_log.summary(w, {kk: vv for kk, vv in reg.items()})
            wandb_log.finish(w)

    (out / "e2b_results.json").write_text(
        json.dumps(resultados, indent=2, ensure_ascii=False), encoding="utf-8")
    with open(out / "e2b_results.csv", "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(resultados[0].keys()))
        wr.writeheader()
        wr.writerows(resultados)

    melhor = max((r for r in resultados if r["escopo"] == "AGREGADO"),
                 key=lambda r: r.get("F1@10") or 0)
    print(f"\nMecanismo léxico no SoccerNet ({len(dados)} jogos) — melhor "
          f"AGREGADO: k={melhor['k_sigma']} F1@10={melhor['F1@10']}")
    print(f"OK: {out/'e2b_results.csv'}  ({len(resultados)} linhas)")
    return resultados


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--echoes-root", required=True,
                    help="raiz das transcrições (ex.: data/soccernet/whisper_v3)")
    ap.add_argument("--labels-root", required=True,
                    help="raiz dos Labels-v2.json (ex.: data/soccernet)")
    ap.add_argument("--half", type=int, default=1, choices=[1, 2])
    ap.add_argument("--out", default="out/e2b")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    run(args.echoes_root, args.labels_root, args.out, half=args.half,
        config=args.config)


if __name__ == "__main__":
    main()

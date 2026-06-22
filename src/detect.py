"""Etapa 4 — detecção de picos e segmentação em highlights (§7).

Janela de corte assimétrica (pre < post): o "gooool" vem DEPOIS do lance,
então preserva-se mais tempo após o pico.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def funde_sobrepostos(segs: list[list[float]]) -> list[list[float]]:
    if not segs:
        return []
    segs = sorted(segs)
    out = [list(segs[0])]
    for ini, fim in segs[1:]:
        if ini <= out[-1][1]:
            out[-1][1] = max(out[-1][1], fim)
        else:
            out.append([ini, fim])
    return out


def detecta_highlights(score, cfg) -> dict:
    score = np.asarray(score, dtype=float)
    janela = cfg["audio"]["window_s"]
    d = cfg["detect"]
    limiar = score.mean() + d["k_sigma"] * score.std()
    picos, _ = find_peaks(score, height=limiar,
                          distance=max(1, int(d["min_dist_s"] / janela)))
    picos_s = [float(p * janela) for p in picos]
    segmentos = [[max(0.0, t - d["pre_s"]), t + d["post_s"]] for t in picos_s]
    return {"peaks_s": picos_s, "segments": funde_sobrepostos(segmentos),
            "threshold": float(limiar)}

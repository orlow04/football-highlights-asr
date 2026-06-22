"""Etapa 3 — fusão tardia (late fusion) na grade de 1 s (§6).

    s = alpha * z(s_kw) + beta * z(s_ser)

Permite isolar cada modalidade: alpha=1,beta=0 → SÓ-ASR; alpha=0,beta=1 → SÓ-SER.
"""
from __future__ import annotations

import numpy as np

from common import zscore


def fusao(s_kw, s_ser, alpha: float = 0.6, beta: float = 0.4) -> np.ndarray:
    s_kw = np.asarray(s_kw, dtype=float)
    s_ser = np.asarray(s_ser, dtype=float)
    n = min(len(s_kw), len(s_ser))
    return alpha * zscore(s_kw[:n]) + beta * zscore(s_ser[:n])

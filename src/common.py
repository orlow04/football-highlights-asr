"""Utilitários compartilhados: config, normalização de texto, grade de 1 s."""
from __future__ import annotations

import unicodedata
from pathlib import Path
from typing import Any

import numpy as np
import yaml

# Raiz do repositório (este arquivo está em src/).
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "params.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normaliza(t: str) -> str:
    """minúsculas + remoção de acentos (NFD) e de pontuação, para casar léxico/negadores.

    O checkpoint Parakeet é pontuado, então os tokens vêm com pontuação grudada
    ('gol,', 'rede!', 'Obrigado.'). Removê-la torna o casamento robusto — em especial
    o multi-palavra por tokens exatos ('na rede!' → ('na','rede') == chave). Espaços
    são preservados, pois as chaves multi-palavra do léxico/negadores dependem deles.
    """
    t = t.lower()
    t = "".join(
        c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn"
    )
    return "".join(c for c in t if not unicodedata.category(c).startswith("P"))


def zscore(x: np.ndarray) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float))
    return (x - x.mean()) / (x.std() + 1e-8)


def n_windows(duracao_s: float, janela: float) -> int:
    return int(np.ceil(duracao_s / janela))


def set_seed(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
